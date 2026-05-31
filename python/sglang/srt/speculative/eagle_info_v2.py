from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)
from sglang.srt.utils.common import is_cuda, is_hip, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

logger = logging.getLogger(__name__)


def _get_req_key(req: Any) -> Any:
    return getattr(req, "rid", None) or id(req)


def _unique_request_indices(batch: ScheduleBatch) -> list[int] | None:
    seen = set()
    keep_indices = []
    for i, req in enumerate(batch.reqs):
        key = _get_req_key(req)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(i)
    return None if len(keep_indices) == len(batch.reqs) else keep_indices


def _filter_list_if_batch_sized(
    values: list[Any] | None, keep_indices: list[int], original_bs: int
) -> list[Any] | None:
    if values is None or len(values) != original_bs:
        return values
    return [values[i] for i in keep_indices]


def _filter_tensor_if_batch_sized(
    tensor: torch.Tensor | None,
    keep_indices: list[int],
    keep_indices_device: torch.Tensor,
    original_bs: int,
) -> torch.Tensor | None:
    if tensor is None or tensor.shape[0] != original_bs:
        return tensor
    if tensor.device.type == "cpu":
        return tensor[keep_indices]
    return tensor[keep_indices_device]


def _compact_duplicate_scheduler_requests(
    batch: ScheduleBatch, *, context: str
) -> tuple[list[int] | None, torch.Tensor | None]:
    keep_indices = _unique_request_indices(batch)
    if keep_indices is None:
        return None, None

    original_bs = len(batch.reqs)
    keep_indices_device = torch.tensor(
        keep_indices, dtype=torch.int64, device=batch.device
    )
    logger.debug(
        "EAGLE3 received duplicate scheduler request rows; compacting by "
        f"request id before {context}. original_bs={original_bs}, "
        f"compacted_bs={len(keep_indices)}, forward_mode={batch.forward_mode}.",
    )

    batch.reqs = [batch.reqs[i] for i in keep_indices]
    batch.decoding_reqs = _filter_list_if_batch_sized(
        getattr(batch, "decoding_reqs", None), keep_indices, original_bs
    )
    batch.multimodal_inputs = _filter_list_if_batch_sized(
        getattr(batch, "multimodal_inputs", None), keep_indices, original_bs
    )

    batch.req_pool_indices = _filter_tensor_if_batch_sized(
        batch.req_pool_indices, keep_indices, keep_indices_device, original_bs
    )
    batch.seq_lens = _filter_tensor_if_batch_sized(
        batch.seq_lens, keep_indices, keep_indices_device, original_bs
    )
    if batch.seq_lens_cpu is not None and len(batch.seq_lens_cpu) == original_bs:
        batch.seq_lens_cpu = batch.seq_lens_cpu[keep_indices]
    batch.orig_seq_lens = _filter_tensor_if_batch_sized(
        batch.orig_seq_lens, keep_indices, keep_indices_device, original_bs
    )
    batch.input_ids = _filter_tensor_if_batch_sized(
        batch.input_ids, keep_indices, keep_indices_device, original_bs
    )
    batch.output_ids = _filter_tensor_if_batch_sized(
        getattr(batch, "output_ids", None),
        keep_indices,
        keep_indices_device,
        original_bs,
    )

    batch.top_logprobs_nums = _filter_list_if_batch_sized(
        getattr(batch, "top_logprobs_nums", None), keep_indices, original_bs
    )
    batch.token_ids_logprobs = _filter_list_if_batch_sized(
        getattr(batch, "token_ids_logprobs", None), keep_indices, original_bs
    )

    batch.has_stream = any(req.stream for req in batch.reqs)
    batch.has_grammar = any(req.grammar for req in batch.reqs)
    batch.return_logprob = any(req.return_logprob for req in batch.reqs)
    if not batch.return_logprob:
        batch.top_logprobs_nums = None
        batch.token_ids_logprobs = None
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
    elif batch.seq_lens is not None:
        batch.seq_lens_sum = int(batch.seq_lens.sum().item())

    if batch.sampling_info is not None:
        sampling_bs = (
            batch.sampling_info.temperatures.shape[0]
            if batch.sampling_info.temperatures is not None
            else original_bs
        )
        if sampling_bs == original_bs:
            batch.sampling_info.filter_batch(keep_indices, keep_indices_device)

    return keep_indices, keep_indices_device


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def _infer_v2_draft_batch_size(self: EagleDraftInput, batch: ScheduleBatch) -> int:
        """Return the real, unpadded request count for EAGLE3 draft decode."""
        if batch.forward_mode.is_idle():
            return 0

        # The scheduler request list is the ownership ledger for live requests.
        # Tensor fields can be padded by CUDA graph / MLP-sync paths and must not
        # redefine the logical draft batch size.
        real_bs = batch.batch_size()

        request_level_candidates = []
        for name in ("new_seq_lens", "accept_length"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                request_level_candidates.append((name, tensor.shape[0]))

        for name in ("topk_p", "topk_index", "hidden_states"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                request_level_candidates.append((name, tensor.shape[0]))

        undersized = [
            (name, size) for name, size in request_level_candidates if size < real_bs
        ]
        if undersized:
            raise RuntimeError(
                "EAGLE3 draft input has fewer rows than live scheduler requests: "
                f"{undersized}, len(reqs)={real_bs}."
            )

        return real_bs

    def _slice_v2_draft_tensor(
        self: EagleDraftInput,
        name: str,
        real_bs: int,
        *,
        exact: bool = False,
    ) -> None:
        tensor = getattr(self, name, None)
        if tensor is None:
            return

        cur_bs = tensor.shape[0]
        if cur_bs == real_bs:
            return
        if cur_bs < real_bs or exact:
            raise RuntimeError(
                "EAGLE3 draft input tensor has an invalid batch dimension: "
                f"{name}.shape={tuple(tensor.shape)}, real_bs={real_bs}."
            )
        setattr(self, name, tensor[:real_bs])

    def normalize_v2_draft_batch(
        self: EagleDraftInput, batch: ScheduleBatch, topk: int
    ) -> int:
        """Remove CUDA-graph padding before building EAGLE3 draft metadata."""
        keep_indices, keep_indices_device = _compact_duplicate_scheduler_requests(
            batch, context="draft metadata init"
        )
        if keep_indices_device is not None:
            self.filter_batch(keep_indices_device, has_been_filtered=True)

        real_bs = self._infer_v2_draft_batch_size(batch)

        # These tensors are produced by graph-enabled target/draft forwards and
        # may carry padded rows. Slice them back to the scheduler-owned request
        # count before building draft attention metadata.
        self._slice_v2_draft_tensor("topk_p", real_bs)
        self._slice_v2_draft_tensor("topk_index", real_bs)
        self._slice_v2_draft_tensor("hidden_states", real_bs)
        self._slice_v2_draft_tensor("verified_id", real_bs)
        self._slice_v2_draft_tensor("new_seq_lens", real_bs)
        self._slice_v2_draft_tensor("accept_length", real_bs)

        if self.topk_p is not None and self.topk_p.shape[1] != topk:
            raise RuntimeError(
                "EAGLE3 draft top-k metadata mismatch: "
                f"topk_p.shape={tuple(self.topk_p.shape)}, topk={topk}."
            )
        if self.topk_index is not None and self.topk_index.shape[1] != topk:
            raise RuntimeError(
                "EAGLE3 draft top-k metadata mismatch: "
                f"topk_index.shape={tuple(self.topk_index.shape)}, topk={topk}."
            )

        if len(batch.reqs) != real_bs:
            raise RuntimeError(
                "EAGLE3 scheduler request count does not match draft state: "
                f"len(reqs)={len(batch.reqs)}, real_bs={real_bs}."
            )

        if len(batch.seq_lens) < real_bs:
            raise RuntimeError(
                "EAGLE3 scheduler batch has fewer seq_lens than draft state: "
                f"len(seq_lens)={len(batch.seq_lens)}, real_bs={real_bs}."
            )
        if len(batch.seq_lens) > real_bs:
            batch.seq_lens = batch.seq_lens[:real_bs]

        if batch.seq_lens_cpu is not None:
            if len(batch.seq_lens_cpu) < real_bs:
                raise RuntimeError(
                    "EAGLE3 scheduler batch has fewer CPU seq_lens than draft "
                    f"state: len(seq_lens_cpu)={len(batch.seq_lens_cpu)}, "
                    f"real_bs={real_bs}."
                )
            batch.seq_lens_cpu = batch.seq_lens_cpu[:real_bs]
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        else:
            batch.seq_lens_sum = int(batch.seq_lens.sum().item())

        if batch.req_pool_indices.shape[0] < real_bs:
            raise RuntimeError(
                "EAGLE3 scheduler batch has fewer req_pool_indices than draft "
                f"state: req_pool_indices.shape={tuple(batch.req_pool_indices.shape)}, "
                f"real_bs={real_bs}."
            )
        batch.req_pool_indices = batch.req_pool_indices[:real_bs]

        if batch.orig_seq_lens is not None:
            if batch.orig_seq_lens.shape[0] < real_bs:
                raise RuntimeError(
                    "EAGLE3 scheduler batch has fewer orig_seq_lens than draft "
                    f"state: orig_seq_lens.shape={tuple(batch.orig_seq_lens.shape)}, "
                    f"real_bs={real_bs}."
                )
            batch.orig_seq_lens = batch.orig_seq_lens[:real_bs]

        if batch.input_ids is not None and batch.forward_mode.is_decode_or_idle():
            if batch.input_ids.shape[0] < real_bs:
                raise RuntimeError(
                    "EAGLE3 scheduler decode batch has fewer input_ids than "
                    f"draft state: input_ids.shape={tuple(batch.input_ids.shape)}, "
                    f"real_bs={real_bs}."
                )
            batch.input_ids = batch.input_ids[:real_bs]

        return real_bs

    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Now seq_lens is correct
        batch.maybe_wait_verify_done()

        # Accumulate penalty
        # This is a relaxed version of penalties for speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        alloc_len_per_decode = get_alloc_len_per_decode()
        for r in batch.reqs:
            # Over-allocation happens here
            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x
            r.decode_batch_idx += 1
            # Pre-claim bonus slot here (like normal decode); resolve subtracts 1.
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        real_bs = self.normalize_v2_draft_batch(batch, topk)

        if not batch.forward_mode.is_idle():
            bs = real_bs

            # Assign cache locations
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=batch.input_ids.device,
            )
            # FIXME(lsyin): align with the default code path
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)

        is_standalone = draft_model_runner.spec_algorithm.is_standalone()
        batch.capture_hidden_mode = (
            CaptureHiddenMode.NULL if is_standalone else CaptureHiddenMode.LAST
        )
        batch.return_hidden_states_before_norm = not is_standalone

        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        if forward_batch.batch_size != real_bs:
            raise RuntimeError(
                "EAGLE3 draft ForwardBatch was not normalized: "
                f"forward_batch.batch_size={forward_batch.batch_size}, "
                f"real_bs={real_bs}, len(seq_lens)={len(batch.seq_lens)}, "
                f"len(reqs)={len(batch.reqs)}."
            )
        if (
            draft_model_runner.is_draft_worker
            and draft_model_runner.tp_size == 1
            and batch.global_num_tokens is not None
        ):
            # Single-rank draft execution is already local. Do not carry target
            # TP/global-token padding into draft graph selection or metadata.
            forward_batch.original_global_num_tokens_cpu = None
            forward_batch.global_num_tokens_cpu = None
            forward_batch.global_num_tokens_gpu = None
            forward_batch.global_num_tokens_for_logprob_cpu = None
            forward_batch.global_num_tokens_for_logprob_gpu = None
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict

        batch.extend_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens

        if batch.input_ids.numel() != batch.extend_num_tokens:
            raise RuntimeError(
                "EAGLE3 draft extend metadata mismatch: "
                f"input_ids.numel()={batch.input_ids.numel()}, "
                f"extend_num_tokens={batch.extend_num_tokens}, "
                f"extend_lens={batch.extend_lens}"
        )

        is_standalone = draft_model_runner.spec_algorithm.is_standalone()
        batch.capture_hidden_mode = (
            CaptureHiddenMode.NULL if is_standalone else CaptureHiddenMode.FULL
        )
        batch.return_hidden_states_before_norm = not is_standalone

        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )

        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)

        forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
        forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())

        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def _normalize_v2_verify_scheduler_metadata(
        self: EagleVerifyInput, batch: ScheduleBatch, real_bs: int
    ) -> None:
        if len(batch.reqs) != real_bs:
            raise RuntimeError(
                "EAGLE3 target verify scheduler request count does not match "
                f"draft tokens: len(reqs)={len(batch.reqs)}, real_bs={real_bs}."
            )

        sampling_info = batch.sampling_info
        sampling_bs = (
            sampling_info.temperatures.shape[0]
            if sampling_info is not None and sampling_info.temperatures is not None
            else real_bs
        )
        if sampling_bs < real_bs:
            raise RuntimeError(
                "EAGLE3 target verify sampling metadata is smaller than draft "
                f"tokens: sampling_bs={sampling_bs}, real_bs={real_bs}."
            )
        if sampling_bs > real_bs:
            keep_indices = list(range(real_bs))
            keep_indices_device = torch.arange(
                real_bs, dtype=torch.int64, device=batch.device
            )
            sampling_info.filter_batch(keep_indices, keep_indices_device)

    def _normalize_v2_verify_batch(self: EagleVerifyInput, batch: ScheduleBatch) -> int:
        if self.draft_token_num <= 0:
            raise RuntimeError(
                "EAGLE3 target verify received invalid draft_token_num: "
                f"{self.draft_token_num}."
            )
        if self.draft_token.numel() % self.draft_token_num != 0:
            raise RuntimeError(
                "EAGLE3 target verify draft tokens are not divisible by "
                f"draft_token_num: draft_token.numel()={self.draft_token.numel()}, "
                f"draft_token_num={self.draft_token_num}."
            )

        real_bs = self.draft_token.numel() // self.draft_token_num
        _compact_duplicate_scheduler_requests(batch, context="target verify")
        self._normalize_v2_verify_scheduler_metadata(batch, real_bs)

        if len(batch.seq_lens) < real_bs:
            raise RuntimeError(
                "EAGLE3 target verify scheduler batch has fewer seq_lens than "
                f"draft tokens: len(seq_lens)={len(batch.seq_lens)}, "
                f"real_bs={real_bs}."
            )
        batch.seq_lens = batch.seq_lens[:real_bs]

        if batch.seq_lens_cpu is not None:
            if len(batch.seq_lens_cpu) < real_bs:
                raise RuntimeError(
                    "EAGLE3 target verify scheduler batch has fewer CPU seq_lens "
                    f"than draft tokens: len(seq_lens_cpu)={len(batch.seq_lens_cpu)}, "
                    f"real_bs={real_bs}."
                )
            batch.seq_lens_cpu = batch.seq_lens_cpu[:real_bs]
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        else:
            batch.seq_lens_sum = int(batch.seq_lens.sum().item())

        if batch.req_pool_indices.shape[0] < real_bs:
            raise RuntimeError(
                "EAGLE3 target verify scheduler batch has fewer req_pool_indices "
                f"than draft tokens: req_pool_indices.shape="
                f"{tuple(batch.req_pool_indices.shape)}, real_bs={real_bs}."
            )
        batch.req_pool_indices = batch.req_pool_indices[:real_bs]

        if batch.orig_seq_lens is not None:
            if batch.orig_seq_lens.shape[0] < real_bs:
                raise RuntimeError(
                    "EAGLE3 target verify scheduler batch has fewer orig_seq_lens "
                    f"than draft tokens: orig_seq_lens.shape="
                    f"{tuple(batch.orig_seq_lens.shape)}, real_bs={real_bs}."
                )
            batch.orig_seq_lens = batch.orig_seq_lens[:real_bs]

        return real_bs

    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = self._normalize_v2_verify_batch(batch)

            batch.input_ids = self.draft_token

            verify_num_tokens = self.draft_token_num
            batch.extend_lens = [verify_num_tokens for _ in range(bs)]
            batch.prefix_lens = batch.seq_lens_cpu.tolist()
            batch.extend_num_tokens = bs * verify_num_tokens

            if batch.input_ids.numel() != batch.extend_num_tokens:
                raise RuntimeError(
                    "EAGLE3 target verify metadata mismatch: "
                    f"input_ids.numel()={batch.input_ids.numel()}, "
                    f"extend_num_tokens={batch.extend_num_tokens}, "
                    f"extend_lens={batch.extend_lens}"
            )

            device = batch.input_ids.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

            # Set mamba_track_indices for mamba prefix-cache state tracking
            if get_global_server_args().enable_mamba_extra_buffer():
                mapping = (
                    req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping
                )
                req_pool_idx_tensor = batch.req_pool_indices.to(
                    device=mapping.device, dtype=torch.int64
                )
                track_col_idx = torch.tensor(
                    [req.mamba_next_track_idx for req in batch.reqs],
                    dtype=torch.int64,
                    pin_memory=True,
                ).to(mapping.device, non_blocking=True)
                batch.mamba_track_indices = mapping[
                    req_pool_idx_tensor, track_col_idx
                ].to(dtype=torch.int64)
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        
        capture_mode = (
            CaptureHiddenMode.NULL
            if target_worker.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.capture_hidden_mode = capture_mode
        batch.return_hidden_states_before_norm = not capture_mode == CaptureHiddenMode.NULL
        batch.spec_info = self

        forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)
        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    forward_batch
                )

        return forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            accept_length = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # Apply penalty
        # This is a relaxed version of penalties for speculative decoding.
        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.draft_token_num, dim=0
                )
            )
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.draft_token_num, dim=0
                )
            )

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(accept_length, src=0)

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
