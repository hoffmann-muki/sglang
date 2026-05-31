import copy
import contextlib
import logging
import time
import traceback
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    SingleRankDraftGroup,
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    maybe_detect_nan,
    maybe_detect_oob,
    single_rank_draft_context,
    select_top_k_tokens,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BroadcastedEagleDraftResult:
    ok: bool
    payload: Optional[dict] = None
    draft_proposal_time: float = 0.0
    error: Optional[str] = None


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self._single_rank_draft_mode = (
            server_args.speculative_draft_tp_size == 1 and server_args.tp_size > 1
        )
        self._is_root_draft_rank = tp_rank == 0
        self._local_draft_tp_group = (
            SingleRankDraftGroup(target_worker.world_group)
            if self._single_rank_draft_mode
            else None
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        if self._single_rank_draft_mode:
            draft_args = copy.copy(server_args)
            draft_args.tp_size = 1
            draft_args.ep_size = 1
            draft_args.enable_dp_attention = False
            draft_args.speculative_draft_tp_size = 1
            draft_args.context_length = target_worker.model_runner.model_config.context_len
            draft_args.disable_cuda_graph = True
            if self._is_root_draft_rank:
                with (
                    single_rank_draft_context(self._local_draft_tp_group),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    self.draft_worker = TpModelWorker(
                        server_args=draft_args,
                        gpu_id=gpu_id,
                        tp_rank=0,
                        pp_rank=0,  # FIXME
                        dp_rank=dp_rank,
                        moe_ep_rank=moe_ep_rank,
                        attn_cp_rank=attn_cp_rank,
                        moe_dp_rank=moe_dp_rank,
                        nccl_port=nccl_port,
                        is_draft_worker=True,
                        req_to_token_pool=self.req_to_token_pool,
                        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                        memory_pool_config=target_worker.model_runner.memory_pool_config,
                    )
                self.draft_runner = self.draft_worker.model_runner
            else:
                self.draft_worker = None
                self.draft_runner = None
        else:
            # Init draft worker
            if (
                server_args.enable_dp_attention
                and self.speculative_algorithm.is_eagle3()
            ):
                ctx = draft_tp_context(get_attention_tp_group())
            else:
                ctx = empty_context()
            with (
                ctx
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                # Init draft worker
                self.draft_worker = TpModelWorker(
                    server_args=server_args,
                    gpu_id=gpu_id,
                    tp_rank=tp_rank,
                    pp_rank=0,  # FIXME
                    dp_rank=dp_rank,
                    moe_ep_rank=moe_ep_rank,
                    attn_cp_rank=attn_cp_rank,
                    moe_dp_rank=moe_dp_rank,
                    nccl_port=nccl_port,
                    is_draft_worker=True,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    memory_pool_config=target_worker.model_runner.memory_pool_config,
                )

            # Alias for better readability
            self.draft_runner = self.draft_worker.model_runner
        self.eagle_use_aux_hidden_state = False
        if self.draft_runner is not None and self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        if self.draft_runner is not None:
            self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            empty_context
            if self._single_rank_draft_mode
            else (draft_tp_context if server_args.enable_dp_attention else empty_context)
        )
        with self.draft_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @contextmanager
    def draft_context(self):
        if self.draft_worker is None or self._single_rank_draft_mode:
            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                yield
            return

        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            yield

    def _get_draft_embedding_capacity(self) -> Optional[int]:
        if self.draft_runner is None:
            return None

        model = self.draft_runner.model
        embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed_tokens is None:
            embed_tokens = getattr(model, "embed_tokens", None)
        weight = getattr(embed_tokens, "weight", None)
        if weight is not None:
            return weight.shape[0]
        return getattr(embed_tokens, "num_embeddings", None)

    def _validate_draft_prefill_token_ids(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> None:
        if next_token_ids.numel() != len(batch.extend_lens):
            raise RuntimeError(
                "EAGLE3 draft prefill received inconsistent next-token metadata. "
                f"next_token_ids.numel()={next_token_ids.numel()}, "
                f"batch_size={len(batch.extend_lens)}, "
                f"extend_lens={batch.extend_lens}"
            )

        if batch.input_ids is None or batch.input_ids.numel() == 0:
            return

        if target_hidden_states.shape[0] != batch.input_ids.numel():
            raise RuntimeError(
                "EAGLE3 draft prefill hidden-state row count does not match "
                "the draft input token count. "
                f"hidden_states.shape={tuple(target_hidden_states.shape)}, "
                f"input_ids.shape={tuple(batch.input_ids.shape)}, "
                f"extend_lens={batch.extend_lens}, "
                f"seq_lens={batch.seq_lens_cpu.tolist() if hasattr(batch.seq_lens_cpu, 'tolist') else batch.seq_lens_cpu}"
            )

        draft_embedding_capacity = self._get_draft_embedding_capacity()
        if draft_embedding_capacity is None:
            return

        ids_min = int(batch.input_ids.min().item())
        ids_max = int(batch.input_ids.max().item())
        next_min = int(next_token_ids.min().item()) if next_token_ids.numel() else None
        next_max = int(next_token_ids.max().item()) if next_token_ids.numel() else None

        if ids_min < 0 or ids_max >= draft_embedding_capacity:
            draft_config = self.draft_runner.model_config
            draft_hf_config = draft_config.hf_config
            target_config = self.target_worker.model_runner.model_config
            raise RuntimeError(
                "EAGLE3 draft prefill received token ids outside the draft "
                "embedding vocabulary. This usually means the draft speculator "
                "config/tokenizer vocabulary does not match the target tokenizer, "
                "or target token ids need to be translated before the draft "
                "prefill forward. "
                f"input_ids_range=[{ids_min}, {ids_max}], "
                f"next_token_ids_range=[{next_min}, {next_max}], "
                f"draft_embedding_capacity={draft_embedding_capacity}, "
                f"target_model_vocab_size={target_config.vocab_size}, "
                f"draft_model_config_vocab_size={draft_config.vocab_size}, "
                f"draft_hf_vocab_size={getattr(draft_hf_config, 'vocab_size', None)}, "
                f"draft_hf_draft_vocab_size={getattr(draft_hf_config, 'draft_vocab_size', None)}, "
                f"batch_size={len(batch.extend_lens)}, "
                f"extend_lens={batch.extend_lens}, "
                f"seq_lens={batch.seq_lens_cpu.tolist() if hasattr(batch.seq_lens_cpu, 'tolist') else batch.seq_lens_cpu}"
            )

    def _serialize_verify_input(self, spec_info: EagleVerifyInput) -> dict:
        return {
            "draft_token": spec_info.draft_token.detach().cpu(),
            "custom_mask": spec_info.custom_mask.detach().cpu(),
            "positions": spec_info.positions.detach().cpu(),
            "retrieve_index": spec_info.retrieve_index.detach().cpu(),
            "retrieve_next_token": spec_info.retrieve_next_token.detach().cpu(),
            "retrieve_next_sibling": spec_info.retrieve_next_sibling.detach().cpu(),
            "retrieve_cum_len": (
                None if spec_info.retrieve_cum_len is None else spec_info.retrieve_cum_len.detach().cpu()
            ),
            "spec_steps": spec_info.spec_steps,
            "topk": spec_info.topk,
            "draft_token_num": spec_info.draft_token_num,
            "capture_hidden_mode": spec_info.capture_hidden_mode,
            "seq_lens_sum": spec_info.seq_lens_sum,
            "seq_lens_cpu": (
                None if spec_info.seq_lens_cpu is None else spec_info.seq_lens_cpu.detach().cpu()
            ),
        }

    def _deserialize_verify_input(self, payload: dict) -> EagleVerifyInput:
        def to_device(tensor: Optional[torch.Tensor]):
            return None if tensor is None else tensor.to(self.device)

        return EagleVerifyInput(
            draft_token=to_device(payload["draft_token"]),
            custom_mask=to_device(payload["custom_mask"]),
            positions=to_device(payload["positions"]),
            retrieve_index=to_device(payload["retrieve_index"]),
            retrieve_next_token=to_device(payload["retrieve_next_token"]),
            retrieve_next_sibling=to_device(payload["retrieve_next_sibling"]),
            retrieve_cum_len=to_device(payload["retrieve_cum_len"]),
            spec_steps=payload["spec_steps"],
            topk=payload["topk"],
            draft_token_num=payload["draft_token_num"],
            capture_hidden_mode=payload["capture_hidden_mode"],
            seq_lens_sum=payload["seq_lens_sum"],
            seq_lens_cpu=to_device(payload["seq_lens_cpu"]),
        )

    def _broadcast_draft_result(self, payload: list[_BroadcastedEagleDraftResult]) -> _BroadcastedEagleDraftResult:
        world_group = self.target_worker.world_group
        return broadcast_pyobj(
            payload,
            world_group.rank,
            world_group.cpu_group,
            src=world_group.first_rank,
        )[0]

    def _sync_single_rank_draft_extend(
        self,
        phase: str,
        error: Optional[str] = None,
        raise_on_error: bool = True,
    ):
        if not self._single_rank_draft_mode:
            return

        payload = (
            [_BroadcastedEagleDraftResult(ok=error is None, error=error)]
            if self._is_root_draft_rank
            else []
        )
        broadcasted = self._broadcast_draft_result(payload)
        if raise_on_error and not broadcasted.ok:
            raise RuntimeError(
                f"EAGLE3 single-rank draft {phase} failed on root rank:\n"
                f"{broadcasted.error}"
            )

    def _draft_single_rank(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        draft_input.normalize_v2_draft_batch(batch, self.topk)

        if self._is_root_draft_rank:
            try:
                forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
                    self.req_to_token_pool,
                    batch,
                    self.cuda_graph_runner,
                    self.draft_runner,
                    self.topk,
                    self.speculative_num_steps,
                )

                if can_cuda_graph:
                    parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                        forward_batch,
                    )
                else:
                    if (
                        not forward_batch.forward_mode.is_idle()
                        and self.speculative_num_steps > 1
                    ):
                        self.draft_attn_backend.init_forward_metadata(forward_batch)
                        
                    parent_list, top_scores_index, draft_tokens = self.draft_forward(
                        forward_batch
                    )

                if batch.forward_mode.is_idle():
                    spec_info = EagleVerifyInput.create_idle_input(
                        self.topk,
                        self.speculative_num_steps,
                        self.speculative_num_draft_tokens,
                    )
                else:
                    tree_mask_buf, position_buf = (
                        self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
                    )
                    (
                        tree_mask,
                        position,
                        retrieve_index,
                        retrieve_next_token,
                        retrieve_next_sibling,
                        draft_tokens,
                    ) = build_tree_kernel_efficient(
                        draft_input.verified_id,
                        parent_list,
                        top_scores_index,
                        draft_tokens,
                        batch.seq_lens,
                        batch.seq_lens_sum,
                        self.topk,
                        self.speculative_num_steps,
                        self.speculative_num_draft_tokens,
                        self.tree_mask_mode,
                        tree_mask_buf,
                        position_buf,
                    )
                    spec_info = EagleVerifyInput(
                        draft_token=draft_tokens,
                        custom_mask=tree_mask,
                        positions=position,
                        retrieve_index=retrieve_index,
                        retrieve_next_token=retrieve_next_token,
                        retrieve_next_sibling=retrieve_next_sibling,
                        retrieve_cum_len=None,
                        spec_steps=self.speculative_num_steps,
                        topk=self.topk,
                        draft_token_num=self.speculative_num_draft_tokens,
                        capture_hidden_mode=None,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )

                payload = [
                    _BroadcastedEagleDraftResult(
                        ok=True,
                        payload=self._serialize_verify_input(spec_info),
                    )
                ]
            except Exception:
                logger.exception("EAGLE3 single-rank draft failed on root rank.")
                payload = [
                    _BroadcastedEagleDraftResult(
                        ok=False,
                        error=traceback.format_exc(),
                    )
                ]

            broadcasted = self._broadcast_draft_result(payload)
            if not broadcasted.ok:
                raise RuntimeError(
                    "EAGLE3 single-rank draft failed on root rank:\n"
                    f"{broadcasted.error}"
                )
            return self._deserialize_verify_input(broadcasted.payload)

        broadcasted = self._broadcast_draft_result([])
        if not broadcasted.ok:
            raise RuntimeError(
                "EAGLE3 single-rank draft failed on root rank:\n"
                f"{broadcasted.error}"
            )
        return self._deserialize_verify_input(broadcasted.payload)

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            if self._single_rank_draft_mode:
                tp_group = self.target_worker.model_runner.tp_group
                embed = tp_group.all_gather(embed, dim=0)

                needs_target_head = False
                if self._is_root_draft_rank and self.draft_runner is not None:
                    needs_target_head = (
                        hasattr(self.draft_runner.model, "load_lm_head_from_target")
                        and self.draft_runner.model.load_lm_head_from_target
                    )
                needs_target_head = broadcast_pyobj(
                    [needs_target_head] if self._is_root_draft_rank else [],
                    self.target_worker.world_group.rank,
                    self.target_worker.world_group.cpu_group,
                    src=self.target_worker.world_group.first_rank,
                )[0]
                if needs_target_head:
                    head = tp_group.all_gather(head, dim=0)

                if self.draft_worker is None:
                    return

                if needs_target_head:
                    self.draft_runner.model.set_embed_and_head(embed, head)
                else:
                    self.draft_runner.model.set_embed(embed)

                # grab hot token ids
                if self.draft_runner.model.hot_token_id is not None:
                    self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                        embed.device
                    )
                return

            if self.draft_worker is None:
                return

            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.draft_worker is None:
                return

            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        if self.draft_worker is None:
            self.draft_attn_backend = None
            self.draft_extend_attn_backend = None
            return

        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.draft_worker is None:
            return

        if self.server_args.disable_cuda_graph:
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = _is_cuda and (
            isinstance(self.draft_extend_attn_backend, TritonAttnBackend)
            or isinstance(self.draft_extend_attn_backend, TRTLLMMLABackend)
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def draft(self, batch: ScheduleBatch):
        if self._single_rank_draft_mode:
            return self._draft_single_rank(batch)

        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for 1-step draft,
                # `draft_forward` only does sample in this case.
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        if self._single_rank_draft_mode and not self._is_root_draft_rank:
            self._sync_single_rank_draft_extend("prefill")
            return None

        try:
            # Construct input_ids
            if not batch.forward_mode.is_idle():
                pt = 0
                for i, extend_len in enumerate(batch.extend_lens):
                    input_ids = batch.input_ids[pt : pt + extend_len]
                    batch.input_ids[pt : pt + extend_len] = torch.cat(
                        (input_ids[1:], next_token_ids[i].reshape(1))
                    )
                    pt += extend_len

            # Construct spec_info
            next_draft_input = EagleDraftInput(
                hidden_states=target_hidden_states,
                verified_id=next_token_ids,
                new_seq_lens=batch.seq_lens,
                # draft mode is same with decode mode, only 1 token per req
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )

            batch.spec_info = next_draft_input

            is_standalone = self.speculative_algorithm.is_standalone()
            batch.capture_hidden_mode = (
                CaptureHiddenMode.NULL if is_standalone else CaptureHiddenMode.LAST
            )
            batch.return_hidden_states_before_norm = not is_standalone

            forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
            forward_batch.return_logprob = False

            if mm_input_embeds is not None:
                forward_batch.mm_input_embeds = mm_input_embeds

            self._validate_draft_prefill_token_ids(
                batch, target_hidden_states, next_token_ids
            )

            logits_output = self.draft_runner.forward(forward_batch).logits_output
            maybe_detect_nan(
                logits_output.next_token_logits, "draft_extend_for_prefill"
            )

            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
                probs, self.topk, dim=-1
            )
            next_draft_input.hidden_states = logits_output.hidden_states
        except Exception:
            if self._single_rank_draft_mode and self._is_root_draft_rank:
                self._sync_single_rank_draft_extend(
                    "prefill", traceback.format_exc(), raise_on_error=False
                )
            raise

        if self._single_rank_draft_mode and self._is_root_draft_rank:
            self._sync_single_rank_draft_extend("prefill")

        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        next_draft_input = batch_result.next_draft_input

        if self._single_rank_draft_mode and not self._is_root_draft_rank:
            batch.spec_info = next_draft_input
            self._sync_single_rank_draft_extend("decode")
            return

        original_forward_mode = batch.forward_mode
        original_input_ids = batch.input_ids
        original_out_cache_loc = batch.out_cache_loc
        original_extend_lens = batch.extend_lens
        original_prefix_lens = batch.prefix_lens
        original_extend_num_tokens = batch.extend_num_tokens
        original_capture_hidden_mode = batch.capture_hidden_mode
        original_return_hidden_states_before_norm = (
            batch.return_hidden_states_before_norm
        )

        try:
            # Batch 2: Draft extend
            draft_input = EagleDraftInput(
                hidden_states=batch_result.logits_output.hidden_states,
                num_tokens_per_req=self.speculative_num_steps + 1,
                num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
            )
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )

            # Prepare for draft extend in a separate stream
            with self.plan_stream_ctx:
                forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                    batch,
                    batch_result.next_token_ids,
                    self.speculative_num_draft_tokens,
                    self.draft_runner,
                    self.cuda_graph_runner_for_draft_extend,
                )

            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )

            if forward_batch.spec_info.accept_length is None:
                forward_batch.spec_info.accept_length = batch_result.accept_lens

            # Run draft extend batch in the main compute stream
            can_cuda_graph = (
                self.cuda_graph_runner_for_draft_extend
                and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
            )
            if can_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output

            maybe_detect_nan(
                draft_logits_output.next_token_logits,
                f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
            )

            # Reorganize the spec info for the next batch
            draft_logits_output.next_token_logits = (
                draft_logits_output.next_token_logits[select_index]
            )
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
            probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_hidden_states = draft_logits_output.hidden_states

            # Construct the return values
            (
                next_draft_input.topk_p,
                next_draft_input.topk_index,
                next_draft_input.hidden_states,
            ) = (
                ret_topk_p,
                ret_topk_index,
                ret_hidden_states,
            )
            batch.spec_info = next_draft_input
        except Exception:
            if self._single_rank_draft_mode and self._is_root_draft_rank:
                self._sync_single_rank_draft_extend(
                    "decode", traceback.format_exc(), raise_on_error=False
                )
            raise
        finally:
            # The draft-extend forward temporarily repurposes the scheduler batch
            # as DRAFT_EXTEND_V2. Restore scheduler-visible decode metadata so TP
            # ranks that did not run the single-rank draft path stay identical.
            batch.forward_mode = original_forward_mode
            batch.input_ids = original_input_ids
            batch.out_cache_loc = original_out_cache_loc
            batch.extend_lens = original_extend_lens
            batch.prefix_lens = original_prefix_lens
            batch.extend_num_tokens = original_extend_num_tokens
            batch.capture_hidden_mode = original_capture_hidden_mode
            batch.return_hidden_states_before_norm = (
                original_return_hidden_states_before_norm
            )

        if self._single_rank_draft_mode and self._is_root_draft_rank:
            self._sync_single_rank_draft_extend("decode")


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self._single_rank_draft_mode = (
            server_args.speculative_draft_tp_size == 1 and server_args.tp_size > 1
        )
        self._is_root_draft_rank = tp_rank == 0
        self._local_draft_tp_group = (
            SingleRankDraftGroup(target_worker.world_group)
            if self._single_rank_draft_mode
            else None
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def draft_runner(self):
        return self._draft_worker.draft_runner

    @property
    def model_runner(self):
        return self._draft_worker.draft_runner

    @property
    def is_v2_draft_worker(self):
        return True

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, batch: ScheduleBatch):
        # EAGLE v2 keeps ScheduleBatch as the canonical mutable scheduler state.
        # ForwardBatch.init_new is responsible for materializing ModelWorkerBatch
        # when the lower-level runner needs it.

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill: EAGLE3 needs full target hidden states.
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch.return_hidden_states_before_norm = (
                target_capture_mode != CaptureHiddenMode.NULL
            )
            
            batch_output = self.target_worker.forward_batch_generation(batch)

            target_hidden_states = getattr(batch_output.logits_output, "hidden_states", None)
            if target_capture_mode != CaptureHiddenMode.NULL and target_hidden_states is None:
                raise RuntimeError(
                    "EAGLE3 target prefill did not return hidden_states. "
                    f"capture_hidden_mode={target_capture_mode}, "
                    f"logits_output_type={type(batch_output.logits_output)}"
            )

            # Draft prefill
            with self.draft_worker.draft_context():
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        target_hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )

            return batch_output
        
        else:
            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.spec_hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                )

            with self.draft_worker.draft_context():
                verify_input: EagleVerifyInput = self.draft_worker.draft(batch)

            assert verify_input.is_verify_input()

            if self.plan_stream:
                self._draft_done_event = torch.get_device_module(self.device).Event()
                self._draft_done_event.record()

            batch.spec_info = verify_input

            batch_output = self.verify(batch)

            with self.draft_worker.draft_context():
                self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output

    def verify(self, batch: ScheduleBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        target_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.capture_hidden_mode = target_capture_mode
        batch.return_hidden_states_before_norm = (
            target_capture_mode != CaptureHiddenMode.NULL
        )

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            # Wait for the draft CUDA graph to finish before plan_stream
            # begins its work. Using an event is more targeted than
            # wait_stream(main_stream) — it only waits for draft GPU
            # work, not all queued main_stream operations.
            if self.plan_stream and hasattr(self, "_draft_done_event"):
                self.plan_stream.wait_event(self._draft_done_event)
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_length

        # Update mamba state for hybrid GDN models after verification
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_length, accept_index, bs
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not batch.forward_mode.is_idle():
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_length, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_length,
                verified_id,
                self.speculative_num_draft_tokens,
            )
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            routed_experts_output=forward_batch_output.routed_experts_output,
        )

    def _mamba_verify_update(
        self,
        batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        accept_length: torch.Tensor,
        accept_index: torch.Tensor,
        bs: int,
    ):
        """Update mamba state for hybrid GDN models after verification."""
        # Calculate accepted_steps for mamba state update
        # Include the bonus token (+1)
        accepted_length_with_bonus = accept_length
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:
                raise ValueError("Spec v2 currently only supports topk = 1.")

            accepted_indices_offset = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accepted_length_with_bonus.dtype,
                device=accepted_length_with_bonus.device,
            )
            accepted_steps = accepted_length_with_bonus - 1

            if batch.mamba_track_indices is not None:
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accepted_length_with_bonus
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accepted_length_with_bonus.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),
                )
            else:
                mamba_steps_to_track = None

            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                accepted_steps=accepted_steps,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            accept_length: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
