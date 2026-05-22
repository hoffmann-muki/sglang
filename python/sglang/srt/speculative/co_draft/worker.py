"""Colocated AR+dLLM draft worker infrastructure."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sglang.srt.speculative.asymmetric_tli_worker import AsymmetricTLIWorker
from sglang.srt.speculative.co_draft.bridge import (
    CoDraftBridge,
    UnimplementedCoDraftBridge,
)
from sglang.srt.speculative.co_draft.executor import (
    ArTliDraftExecutor,
    CoDraftStrategy,
    DllmDraftExecutor,
)
from sglang.srt.speculative.co_draft.tp import LocalDraftTpPlan
from sglang.srt.speculative.tli_worker import TLIWorker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CoDraftExecutors:
    ar: ArTliDraftExecutor
    dllm: DllmDraftExecutor
    bridge: CoDraftBridge
    strategy: CoDraftStrategy


class CoDraftWorkerMixin:
    """Shared co-draft metadata initialization for symmetric/asymmetric workers."""

    def _finish_co_draft_init(self) -> None:
        self.co_draft = self._init_co_draft_executors()

    def _init_co_draft_executors(self) -> CoDraftExecutors:
        server_args = self.server_args
        ar_tp_plan = LocalDraftTpPlan(
            name="AR co-draft",
            target_tp_size=server_args.tp_size,
            draft_tp_size=server_args.speculative_draft_tp_size,
        )
        dllm_tp_plan = LocalDraftTpPlan(
            name="dLLM co-draft",
            target_tp_size=server_args.tp_size,
            draft_tp_size=server_args.codraft_dllm_draft_tp_size,
        )
        executors = CoDraftExecutors(
            ar=ArTliDraftExecutor(tp_plan=ar_tp_plan, owner=self),
            dllm=DllmDraftExecutor(
                tp_plan=dllm_tp_plan,
                model_path=server_args.codraft_dllm_draft_model_path,
                tokenizer_path=server_args.codraft_dllm_tokenizer_path,
                algorithm=server_args.codraft_dllm_algorithm,
                algorithm_config=server_args.codraft_dllm_algorithm_config,
            ),
            bridge=UnimplementedCoDraftBridge(),
            strategy=server_args.codraft_strategy,
        )
        if executors.strategy != "ar_only":
            raise NotImplementedError(
                f"CO_DRAFT strategy {executors.strategy!r} is not implemented yet. "
                "Use 'ar_only' until a strategy adapter is wired."
            )
        logger.info(
            "CO_DRAFT initialized: strategy=%s, ar_tp=%s/%s, dllm_tp=%s/%s.",
            executors.strategy,
            ar_tp_plan.draft_tp_size,
            ar_tp_plan.target_tp_size,
            dllm_tp_plan.draft_tp_size,
            dllm_tp_plan.target_tp_size,
        )
        return executors


class CoDraftWorker(CoDraftWorkerMixin, AsymmetricTLIWorker):
    """Asymmetric colocated co-draft worker.

    The first executable strategy routes through the existing AR/TLI path while
    carrying a dLLM executor descriptor with its own TP plan. Strategy adapters
    can later consume both executors and emit one target verification payload.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._finish_co_draft_init()


class SymmetricCoDraftWorker(CoDraftWorkerMixin, TLIWorker):
    """Symmetric colocated co-draft worker where the AR draft uses target TP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._finish_co_draft_init()
