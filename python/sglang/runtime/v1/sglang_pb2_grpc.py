"""gRPC service bindings for the SGLang runtime proto."""

from __future__ import annotations

import grpc

from . import sglang_pb2 as sglang__pb2


class TliSpeculativeServiceStub:
    """Client stub for the TLI speculative DraftForward RPC."""

    def __init__(self, channel):
        self.DraftForward = channel.unary_unary(
            "/sglang.runtime.v1.TliSpeculativeService/DraftForward",
            request_serializer=sglang__pb2.TliDraftRequest.SerializeToString,
            response_deserializer=sglang__pb2.TliDraftResponse.FromString,
        )


class TliSpeculativeServiceServicer:
    """Base class for TLI speculative DraftForward service handlers."""

    def DraftForward(self, request, context):
        raise NotImplementedError("DraftForward is not implemented.")


def add_TliSpeculativeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "DraftForward": grpc.unary_unary_rpc_method_handler(
            servicer.DraftForward,
            request_deserializer=sglang__pb2.TliDraftRequest.FromString,
            response_serializer=sglang__pb2.TliDraftResponse.SerializeToString,
        )
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "sglang.runtime.v1.TliSpeculativeService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))

