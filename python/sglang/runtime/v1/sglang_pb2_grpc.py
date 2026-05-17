"""gRPC service bindings for the SGLang runtime proto."""

from __future__ import annotations

import grpc

from . import sglang_pb2 as sglang__pb2


class DraftForwardServiceStub:
    """Client stub for the DraftForward RPC service."""

    def __init__(self, channel):
        self.DraftForward = channel.unary_unary(
            "/sglang.runtime.v1.DraftForwardService/DraftForward",
            request_serializer=sglang__pb2.DraftForwardRequest.SerializeToString,
            response_deserializer=sglang__pb2.DraftForwardResponse.FromString,
        )
        self.DraftForwardStream = channel.stream_stream(
            "/sglang.runtime.v1.DraftForwardService/DraftForwardStream",
            request_serializer=sglang__pb2.DraftForwardRequest.SerializeToString,
            response_deserializer=sglang__pb2.DraftForwardResponse.FromString,
        )


class DraftForwardServiceServicer:
    """Base class for DraftForward service handlers."""

    def DraftForward(self, request, context):
        raise NotImplementedError("DraftForward is not implemented.")

    def DraftForwardStream(self, request_iterator, context):
        raise NotImplementedError("DraftForwardStream is not implemented.")


def add_DraftForwardServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "DraftForward": grpc.unary_unary_rpc_method_handler(
            servicer.DraftForward,
            request_deserializer=sglang__pb2.DraftForwardRequest.FromString,
            response_serializer=sglang__pb2.DraftForwardResponse.SerializeToString,
        ),
        "DraftForwardStream": grpc.stream_stream_rpc_method_handler(
            servicer.DraftForwardStream,
            request_deserializer=sglang__pb2.DraftForwardRequest.FromString,
            response_serializer=sglang__pb2.DraftForwardResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "sglang.runtime.v1.DraftForwardService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
