#!/usr/bin/env python3
"""Minimal raw-bytes gRPC streaming benchmark.

Run this file as a server on one node and as a client on another. It avoids
protobuf codegen on purpose so the measurement is mostly the gRPC transport.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent import futures
from typing import Iterator

import grpc


METHOD = "/bench.Stream/PingPong"


def _identity(x: bytes) -> bytes:
    return x


def serve(args: argparse.Namespace) -> None:
    response = os.urandom(args.response_bytes)

    def ping_pong(requests: Iterator[bytes], context) -> Iterator[bytes]:
        for _ in requests:
            yield response

    handler = grpc.method_handlers_generic_handler(
        "bench.Stream",
        {
            "PingPong": grpc.stream_stream_rpc_method_handler(
                ping_pong,
                request_deserializer=_identity,
                response_serializer=_identity,
            )
        },
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", args.max_message_bytes),
            ("grpc.max_receive_message_length", args.max_message_bytes),
        ],
    )
    server.add_generic_rpc_handlers((handler,))
    server.add_insecure_port(args.bind)
    server.start()
    print(f"listening on {args.bind}, response_bytes={args.response_bytes}", flush=True)
    server.wait_for_termination()


def client(args: argparse.Namespace) -> None:
    request = os.urandom(args.request_bytes)

    def requests() -> Iterator[bytes]:
        for _ in range(args.messages):
            yield request

    channel = grpc.insecure_channel(
        args.target,
        options=[
            ("grpc.max_send_message_length", args.max_message_bytes),
            ("grpc.max_receive_message_length", args.max_message_bytes),
        ],
    )
    stub = channel.stream_stream(
        METHOD,
        request_serializer=_identity,
        response_deserializer=_identity,
    )

    received = 0
    start = time.perf_counter()
    for response in stub(requests(), timeout=args.timeout):
        received += len(response)
    elapsed = time.perf_counter() - start

    sent = args.messages * args.request_bytes
    total = sent + received
    print(f"messages: {args.messages}")
    print(f"elapsed_s: {elapsed:.6f}")
    print(f"sent_MB: {sent / 1e6:.3f}")
    print(f"received_MB: {received / 1e6:.3f}")
    print(f"aggregate_MBps: {total / elapsed / 1e6:.3f}")
    print(f"client_to_server_MBps: {sent / elapsed / 1e6:.3f}")
    print(f"server_to_client_MBps: {received / elapsed / 1e6:.3f}")
    print(f"messages_per_s: {args.messages / elapsed:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--bind", default="0.0.0.0:50051")
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--messages", type=int, default=10000)
    parser.add_argument("--request-bytes", type=int, default=65536)
    parser.add_argument("--response-bytes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-message-bytes", type=int, default=256 * 1024 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.role == "server":
        serve(args)
    else:
        client(args)


if __name__ == "__main__":
    main()
