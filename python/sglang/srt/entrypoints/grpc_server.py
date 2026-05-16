"""Thin gRPC server wrapper — delegates to smg-grpc-servicer package.

The wrapper keeps the serving node on the gRPC launch path and starts the TLI
DraftForward sidecar from an explicit request-manager construction hook.
"""

import asyncio
import inspect
import json
import logging
import time

from aiohttp import web

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.utils.common import get_bool_env_var
from sglang.srt.speculative.tli_disaggregation import (
    start_tli_draft_service,
    tli_draft_service_enabled,
)

logger = logging.getLogger(__name__)

_REQUEST_MANAGER_CLASS_NAMES = ("GrpcRequestManager", "RequestManager")
_REQUEST_MANAGER_MODULE_NAMES = (
    "smg_grpc_servicer.sglang.server",
    "smg_grpc_servicer.sglang.servicer",
    "smg_grpc_servicer.sglang.grpc_request_manager",
    "smg_grpc_servicer.sglang.request_manager",
)


async def _start_sidecar_server(host: str, port: int, app):
    """Start the aiohttp sidecar and return the runner for cleanup."""
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    logger.info("HTTP sidecar server started on http://%s:%d", host, port)
    return runner


def _add_metrics_routes(app):
    """Add Prometheus /metrics endpoint to the aiohttp app."""
    from prometheus_client import (
        CollectorRegistry,
        multiprocess,
    )
    from prometheus_client.openmetrics.exposition import (
        CONTENT_TYPE_LATEST,
        generate_latest,
    )

    async def metrics_handler(request):
        try:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
            return web.Response(
                body=data,
                headers={"Content-Type": CONTENT_TYPE_LATEST},
            )
        except Exception:
            logger.exception("Failed to generate Prometheus metrics")
            return web.Response(status=500, text="Failed to generate metrics")

    app.router.add_get("/metrics", metrics_handler)


def _check_communicator_results(results, action):
    """Return a web.Response error if results indicate failure, else None."""
    if not results:
        return web.Response(status=500, text="No response from scheduler\n")
    failures = [r for r in results if not r.success]
    if failures:
        msgs = " | ".join(r.message for r in failures)
        return web.Response(status=500, text=f"{action} failed: {msgs}\n")
    return None


def _add_admin_routes(app, request_manager):
    """Add admin endpoints to the aiohttp app.

    Endpoints: /start_profile, /stop_profile.
    Business logic (request construction, env var handling, response interpretation)
    lives here; request_manager only provides the transport to the scheduler.
    """

    async def start_profile_handler(request):
        try:
            if request.content_length and request.content_length > 0:
                try:
                    body = await request.json()
                except json.JSONDecodeError as e:
                    return web.Response(
                        status=400,
                        text=f"Invalid JSON in request body: {e}",
                    )
            else:
                body = {}

            # Build ProfileReq with env var overrides (same as tokenizer_communicator_mixin)
            with_stack = body.get("with_stack")
            env_with_stack = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
            with_stack = (with_stack is not False) and env_with_stack
            record_shapes = body.get("record_shapes")
            env_record_shapes = get_bool_env_var("SGLANG_PROFILE_RECORD_SHAPES", "true")
            record_shapes = (record_shapes is not False) and env_record_shapes

            req = ProfileReq(
                type=ProfileReqType.START_PROFILE,
                output_dir=body.get("output_dir"),
                start_step=body.get("start_step"),
                num_steps=body.get("num_steps"),
                activities=body.get("activities"),
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=body.get("profile_by_stage", False),
                profile_id=str(time.time()),
                merge_profiles=body.get("merge_profiles", False),
                profile_prefix=body.get("profile_prefix"),
                profile_stages=body.get("profile_stages"),
            )
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Start Profile")
            if err:
                return err
            return web.Response(text="Start profiling.\n")
        except Exception as e:
            logger.exception("Failed to start profile")
            return web.Response(
                status=500,
                text=f"Internal error: {type(e).__name__}. Check server logs.\n",
            )

    async def stop_profile_handler(request):
        try:
            req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Stop profile")
            if err:
                return err
            return web.Response(text="Stop profiling. This will take some time.\n")
        except Exception as e:
            logger.exception("Failed to stop profile")
            return web.Response(
                status=500,
                text=f"Internal error: {type(e).__name__}. Check server logs.\n",
            )

    app.router.add_post("/start_profile", start_profile_handler)
    app.router.add_post("/stop_profile", stop_profile_handler)


def _iter_request_manager_classes(smg_grpc_server):
    """Yield concrete request-manager classes that can be wrapped explicitly."""

    seen: set[int] = set()
    seen_classes: set[int] = set()
    search_modules = [smg_grpc_server]
    for module_name in _REQUEST_MANAGER_MODULE_NAMES:
        module = getattr(smg_grpc_server, module_name.rsplit(".", 1)[-1], None)
        if inspect.ismodule(module):
            search_modules.append(module)
        try:
            imported_module = __import__(module_name, fromlist=["*"])
        except ImportError:
            continue
        if inspect.ismodule(imported_module):
            search_modules.append(imported_module)

    for module in search_modules:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        for attr_name in _REQUEST_MANAGER_CLASS_NAMES:
            candidate = getattr(module, attr_name, None)
            if inspect.isclass(candidate) and id(candidate) not in seen_classes:
                seen_classes.add(id(candidate))
                yield candidate


def _install_request_manager_capture(smg_grpc_server):
    """Wrap the request-manager constructor so we get the live instance directly."""

    loop = asyncio.get_running_loop()
    request_manager_future = loop.create_future()
    patched_classes = []

    for request_manager_cls in _iter_request_manager_classes(smg_grpc_server):
        original_init = request_manager_cls.__init__

        def _wrapped_init(self, *args, __original_init=original_init, **kwargs):
            __original_init(self, *args, **kwargs)
            if not request_manager_future.done():
                request_manager_future.set_result(self)

        request_manager_cls.__init__ = _wrapped_init
        patched_classes.append((request_manager_cls, original_init))

    if not patched_classes:
        raise RuntimeError(
            "Unable to locate a gRPC request-manager class to capture explicitly. "
            "TLI draft startup requires smg-grpc-servicer to expose one of: "
            f"{', '.join(_REQUEST_MANAGER_CLASS_NAMES)}."
        )

    def _restore_request_manager_capture():
        for request_manager_cls, original_init in patched_classes:
            request_manager_cls.__init__ = original_init

    return request_manager_future, _restore_request_manager_capture


async def _start_smg_sidecars_when_ready(
    server_args,
    smg_grpc_server,
    request_manager_future,
    sidecar_app,
    sidecar_host,
    sidecar_port,
):
    try:
        request_manager = await asyncio.wait_for(
            request_manager_future, timeout=120.0
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            "Timed out waiting for the gRPC request-manager constructor to "
            "run so the sidecar can start."
        ) from exc
    logger.info("Captured gRPC request manager; starting gRPC sidecar.")
    try:
        _add_admin_routes(sidecar_app, request_manager)
    except Exception as e:
        logger.error(
            "Failed to set up admin routes: %s. Continuing without admin endpoints.",
            e,
            exc_info=True,
        )

    sidecar_runner = None
    try:
        sidecar_runner = await _start_sidecar_server(
            sidecar_host, sidecar_port, sidecar_app
        )
    except OSError as e:
        logger.error(
            "Failed to start HTTP sidecar server: %s. "
            "Continuing without metrics/profile endpoints.",
            e,
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "Unexpected error starting HTTP sidecar server: %s. "
            "Continuing without metrics/profile endpoints.",
            e,
            exc_info=True,
        )

    try:
        if tli_draft_service_enabled(server_args):
            try:
                tli_runner = await start_tli_draft_service(
                    (request_manager, smg_grpc_server), server_args
                )
            except Exception:
                if sidecar_runner is not None:
                    try:
                        await sidecar_runner.cleanup()
                    except Exception:
                        logger.exception(
                            "Failed to cleanly shut down HTTP sidecar server after TLI startup failure."
                        )
                raise
        return sidecar_runner, tli_runner
    except asyncio.CancelledError:
        if sidecar_runner is not None:
            try:
                await sidecar_runner.cleanup()
            except Exception:
                logger.exception(
                    "Failed to cleanly shut down HTTP sidecar server after cancellation."
                )
        raise


async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler.
    """
    try:
        from smg_grpc_servicer.sglang import server as smg_grpc_server
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e

    sidecar_app = web.Application()
    sidecar_runner = None
    grpc_task = None
    tli_runner = None
    sidecar_task = None
    restore_request_manager_capture = None
    sidecar_port = (
        server_args.grpc_http_sidecar_port
        if server_args.grpc_http_sidecar_port is not None
        else server_args.port + 1
    )

    # Metrics setup: must set PROMETHEUS_MULTIPROC_DIR before scheduler
    # processes import prometheus_client, since the env var is inherited
    # at fork time.
    if server_args.enable_metrics:
        try:
            from sglang.srt.observability.func_timer import enable_func_timer
            from sglang.srt.utils import set_prometheus_multiproc_dir

            set_prometheus_multiproc_dir()
            enable_func_timer()
            _add_metrics_routes(sidecar_app)
        except Exception as e:
            logger.error(
                "Failed to set up metrics: %s. Continuing without metrics.",
                e,
                exc_info=True,
            )

    def _capture_sidecar_result(task):
        nonlocal sidecar_runner, tli_runner
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error("gRPC sidecar startup failed: %s", exc)
            return
        result = task.result()
        if result is None:
            return
        sidecar_runner, tli_result = result
        if tli_result is not None:
            tli_runner = tli_result

    try:
        request_manager_future, restore_request_manager_capture = (
            _install_request_manager_capture(smg_grpc_server)
        )
        grpc_task = asyncio.create_task(
            smg_grpc_server.serve_grpc(server_args, model_info=model_info)
        )
        sidecar_task = asyncio.create_task(
            _start_smg_sidecars_when_ready(
                server_args,
                smg_grpc_server,
                request_manager_future,
                sidecar_app,
                server_args.host,
                sidecar_port,
            )
        )
        sidecar_task.add_done_callback(_capture_sidecar_result)
        if tli_draft_service_enabled(server_args):
            while True:
                done, _ = await asyncio.wait(
                    {grpc_task, sidecar_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if grpc_task in done:
                    exc = grpc_task.exception()
                    if exc is not None:
                        raise exc
                    break

                if sidecar_task in done:
                    exc = sidecar_task.exception()
                    if exc is not None:
                        raise exc
                    sidecar_runner, tli_runner = sidecar_task.result()
                    break

            if not grpc_task.done():
                await grpc_task
        else:
            await grpc_task
    finally:
        if restore_request_manager_capture is not None:
            try:
                restore_request_manager_capture()
            except Exception:
                logger.exception("Failed to restore request-manager constructor.")
        if grpc_task is not None and not grpc_task.done():
            grpc_task.cancel()
            try:
                await grpc_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("gRPC server shutdown raised an exception.")
        if sidecar_task is not None:
            if not sidecar_task.done():
                sidecar_task.cancel()
                try:
                    await sidecar_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Sidecar startup failed.")
            elif not sidecar_task.cancelled() and sidecar_task.exception() is None:
                sidecar_runner, tli_runner = sidecar_task.result()
        if tli_runner is not None:
            try:
                await tli_runner.stop(grace=0)
            except Exception as e:
                logger.exception(
                    "Failed to cleanly shut down TLI gRPC server: %s",
                    e,
                )
        if sidecar_runner is not None:
            try:
                await sidecar_runner.cleanup()
            except Exception as e:
                logger.exception(
                    "Failed to cleanly shut down HTTP sidecar server: %s",
                    e,
                )
