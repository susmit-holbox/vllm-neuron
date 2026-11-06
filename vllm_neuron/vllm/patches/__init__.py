# SPDX-License-Identifier: Apache-2.0
"""
Centralized patch management for vllm-neuron.

All monkey-patches to upstream vLLM are registered here. Startup-critical
patches use vLLM's general-plugin hook so they are applied in spawned engine,
API, and worker processes.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

# Tracks whether the "neuron" sleep-mode backend has been registered in vLLM's
# lazy SleepModeBackendFactory. Registration is idempotent per process: the
# factory raises on a duplicate name, and apply_patches can run more than once.
_sleep_mode_backend_registered = False


def _register_sleep_mode_backend() -> None:
    """Register the ``"neuron"`` sleep-mode backend in vLLM's factory.

    Selected with ``--sleep-mode-backend neuron``. The factory is lazy and
    module-path based (mirrors ``KVConnectorFactory``), so the backend is
    registered from vLLM Neuron's own package without patching upstream. The
    backend module is imported only when a worker first calls ``sleep()``.
    """
    global _sleep_mode_backend_registered
    if _sleep_mode_backend_registered:
        return

    try:
        from vllm.device_allocator.sleep_mode_backend import (
            SleepModeBackendFactory,
        )
    except ImportError:
        # The sleep-mode framework arrived in a specific vLLM version. Without
        # it the /sleep endpoint is unavailable and there is nothing to register
        # against.
        logger.warning(
            "vLLM sleep-mode framework is unavailable; the 'neuron' "
            "sleep-mode backend was not registered."
        )
        return

    SleepModeBackendFactory.register_backend(
        "neuron",
        "vllm_neuron.vllm.worker.neuron_sleep_mode",
        "NeuronSleepModeBackend",
    )
    _sleep_mode_backend_registered = True


def apply_patches():
    """Apply all vllm-neuron patches to upstream vLLM."""
    from vllm_neuron.vllm.patches.dp_load_balancer_patch import (
        apply_dp_load_balancer_patch,
    )

    apply_dp_load_balancer_patch()

    _register_sleep_mode_backend()

    # patches/jf_grammar_patch.py wraps a specific
    # StructuredOutputManager and is applied from NeuronScheduler.__init__.

    # patches/shm_reader_spin_patch.py is applied from NeuronWorker.init_device and from the
    # Neuron run_engine_core wrapper, not here: it has to take effect before a process builds
    # a MessageQueue reader, and importing shm_broadcast this early can raise. It is
    # idempotent, so both sites are safe.
