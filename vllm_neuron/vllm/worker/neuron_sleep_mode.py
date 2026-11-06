# SPDX-License-Identifier: Apache-2.0
"""Neuron sleep-mode backend (host-staging weight offload).

Implements vLLM's pluggable
:class:`~vllm.device_allocator.sleep_mode_backend.SleepModeBackend` contract for
AWS Trainium/Neuron. It is selected with ``--sleep-mode-backend neuron``
(``ModelConfig.sleep_mode_backend``) and registered from the plugin's
``vllm.general_plugins`` entry point (see ``vllm_neuron.vllm.patches``).

The correctness contract mirrors the Neuron weight-transfer backend
(``NeuronWeightTransferEngine``): the compiled graph reads the model's parameter
tensors by address, so suspend/resume must never swap a parameter object or its
storage. We therefore stage a host (CPU) copy of every weight on ``suspend`` and
restore it with an in-place ``copy_`` into the *same* resident tensor on
``resume``.

Phase-1 mechanism (host-staging): weights are copied to host RAM on ``suspend``
and copied back on ``resume``. The device storage is zeroed in place to mark the
weights as offloaded without invalidating the compiled graph; this is the
correctness/round-trip skeleton. Zeroing does **not** return HBM to the runtime
yet -- the real free (a neuronx device-allocator offload, analogous to vLLM's
``CuMemAllocator``) replaces :meth:`_zero_device_storage` once its API is
confirmed on Trainium. The seam is deliberately narrow so that swap is a
drop-in.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
from vllm.logger import init_logger

logger = init_logger(__name__)


class NeuronSleepModeBackend(SleepModeBackend):
    """Offload/restore resident model weights to/from host RAM, in place.

    ``attach_model`` must be called before ``suspend``/``resume`` so the backend
    knows which tensors to stage. The worker wires the live model (and the
    spec-decode draft model, if any) in from ``NeuronWorker._get_sleep_mode_backend``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: nn.Module | None = None
        self._drafter_model: nn.Module | None = None
        # Host-staged copies keyed by (owner, name): owner 0 = model, 1 = draft.
        self._staged: dict[tuple[int, str], torch.Tensor] = {}
        # Level-2 buffer save: non-persistent buffers dropped on level-2 sleep.
        self._saved_buffers: dict[tuple[int, str], torch.Tensor] = {}

    def attach_model(
        self, model: nn.Module, drafter_model: nn.Module | None = None
    ) -> None:
        # Unwrap torch.compile's OptimizedModule so named_parameters() walks the
        # inner module — the same resident tensors the compiled graph reads.
        # Mirrors NeuronWeightTransferEngine._param_lookup.
        self._model = getattr(model, "_orig_mod", model)
        self._drafter_model = (
            getattr(drafter_model, "_orig_mod", drafter_model)
            if drafter_model is not None
            else None
        )

    # -- mechanism ---------------------------------------------------------

    def suspend(self, level: int = 1) -> None:
        assert self._model is not None, (
            "attach_model() must be called before suspend()."
        )
        self._state = "SUSPENDED"

        targets = [(self._model, 0)]
        if self._drafter_model is not None:
            targets.append((self._drafter_model, 1))

        for owner, key in targets:
            if level == 2:
                # Level 2 drops buffers; save them first so resume can restore.
                self._saved_buffers.update(
                    {
                        (key, name): buf.detach().cpu().clone()
                        for name, buf in owner.named_buffers()
                    }
                )
            for name, param in owner.named_parameters():
                self._staged[(key, name)] = param.detach().cpu().clone()
                self._zero_device_storage(param)

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        wake_weights = tags is None or "weights" in tags

        if wake_weights and self._staged:
            for key, owner in ((0, self._model), (1, self._drafter_model)):
                if owner is None:
                    continue
                params = dict(owner.named_parameters())
                for (owner_key, name), host in self._staged.items():
                    if owner_key != key:
                        continue
                    param = params.get(name)
                    if param is None:
                        logger.warning(
                            "Sleep resume: parameter '%s' no longer present; "
                            "skipping.",
                            name,
                        )
                        continue
                    self._restore_device_storage(param, host)

        # Restore level-2 buffers into the same storage.
        if wake_weights and self._saved_buffers:
            for key, owner in ((0, self._model), (1, self._drafter_model)):
                if owner is None:
                    continue
                buffers = dict(owner.named_buffers())
                for (owner_key, name), host in self._saved_buffers.items():
                    if owner_key != key:
                        continue
                    buf = buffers.get(name)
                    if buf is None:
                        continue
                    buf.data.copy_(host)

        self._staged = {}
        self._saved_buffers = {}
        self._state = "RUNNING"

    def _zero_device_storage(self, param: nn.Parameter) -> None:
        """Zero the resident weight in place.

        Phase-1 placeholder: zeroing keeps the compiled graph's tensor valid but
        does NOT return HBM to the runtime. Replace with the neuronx
        device-allocator offload (free the backing store, keep the address) once
        its API is confirmed on Trainium.
        """
        with torch.no_grad():
            param.data.zero_()

    def _restore_device_storage(
        self, param: nn.Parameter, host: torch.Tensor
    ) -> None:
        with torch.no_grad():
            param.data.copy_(host.to(device=param.device, dtype=param.dtype))

    # -- capability introspection (see SleepModeBackend) -------------------

    @classmethod
    def preserves_communicators(cls) -> bool:
        # Collective communicators (neuron collectives) are untouched by a
        # weight offload; no reinit is needed on resume.
        return True

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        # We never swap parameter storage, so the compiled NEFF stays valid.
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return True
