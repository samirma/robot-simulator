"""MolmoAct2 as an `arm_task` policy. The only module here that imports torch.

Chosen over the rest of the 2026-08 survey because it is the one top-scoring model with
a working path on an Apple-Silicon machine: plain `transformers` + `trust_remote_code`,
no flash-attn, no CUDA graphs required (its fast path checks `device.type == "cuda"`
and falls back cleanly), and a checkpoint fine-tuned for exactly this arm
(`allenai/MolmoAct2-SO100_101`, absolute joint-pose control, 6-D state).

Kept apart from `arm_task.py` so that module stays importable -- and testable -- without
a 2 GB torch install; the split is the same one `inspect_so101.py` uses for the Inspect
stack.
"""

from __future__ import annotations

import sys

import numpy as np

NORM_TAG = "so100_so101_molmoact2"


class MolmoActPolicy:
    """Load the checkpoint once, then map (image, task, state) -> an action chunk."""

    def __init__(self, model_id: str, device: str = "mps", dtype: str = "bfloat16") -> None:
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            snapshot_download(model_id, local_files_only=True)
        except LocalEntryNotFoundError:
            print(
                f">> first run: downloading {model_id} (~22 GB) from huggingface.co; "
                "this is a one-time cost",
                file=sys.stderr,
            )
            snapshot_download(model_id)

        if device == "mps" and not torch.backends.mps.is_available():
            print("warning: MPS not available, falling back to CPU", file=sys.stderr)
            device = "cpu"
        self._device = device
        self._torch = torch

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        print(f">> loading {model_id} ({dtype} on {device})", file=sys.stderr)
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Explicit dtype + .to(device), never device_map="auto": accelerate's dispatch
        # assumes CUDA memory queries that do not exist on MPS. Eager attention for the
        # same reason -- the checkpoint's remote code offers flash-attn paths that are
        # CUDA-only.
        self._model = (
            AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch_dtype,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )

    def predict_chunk(self, image_rgb: np.ndarray, task: str, state: np.ndarray) -> np.ndarray:
        """One forward pass: (H, W, 3) RGB + task text + 6-D state -> (N, 6) chunk."""
        from PIL import Image

        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
        with self._torch.inference_mode():
            out = self._model.predict_action(
                processor=self._processor,
                images=[image],
                task=task,
                state=np.asarray(state, dtype=np.float32),
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                # CUDA graphs would be skipped on MPS anyway (the manager checks the
                # device); passing False keeps the intent visible.
                enable_cuda_graph=False,
            )
        actions = out.actions if hasattr(out, "actions") else out
        chunk = np.asarray(
            actions.float().cpu().numpy() if hasattr(actions, "float") else actions,
            dtype=np.float64,
        )
        # Some paths return a leading batch dim; the contract here is (N, 6).
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        return chunk
