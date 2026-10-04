"""Model wrapper and preprocessing for kawaimasa/kawai-aesthetic-scorer-convnextv2.

Kept free of ComfyUI imports so it can be used (and tested) on its own.
"""

import os
import re

import numpy as np
import torch
from PIL import Image

DEFAULT_REPO_ID = "kawaimasa/kawai-aesthetic-scorer-convnextv2"

# Grade order and example weights from the model card (SS, S, A, B, C).
GRADES = ("SS", "S", "A", "B", "C")
DEFAULT_WEIGHTS = (1.0, 0.9, 0.5, -0.5, -1.0)

# Input size and normalization the model was trained with.
IMAGE_SIZE = 768
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def label_to_grade(label):
    """'SS_tier' -> 'SS'. Labels that don't follow that pattern are returned as-is."""
    label = str(label)
    return re.sub(r"[_\- ]?tier$", "", label, flags=re.IGNORECASE)


def parse_weights(text):
    """Parse a comma/space separated list of 5 floats in SS, S, A, B, C order."""
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) != len(GRADES):
        raise ValueError(
            f"score_weights needs {len(GRADES)} numbers ({', '.join(GRADES)}), got {len(parts)}: {text!r}"
        )
    return tuple(float(p) for p in parts)


def pad_resize(img, size=IMAGE_SIZE):
    """Resize so the long edge equals `size` (LANCZOS) and center it on a black square.

    Mirrors the PadResizeProcessor from the model card; the model expects exactly this.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_normalized_tensor(img):
    """PIL RGB -> normalized CHW float tensor (ToTensor + ImageNet Normalize)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def comfy_image_to_pil(image):
    """One ComfyUI image ([H, W, C] float in 0..1) -> PIL RGB."""
    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def resolve_model_path(repo_id_or_path, model_dirs):
    """Return a local directory holding the model, downloading it if needed.

    Order: an existing directory given directly, then `<dir>/<repo name>` for each
    of `model_dirs`, otherwise download the Hugging Face repo into the first one.
    """
    candidate = os.path.expanduser(repo_id_or_path.strip())
    if os.path.isdir(candidate):
        return candidate

    name = candidate.rstrip("/").split("/")[-1]
    for model_dir in model_dirs:
        local_dir = os.path.join(model_dir, name)
        if os.path.isfile(os.path.join(local_dir, "config.json")):
            return local_dir

    from huggingface_hub import snapshot_download

    local_dir = os.path.join(model_dirs[0], name)
    print(f"[KawaiAestheticScorer] Downloading {candidate} to {local_dir}")
    snapshot_download(repo_id=candidate, local_dir=local_dir)
    return local_dir


class KawaiAestheticModel:
    """Loaded scorer plus the device/dtype bookkeeping needed to run it."""

    def __init__(self, model, device, dtype, offload_device=None):
        self.model = model.eval()
        self.device = torch.device(device)
        self.dtype = dtype
        # When set, the model is parked here between runs to free VRAM.
        self.offload_device = torch.device(offload_device) if offload_device else None

        id2label = getattr(model.config, "id2label", None) or {}
        labels = [id2label.get(i, id2label.get(str(i))) for i in range(model.config.num_labels)]
        if not all(labels) or all(str(l).upper().startswith("LABEL_") for l in labels):
            # Config without meaningful labels: fall back to the model card's order.
            labels = [f"{g}_tier" for g in GRADES]
        self.labels = labels
        self.grades = [label_to_grade(l) for l in labels]

        self.model.to(self.offload_device or self.device, dtype=self.dtype)

    @classmethod
    def from_pretrained(cls, path, device, dtype, offload_device=None):
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(path)
        return cls(model, device, dtype, offload_device)

    def weights_for_labels(self, weights=DEFAULT_WEIGHTS):
        """Reorder SS/S/A/B/C weights to match the model's label indices."""
        by_grade = {g.upper(): w for g, w in zip(GRADES, weights)}
        return torch.tensor([by_grade.get(g.upper(), 0.0) for g in self.grades], dtype=torch.float32)

    @torch.no_grad()
    def predict_probs(self, pil_images, batch_size=4, on_batch=None):
        """Softmax probabilities, shape [N, num_labels], on the CPU in float32.

        `on_batch(done, total)` is called after each batch (progress / cancellation).
        """
        if self.offload_device is not None:
            self.model.to(self.device)
        try:
            out = []
            for start in range(0, len(pil_images), batch_size):
                chunk = pil_images[start:start + batch_size]
                pixels = torch.stack([to_normalized_tensor(pad_resize(im)) for im in chunk])
                pixels = pixels.to(self.device, dtype=self.dtype)
                logits = self.model(pixel_values=pixels).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu())
                if on_batch is not None:
                    on_batch(start + len(chunk), len(pil_images))
            return torch.cat(out, dim=0)
        finally:
            if self.offload_device is not None:
                self.model.to(self.offload_device)

    def score(self, pil_images, batch_size=4, weights=DEFAULT_WEIGHTS, on_batch=None):
        """Grade each image. Returns one dict per image."""
        probs = self.predict_probs(pil_images, batch_size, on_batch)
        scores = (probs * self.weights_for_labels(weights)).sum(dim=-1)
        results = []
        for p, s in zip(probs, scores):
            idx = int(p.argmax())
            results.append({
                "grade": self.grades[idx],
                "label": self.labels[idx],
                "confidence": float(p[idx]),
                "score": float(s),
                "probs": {g: float(v) for g, v in zip(self.grades, p)},
            })
        return results
