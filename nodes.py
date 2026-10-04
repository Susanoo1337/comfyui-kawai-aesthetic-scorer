import os
import re

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .kawai_scorer import (
    DEFAULT_REPO_ID,
    DEFAULT_WEIGHTS,
    DTYPES,
    KawaiAestheticModel,
    comfy_image_to_pil,
    parse_weights,
    resolve_model_path,
)

# Models live in ComfyUI/models/aesthetic (extra_model_paths.yaml can add more dirs).
MODEL_FOLDER = "aesthetic"
folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))

_PLACEHOLDER = re.compile(r"\{(grade|label|score|confidence|index)(?::([^{}]*))?\}")


def format_filename(template, **values):
    """Fill {grade}, {label}, {score}, {confidence} and {index} in `template`.

    A format spec is allowed, e.g. {score:.3f} or {index:04d}. Other text, such as
    the %year%/%month%/%day% tokens Save Image expands itself, is left untouched.
    """
    def repl(m):
        value, spec = values[m.group(1)], m.group(2)
        if spec is None and isinstance(value, float):
            spec = ".2f"
        return format(value, spec or "")

    return _PLACEHOLDER.sub(repl, template)


class KawaiAestheticScorerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": DEFAULT_REPO_ID,
                    "tooltip": "Hugging Face repo id or a local folder. Repos are downloaded once to "
                               "ComfyUI/models/aesthetic.",
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "precision": (list(DTYPES), {
                    "default": "fp32",
                    "tooltip": "fp32 matches the model card. bf16/fp16 are faster on GPU but can shift "
                               "borderline grades.",
                }),
                "keep_in_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep the scorer on the GPU between runs instead of moving it back to RAM.",
                }),
            }
        }

    RETURN_TYPES = ("KAWAI_AESTHETIC_MODEL",)
    RETURN_NAMES = ("scorer",)
    FUNCTION = "load"
    CATEGORY = "image/aesthetic"
    DESCRIPTION = "Loads the kawai-aesthetic-scorer-convnextv2 model (SS/S/A/B/C grades for anime/illustrations)."

    def load(self, model_id, device, precision, keep_in_vram):
        device = mm.get_torch_device() if device == "auto" else torch.device(device)
        dtype = DTYPES[precision]
        if device.type == "cpu" and dtype == torch.float16:
            print("[KawaiAestheticScorer] fp16 is not supported well on CPU, using fp32")
            dtype = torch.float32

        offload_device = None
        if not keep_in_vram and device.type != "cpu":
            offload_device = mm.unet_offload_device()

        path = resolve_model_path(model_id, folder_paths.get_folder_paths(MODEL_FOLDER))
        return (KawaiAestheticModel.from_pretrained(path, device, dtype, offload_device),)


class KawaiAestheticScorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scorer": ("KAWAI_AESTHETIC_MODEL",),
                "images": ("IMAGE",),
                "filename_template": ("STRING", {
                    "default": "ComfyUI_{grade}",
                    "tooltip": "Builds the filename_prefix output. Placeholders: {grade} {score} {confidence} "
                               "{label} {index}. Use '/' for subfolders, e.g. 'graded/{grade}/img'.",
                }),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64,
                                       "tooltip": "Images per forward pass. Lower it if you run out of VRAM."}),
            },
            "optional": {
                "score_weights": ("STRING", {
                    "default": ", ".join(str(w) for w in DEFAULT_WEIGHTS),
                    "tooltip": "Weights for SS, S, A, B, C used for the score output "
                               "(sum of probability x weight). Defaults are the model card's example.",
                }),
            },
        }

    # Lists, so every image travels with its own grade: a Save Image node fed these
    # runs once per image and gets that image's filename_prefix.
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "grade", "filename_prefix", "score", "details")
    OUTPUT_IS_LIST = (True, True, True, True, True)
    OUTPUT_TOOLTIPS = (
        "Each input image, one per list item.",
        "Grade of each image: SS, S, A, B or C.",
        "filename_template filled in for each image. Connect to Save Image's filename_prefix.",
        "Weighted score of each image (higher is better).",
        "Readable summary with the probability of every grade.",
    )
    FUNCTION = "score"
    CATEGORY = "image/aesthetic"
    DESCRIPTION = ("Grades images as SS/S/A/B/C. Outputs are lists, so wiring images and filename_prefix into "
                   "Save Image saves every image with its own grade in the filename.")

    def score(self, scorer, images, filename_template, batch_size, score_weights=None):
        scorer, template, batch_size = scorer[0], filename_template[0], batch_size[0]
        weights = parse_weights(score_weights[0]) if score_weights else DEFAULT_WEIGHTS

        # Upstream may send a batch tensor, a list of them, or both; flatten to single images.
        singles = [img for batch in images for img in batch]

        pbar = comfy.utils.ProgressBar(len(singles))

        def on_batch(done, total):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(done, total)

        results = scorer.score([comfy_image_to_pil(img) for img in singles], batch_size, weights, on_batch)

        out_images, grades, prefixes, scores, details = [], [], [], [], []
        for index, (img, r) in enumerate(zip(singles, results)):
            out_images.append(img.unsqueeze(0))
            grades.append(r["grade"])
            prefixes.append(format_filename(
                template, grade=r["grade"], label=r["label"], score=r["score"],
                confidence=r["confidence"], index=index,
            ))
            scores.append(r["score"])
            probs = " ".join(f"{g} {p:.1%}" for g, p in r["probs"].items())
            details.append(f"#{index}: {r['grade']} ({r['confidence']:.1%}), score {r['score']:.3f} | {probs}")
            print(f"[KawaiAestheticScorer] {details[-1]}")

        return (out_images, grades, prefixes, scores, details)


NODE_CLASS_MAPPINGS = {
    "KawaiAestheticScorerLoader": KawaiAestheticScorerLoader,
    "KawaiAestheticScorer": KawaiAestheticScorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KawaiAestheticScorerLoader": "Load Kawai Aesthetic Scorer",
    "KawaiAestheticScorer": "Kawai Aesthetic Score (SS/S/A/B/C)",
}
