# ComfyUI Kawai Aesthetic Scorer

ComfyUI nodes that grade images with
[kawaimasa/kawai-aesthetic-scorer-convnextv2](https://huggingface.co/kawaimasa/kawai-aesthetic-scorer-convnextv2),
a ConvNeXtV2 classifier that sorts anime / 2D illustrations into five tiers: **SS, S, A, B, C**.

Every image comes out paired with its grade as text, so you can save each image in a
batch with its grade in the filename (`ComfyUI_SS_00001_.png`, `ComfyUI_B_00002_.png`, ...).

## Installation

Clone into `ComfyUI/custom_nodes` and install the requirements into ComfyUI's Python:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/susanoo1337/comfyui-kawai-aesthetic-scorer
pip install -r comfyui-kawai-aesthetic-scorer/requirements.txt
```

Restart ComfyUI. The model downloads automatically to `ComfyUI/models/aesthetic/kawai-aesthetic-scorer-convnextv2`
the first time the loader runs. To install it offline, put the files from the Hugging Face repo
(`config.json`, the weights, etc.) in that folder yourself.

## Nodes

Both nodes are under **image/aesthetic**.

### Load Kawai Aesthetic Scorer

| Input | Description |
| --- | --- |
| `model_id` | Hugging Face repo id (default `kawaimasa/kawai-aesthetic-scorer-convnextv2`) or a path to a local folder. |
| `device` | `auto` uses ComfyUI's device. |
| `precision` | `fp32` (default) matches the model card. `bf16`/`fp16` are faster on GPU but can flip borderline grades. |
| `keep_in_vram` | Off by default: the scorer moves back to RAM after each run so it doesn't hold VRAM your other models need. |

### Kawai Aesthetic Score (SS/S/A/B/C)

| Input | Description |
| --- | --- |
| `scorer` | From the loader. |
| `images` | One image or a batch. Image lists work too. |
| `filename_template` | Text for the `filename_prefix` output. See [placeholders](#filename-template). Default `ComfyUI_{grade}`. |
| `batch_size` | Images per forward pass. Lower it if you run out of VRAM. |
| `score_weights` | Weights for SS, S, A, B, C used for `score`. Default `1.0, 0.9, 0.5, -0.5, -1.0` (the model card's example). |

| Output | Description |
| --- | --- |
| `images` | The input images, one per list item. |
| `grade` | `SS`, `S`, `A`, `B` or `C` for each image. |
| `filename_prefix` | `filename_template` filled in for each image. |
| `score` | Weighted score for each image: the sum of each grade's probability times its weight. Useful for ranking within a grade. |
| `details` | Readable summary, e.g. `#0: S (71.4%), score 0.803 \| SS 12.0% S 71.4% A 15.1% B 1.2% C 0.3%`. |

All outputs are **lists** with one entry per image. When a node downstream gets a list, ComfyUI runs it
once per item and pairs items by position. That is how a single Save Image node saves every image with its own name.

## Saving each image with its grade in the filename

```
Load Image(s) ─► Kawai Aesthetic Score ─┬─ images ──────────► Save Image: images
                         ▲              └─ filename_prefix ─► Save Image: filename_prefix
Load Kawai Aesthetic Scorer
```

1. Add **Load Kawai Aesthetic Scorer** and **Kawai Aesthetic Score**, and connect `scorer`.
2. Connect your image or batch to `images`.
3. Add a **Save Image** node and connect the scorer's `images` output to its `images` input.
4. Connect the scorer's `filename_prefix` output to Save Image's `filename_prefix` input. In the current
   frontend you can drop the link right on the widget. Older frontends need the widget converted to an
   input first (right-click → *Convert Widget to Input*).

Save Image adds its usual counter, so with the default template the files look like:

```
ComfyUI_SS_00001_.png
ComfyUI_A_00002_.png
ComfyUI_S_00003_.png
```

You can also connect the plain `grade` output straight to `filename_prefix` to get files named `SS_00001_.png`.

### Filename template

| Placeholder | Example | |
| --- | --- | --- |
| `{grade}` | `SS` | Grade. |
| `{score}` | `0.87` | Weighted score, 2 decimals by default. |
| `{confidence}` | `0.93` | Probability of the predicted grade. |
| `{label}` | `SS_tier` | Raw label from the model config. |
| `{index}` | `0` | Position of the image in this run. |

Python format specs work, e.g. `{score:.3f}` or `{index:04d}`. Everything else passes through unchanged.
Save Image still expands `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%`, `%second%`, `%width%` and
`%height%` (e.g. `%year%-%month%-%day%/{grade}`). It does not expand `%date:yyyy-MM-dd%` or
`%NodeName.widget%`: the frontend replaces those only in a value typed into Save Image itself, not in one
that arrives through a link.

Examples:

* `ComfyUI_{grade}` → `ComfyUI_SS_00001_.png`
* `graded/{grade}/img` → sorts images into `output/graded/SS/`, `output/graded/S/`, ...
* `{grade}_{score:.3f}` → `S_0.803_00001_.png`

## About the model

According to the [model card](https://huggingface.co/kawaimasa/kawai-aesthetic-scorer-convnextv2),
the model reflects its author's taste in anime and 2D illustrations. It was trained on about 60,000 images
that were graded by hand. It is not a general aesthetic or beauty model and may not do well on photos,
3D renders, realistic images or Western painting styles.

Preprocessing follows the model card exactly: convert to RGB, resize the long edge to 768px with LANCZOS,
center on a black square, then apply ImageNet normalization. The grade is the class with the highest probability.

The model has its own license and terms; see its Hugging Face page.
