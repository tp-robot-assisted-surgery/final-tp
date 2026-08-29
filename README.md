# Synthetic Surgical Images for Downstream Segmentation

Generate laparoscopic images from organ masks with Stable Diffusion 1.5 + ControlNet on the
DSAD dataset, then check whether adding them to the real training set actually improves a
downstream SegFormer.

DSAD gives us about 850 annotated frames, which is the whole problem. A mask-conditioned
generator can turn one annotated frame into many, but only if it learns enough structure
from that same small set. So the question here is whether a **Self-Flow** representation
self-distillation loss, bolted onto the usual denoising objective, yields synthetic data
that is more useful downstream than the identical model trained with a plain DDPM loss.

## Pipeline

```
control-net-training/   ->   self-flow-training/   ->   light_downstream/
mask -> image                same generators            real + generated data
generators                   + Self-Flow loss           -> SegFormer -> IoU / Dice
```

The third stage is the one that decides anything. Images are sampled from each generator
checkpoint, mixed into the real data, and a SegFormer is fine-tuned and scored on the real
test split. A generator is only better if that final number moves.

## The three arms

| arm | conditioning | script |
|---|---|---|
| Arm 1 | colour map only | `train_controlnet_colormap.py` |
| Arm 2 | colour + **frozen** depth | `train_controlnet_depth.py` |
| Arm 3 | colour + **trained** depth | `train_controlnet_depth.py --train_depth` |

Depth starts frozen because monocular depth estimators are out of domain on surgical
scenes. It is used as a fixed geometric prior at conditioning scale 0.5, and Arm 3 unfreezes
it to see whether adapting it helps.

## Self-Flow

```
L = L_gen + REP_GAMMA * L_rep
```

`L_gen` is the standard single-timestep DDPM epsilon MSE, left untouched so a run never
drifts away from the baseline it is being compared against. `L_rep` is the negative cosine
similarity between mid-block features of a **student**, fed latents where a fraction
`MASK_RATIO` of positions sits at a second independent timestep, and a stop-grad **EMA
teacher**, fed the cleaner input at the smaller of the two timesteps. `REP_GAMMA = 0` turns
Self-Flow off and gives the matched plain-DDPM baseline, with data, seed, LR, rank, epochs
and augmentation all identical. That is what makes the comparison fair.

The loss can be attached at two points, and both are implemented.

**Approach A — on the ControlNet**, over a warm-started LoRA that stays frozen. The teacher
is an EMA deep copy of the ControlNet, and since the LoRA is frozen and shared, nothing gets
swapped inside the loop.

- `self-flow-training/train-cnet-selflfow/train_cnet_lorafreez_selfflow.py` — seg ControlNet
- `self-flow-training/train-cnet-selflfow/train_mcnt_lorafreez_selfflow.py` — seg + depth

**Approach B — on the UNet LoRA**, with the ControlNet trained plainly on top afterwards.
Student and teacher are two PEFT adapters (`default` / `ema`) on the same UNet, picked with
`set_adapter`, so there is no second copy of the model.

- `self-flow-training/train-lora-selfflow/train_lora_selfflow.py` — LoRA, Self-Flow
- `self-flow-training/train-lora-selfflow/train_cnet_seg_frozen_lora.py` — seg ControlNet
- `self-flow-training/train-lora-selfflow/train_cnet_seg_depth_frozen_lora.py` — seg + depth

These five are configured by editing the constants at the top of each file; there are no CLI
flags. Set `WS`, then `DATA`, `OUT` and any warm-start paths. `MAX_TRAIN_SAMPLES = 2` is a
smoke test. Each run writes `train.log` and per-epoch checkpoints under `OUT/checkpoints/`,
with `final/` as the last. The EMA teacher and projection head are training-only and are not
saved.

## Layout

| folder | contents |
|---|---|
| `control-net-training/` | Canny baseline, colour-map generator, colour + depth generator (CLI flags) |
| `self-flow-training/` | the two Self-Flow approaches, five scripts |
| `light_downstream/` | build combined datasets, fine-tune SegFormer-B3, predict, score (CLI flags) |
| `util-img/` | figures used below |

Downstream runs in four steps: `create_depthtrained_hf_datasets_real_plus_generated.py`
builds a combined dataset from the real split plus one checkpoint's output,
`train_segformer_light_overlap_ignore_cli.py` fine-tunes SegFormer-B3,
`predict_segformer_overlap_ignore_cli.py` writes test-split label maps, and
`compute_basic_seg_metrics_ignore255.py` reports IoU, Dice, precision, recall, specificity.

## Conventions

- **Text-free.** Every generator trains with an empty prompt, so it has to rely on the
  spatial conditioning rather than on text.
- **8 classes.** 0 = background, then abdominal wall, colon, liver, pancreas, small
  intestine, spleen, stomach. Rare organs are painted last so they win overlaps. This
  mapping has to stay identical everywhere it appears.
- **Overlaps are ignored.** Where two masks overlap the downstream label is `255`, excluded
  from training and metrics, so ambiguous boundaries never count either way.
- **Depth maps** are precomputed as `{idx:06d}.png`, keyed to the dataset row index.

## Requirements

`torch diffusers transformers peft datasets torchvision safetensors numpy pandas pillow evaluate`

CUDA in practice. bf16 plus gradient checkpointing keeps 512×512 at batch 16 inside 24 GB.

---

# Results

## What the generators produce

Same colour map, same scene, one image per arm. All three follow the mask; the differences
are in texture and how the geometry holds together.

![Samples from the three baseline arms](util-img/gen-samples/base-cnet.png)

Arm 1 against its Self-Flow version, with the real frame for reference:

![Arm 1 with and without Self-Flow](util-img/gen-samples/self-flow-cnet-arm1.png)

And Arm 3, which also gets the depth map as input:

![Arm 3 with and without Self-Flow](util-img/gen-samples/self-flow-cnet-arm3.png)

Samples are worth a look but they do not settle anything — the numbers below do.

## Downstream comparison

Macro mean-image IoU on the real test split. Two reference points matter: **real only**, and
**real duplicated 2×**, which controls for the fact that adding synthetic data also doubles
the sample count. An arm that beats real-only but not duplicated-real has bought nothing but
repetition.

![Main downstream segmentation results](util-img/graphs/downstream_mIoI.png)

| configuration | macro mean-image IoU |
|---|---:|
| Real only | 0.3228 |
| Real duplicated 2× | 0.4287 |
| Arm 1 + Self-Flow | 0.4410 |
| Arm 1: colour only | 0.4540 |
| Arm 2: frozen depth | 0.4760 |
| Arm 3: trained depth | 0.4790 |
| **Arm 3 + Self-Flow** | **0.4820** |

Every arm clears both baselines, so the synthetic images carry real signal rather than acting
as duplication. Depth is the bigger effect; Self-Flow adds a smaller gain on top of the
strongest arm, and costs Arm 1 slightly.

## Checkpoint sensitivity

The same measurement across generator checkpoints (epochs 4, 8, 12, 20), in macro global IoU.

![Downstream IoU by generator checkpoint](util-img/graphs/checkpoint.png)

The spread across checkpoints is about as large as the spread across arms, so a single
checkpoint is not enough to rank two generators against each other.

## Self-Flow at the LoRA stage (Approach B)

Both conditioning settings, both averaging schemes, macro-averaged over the six classes with
valid ground truth — a different denominator from the figures above, so the two sets of
numbers are not directly comparable. Best checkpoint per column in bold.

| Averaging | Epoch | Colour IoU | Colour Dice | Colour + depth IoU | Colour + depth Dice |
|---|---:|---:|---:|---:|---:|
| Mean-image | 4 | 0.3744 | 0.4270 | 0.3584 | 0.4029 |
| Mean-image | 8 | 0.3474 | 0.4071 | 0.4006 | 0.4488 |
| Mean-image | 12 | 0.3653 | 0.4129 | 0.3884 | 0.4426 |
| Mean-image | **20** | **0.4121** | **0.4605** | **0.4193** | **0.4673** |
| Global | 4 | 0.3827 | 0.4896 | 0.3828 | 0.4855 |
| Global | 8 | 0.3989 | 0.5154 | 0.3954 | 0.5058 |
| Global | 12 | 0.3931 | 0.4991 | 0.4124 | 0.5297 |
| Global | **20** | **0.4119** | **0.5208** | **0.4198** | **0.5305** |

Epoch 20 wins every column, which means the generator was still improving when training
stopped — this sweep does not locate a maximum. Colour + depth beats colour alone under both
averaging schemes at that checkpoint, matching the depth effect seen above.
