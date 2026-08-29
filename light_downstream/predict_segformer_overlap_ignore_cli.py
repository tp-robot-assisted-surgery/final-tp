import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_from_disk
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


mask_cols = [
    ("mask_abdominal_wall", 1),
    ("mask_colon", 2),
    ("mask_liver", 3),
    ("mask_pancreas", 4),
    ("mask_small_intestine", 5),
    ("mask_spleen", 6),
    ("mask_stomach", 7),
]


def mask_to_numpy(mask_img):
    arr = np.array(mask_img)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def build_label_map_overlap_ignore(sample):
    image = sample["image"].convert("RGB")
    w, h = image.size

    organ_masks = []

    for col, class_id in mask_cols:
        if col not in sample or sample[col] is None:
            m = np.zeros((h, w), dtype=np.uint8)
        else:
            m = mask_to_numpy(sample[col])

            if m.shape != (h, w):
                m = np.array(
                    Image.fromarray(m.astype(np.uint8)).resize(
                        (w, h),
                        resample=Image.NEAREST,
                    )
                )

            m = (m > 128).astype(np.uint8)

        organ_masks.append((class_id, m))

    stack = np.stack([m for _, m in organ_masks], axis=0)
    count = stack.sum(axis=0)

    label_map = np.zeros((h, w), dtype=np.uint8)

    for class_id, m in organ_masks:
        label_map[(m == 1) & (count == 1)] = class_id

    label_map[count > 1] = 255

    return image, label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_dir)[args.split]

    out_root = Path(args.output_root)
    pred_dir = out_root / "segformer_predictions"
    gt_dir = out_root / "gt_masks"

    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained(
        args.checkpoint,
        do_reduce_labels=False,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    print("dataset size:", len(ds))
    print("checkpoint:", args.checkpoint)
    print("split:", args.split)
    print("output:", out_root)

    for i, sample in enumerate(ds):
        image, gt = build_label_map_overlap_ignore(sample)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )

            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        name = f"{args.prefix}_{i:05d}.png"

        Image.fromarray(gt).save(gt_dir / name)
        Image.fromarray(pred).save(pred_dir / name)

        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(ds)}")

    print("DONE")


if __name__ == "__main__":
    main()
