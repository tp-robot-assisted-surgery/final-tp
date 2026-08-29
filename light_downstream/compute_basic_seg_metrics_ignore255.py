import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def safe_div(a, b):
    if b == 0:
        return float("nan")
    return float(a / b)


def nanmean(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def compute_binary_metrics(gt_bin, pred_bin, valid):
    tp = np.logical_and(gt_bin, pred_bin).sum()
    fp = np.logical_and(np.logical_not(gt_bin), pred_bin).sum()
    fn = np.logical_and(gt_bin, np.logical_not(pred_bin)).sum()
    tn = np.logical_and(
        np.logical_and(np.logical_not(gt_bin), np.logical_not(pred_bin)),
        valid,
    ).sum()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "iou": safe_div(tp, tp + fp + fn),
        "f1_dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for item in args.labels:
        name, idx = item.split(":")
        labels.append((name, int(idx)))

    gt_files = sorted(gt_dir.glob("*.png"))
    pred_by_name = {p.name: p for p in pred_dir.glob("*.png")}

    pairs = []
    for gt_path in gt_files:
        pred_path = pred_by_name.get(gt_path.name)
        if pred_path is not None:
            pairs.append((gt_path, pred_path))

    if len(pairs) == 0:
        raise RuntimeError("No matching GT/prediction files found.")

    rows = []

    for class_name, class_id in labels:
        global_tp = 0
        global_fp = 0
        global_fn = 0
        global_tn = 0

        image_metrics = []
        gt_imgs = 0
        pred_imgs = 0

        for gt_path, pred_path in pairs:
            gt = np.array(Image.open(gt_path))
            pred = np.array(Image.open(pred_path))

            valid = gt != args.ignore_index

            gt_bin = (gt == class_id) & valid
            pred_bin = (pred == class_id) & valid

            if gt_bin.any():
                gt_imgs += 1
            if pred_bin.any():
                pred_imgs += 1

            m = compute_binary_metrics(gt_bin, pred_bin, valid)

            global_tp += m["tp"]
            global_fp += m["fp"]
            global_fn += m["fn"]
            global_tn += m["tn"]

            if gt_bin.any():
                image_metrics.append(m)

        global_iou = safe_div(global_tp, global_tp + global_fp + global_fn)
        global_dice = safe_div(2 * global_tp, 2 * global_tp + global_fp + global_fn)
        global_precision = safe_div(global_tp, global_tp + global_fp)
        global_recall = safe_div(global_tp, global_tp + global_fn)
        global_specificity = safe_div(global_tn, global_tn + global_fp)

        row = {
            "class": class_name,
            "class_id": class_id,
            "valid_gt": gt_imgs > 0,
            "gt_imgs": gt_imgs,
            "pred_imgs": pred_imgs,
            "global_iou": global_iou,
            "global_f1_dice": global_dice,
            "global_precision": global_precision,
            "global_recall": global_recall,
            "global_specificity": global_specificity,
            "mean_image_iou": nanmean([m["iou"] for m in image_metrics]),
            "mean_image_f1_dice": nanmean([m["f1_dice"] for m in image_metrics]),
            "mean_image_precision": nanmean([m["precision"] for m in image_metrics]),
            "mean_image_recall": nanmean([m["recall"] for m in image_metrics]),
            "mean_image_specificity": nanmean([m["specificity"] for m in image_metrics]),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    valid_df = df[df["valid_gt"] == True]

    summary = {
        "num_classes": len(labels),
        "num_valid_gt_classes": int(valid_df.shape[0]),
        "macro_global_iou_valid_gt": float(valid_df["global_iou"].mean()),
        "macro_global_f1_dice_valid_gt": float(valid_df["global_f1_dice"].mean()),
        "macro_global_precision_valid_gt": float(valid_df["global_precision"].mean()),
        "macro_global_recall_valid_gt": float(valid_df["global_recall"].mean()),
        "macro_global_specificity_valid_gt": float(valid_df["global_specificity"].mean()),
        "macro_mean_image_iou_valid_gt": float(valid_df["mean_image_iou"].mean()),
        "macro_mean_image_f1_dice_valid_gt": float(valid_df["mean_image_f1_dice"].mean()),
        "macro_mean_image_precision_valid_gt": float(valid_df["mean_image_precision"].mean()),
        "macro_mean_image_recall_valid_gt": float(valid_df["mean_image_recall"].mean()),
        "macro_mean_image_specificity_valid_gt": float(valid_df["mean_image_specificity"].mean()),
    }

    df.to_csv(out_dir / "metrics_per_class.csv", index=False)

    with open(out_dir / "metrics_per_class.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("========== SUMMARY ==========\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\n========== PER CLASS MEAN IMAGE METRICS ==========\n")
        for _, row in df.iterrows():
            f.write(
                f"{row['class']}: "
                f"IoU={row['mean_image_iou']:.4f}, "
                f"Dice={row['mean_image_f1_dice']:.4f}, "
                f"Precision={row['mean_image_precision']:.4f}, "
                f"Recall={row['mean_image_recall']:.4f}, "
                f"GT imgs={int(row['gt_imgs'])}, "
                f"Pred imgs={int(row['pred_imgs'])}\n"
            )

    print("========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
