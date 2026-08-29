import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_from_disk
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
import evaluate


label2id = {
    "background": 0,
    "abdominal_wall": 1,
    "colon": 2,
    "liver": 3,
    "pancreas": 4,
    "small_intestine": 5,
    "spleen": 6,
    "stomach": 7,
}

id2label = {v: k for k, v in label2id.items()}
num_classes = len(label2id)

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


def collapse_masks_overlap_ignore(sample):
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

    final_mask = np.zeros((h, w), dtype=np.uint8)

    for class_id, m in organ_masks:
        final_mask[(m == 1) & (count == 1)] = class_id

    final_mask[count > 1] = 255

    return final_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--model-name",
        default="nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("DATASET:", args.dataset_dir)
    print("OUTPUT:", args.output_dir)
    print("MODEL:", args.model_name)
    print("EPOCHS:", args.epochs)
    print("BATCH SIZE:", args.batch_size)
    print("IMAGE SIZE:", args.image_size)

    dataset = load_from_disk(args.dataset_dir)
    print(dataset)

    image_processor = SegformerImageProcessor.from_pretrained(
        args.model_name,
        do_reduce_labels=False,
        size={"height": args.image_size, "width": args.image_size},
    )

    def transforms(example_batch):
        images = []
        labels = []

        batch_size = len(example_batch["image"])

        for i in range(batch_size):
            sample = {k: example_batch[k][i] for k in example_batch.keys()}
            image = sample["image"].convert("RGB")
            mask = collapse_masks_overlap_ignore(sample)

            images.append(image)
            labels.append(mask)

        inputs = image_processor(
            images=images,
            segmentation_maps=labels,
            return_tensors="pt",
        )

        return inputs

    dataset["train"].set_transform(transforms)
    dataset["validation"].set_transform(transforms)

    mean_iou_metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        logits_tensor = torch.from_numpy(logits)
        logits_tensor = F.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        pred_labels = logits_tensor.argmax(dim=1).detach().cpu().numpy()

        metrics = mean_iou_metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        result = {
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
        }

        if "per_category_iou" in metrics:
            for i, v in enumerate(metrics["per_category_iou"]):
                result[f"iou_{id2label[i]}"] = v

        return result

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.config.semantic_loss_ignore_index = 255

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        logging_steps=10,
        remove_unused_columns=False,
        fp16=True,
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    final_model_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_model_path)
    image_processor.save_pretrained(final_model_path)

    print("Saved final model to:", final_model_path)


if __name__ == "__main__":
    main()
