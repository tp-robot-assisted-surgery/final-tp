import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


LABEL_TO_MASK_COL = {
    1: "mask_abdominal_wall",
    2: "mask_colon",
    3: "mask_liver",
    4: "mask_pancreas",
    5: "mask_small_intestine",
    6: "mask_spleen",
    7: "mask_stomach",
}

REQUIRED_COLS = [
    "image",
    "mask_abdominal_wall",
    "mask_colon",
    "mask_liver",
    "mask_pancreas",
    "mask_small_intestine",
    "mask_spleen",
    "mask_stomach",
]


TARGET_DATASETS = [
    "arm3_depthtrained_epoch_4",
    "arm3_depthtrained_epoch_8",
    "arm3_depthtrained_epoch_12",
    "arm3_depthtrained_epoch_20",
]


def get_files(folder: Path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(folder.glob("*" + suffix))
    return sorted(files)


def find_pairs(ds_dir: Path):
    image_dir = ds_dir / "images"
    mask_dir = ds_dir / "masks"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing images/ or masks/ in {ds_dir}")

    image_files = get_files(image_dir, [".png", ".jpg", ".jpeg"])
    mask_files = get_files(mask_dir, [".png"])

    print(f"{ds_dir.name}: image count = {len(image_files)}")
    print(f"{ds_dir.name}: mask count  = {len(mask_files)}")

    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    if len(mask_files) == 0:
        raise RuntimeError(f"No masks found in {mask_dir}")

    mask_by_stem = {p.stem: p for p in mask_files}
    pairs = []

    for img in image_files:
        if img.stem in mask_by_stem:
            pairs.append((img, mask_by_stem[img.stem]))

    if len(pairs) == len(image_files):
        print(f"{ds_dir.name}: pairing mode = filename stem")
        return pairs

    if len(image_files) == len(mask_files):
        print(f"{ds_dir.name}: WARNING stems do not fully match. Using sorted order.")
        return list(zip(image_files, mask_files))

    raise RuntimeError(
        f"Cannot pair images and masks in {ds_dir}. "
        f"images={len(image_files)}, masks={len(mask_files)}, matched={len(pairs)}"
    )


def build_generated_dataset(ds_dir: Path, work_dir: Path, real_features):
    pairs = find_pairs(ds_dir)

    converted_dir = work_dir / ds_dir.name / "converted_binary_masks"

    if converted_dir.exists():
        shutil.rmtree(converted_dir)

    converted_dir.mkdir(parents=True, exist_ok=True)

    data = {col: [] for col in REQUIRED_COLS}

    for idx, (img_path, mask_path) in enumerate(pairs):
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_arr = np.array(mask)

        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

        mask_arr = mask_arr.astype(np.uint8)

        unique_values = sorted(np.unique(mask_arr).tolist())
        bad_values = [v for v in unique_values if v < 0 or v > 7]

        if bad_values:
            raise ValueError(
                f"Invalid label values in {mask_path}: {bad_values}. "
                f"Expected integer labels 0-7. Unique values: {unique_values[:50]}"
            )

        if image.size != (mask_arr.shape[1], mask_arr.shape[0]):
            raise ValueError(
                f"Size mismatch: image={img_path}, image_size={image.size}, "
                f"mask={mask_path}, mask_size={(mask_arr.shape[1], mask_arr.shape[0])}"
            )

        data["image"].append(str(img_path))

        for label_id, mask_col in LABEL_TO_MASK_COL.items():
            out_dir = converted_dir / mask_col
            out_dir.mkdir(parents=True, exist_ok=True)

            binary = (mask_arr == label_id).astype(np.uint8) * 255
            out_path = out_dir / f"{idx:06d}.png"
            Image.fromarray(binary, mode="L").save(out_path)

            data[mask_col].append(str(out_path))

        if (idx + 1) % 100 == 0:
            print(f"{ds_dir.name}: converted {idx + 1}/{len(pairs)}")

    generated_ds = Dataset.from_dict(data, features=real_features)

    print(f"{ds_dir.name}: generated train size = {len(generated_ds)}")

    return generated_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dataset", required=True)
    parser.add_argument("--epoch-sweep-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-generated-size", type=int, default=862)
    args = parser.parse_args()

    real_dataset_path = Path(args.real_dataset)
    epoch_sweep_root = Path(args.epoch_sweep_root)
    output_root = Path(args.output_root)
    work_dir = Path(args.work_dir)

    print("real_dataset:", real_dataset_path)
    print("epoch_sweep_root:", epoch_sweep_root)
    print("output_root:", output_root)
    print("work_dir:", work_dir)

    real_ds = load_from_disk(str(real_dataset_path))

    if not isinstance(real_ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(real_ds)}")

    for split in ["train", "validation", "test"]:
        if split not in real_ds:
            raise KeyError(f"Real DSAD dataset has no {split} split.")

    missing = [c for c in REQUIRED_COLS if c not in real_ds["train"].column_names]
    if missing:
        raise KeyError("Missing columns in real train split: " + str(missing))

    real_train = real_ds["train"].select_columns(REQUIRED_COLS)
    real_val = real_ds["validation"].select_columns(REQUIRED_COLS)
    real_test = real_ds["test"].select_columns(REQUIRED_COLS)

    real_features = real_train.features

    print("real train size:", len(real_train))
    print("real validation size:", len(real_val))
    print("real test size:", len(real_test))

    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "dataset,real_train_size,generated_size,combined_train_size,validation_size,test_size,warning"
    ]

    for name in TARGET_DATASETS:
        ds_dir = epoch_sweep_root / name

        print("\n========================================")
        print("Processing:", name)
        print("========================================")

        if not ds_dir.is_dir():
            raise FileNotFoundError(f"Missing generated dataset folder: {ds_dir}")

        generated_train = build_generated_dataset(
            ds_dir=ds_dir,
            work_dir=work_dir,
            real_features=real_features,
        )

        warning = ""
        if args.expected_generated_size > 0 and len(generated_train) != args.expected_generated_size:
            warning = f"expected_generated_{args.expected_generated_size}_but_got_{len(generated_train)}"
            print("WARNING:", warning)

        combined_train = concatenate_datasets([real_train, generated_train])

        out_ds = DatasetDict({
            "train": combined_train,
            "validation": real_val,
            "test": real_test,
        })

        out_path = output_root / name

        if out_path.exists():
            shutil.rmtree(out_path)

        out_ds.save_to_disk(str(out_path))

        print("Saved:", out_path)
        print("real train size:", len(real_train))
        print("generated size:", len(generated_train))
        print("combined train size:", len(combined_train))
        print("validation size:", len(real_val))
        print("test size:", len(real_test))

        summary_lines.append(
            f"{name},{len(real_train)},{len(generated_train)},{len(combined_train)},{len(real_val)},{len(real_test)},{warning}"
        )

    summary_path = output_root / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\nDONE")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
