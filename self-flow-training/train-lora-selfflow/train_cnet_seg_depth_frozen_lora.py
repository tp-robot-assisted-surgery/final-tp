"""Stage 2: segmentation + depth ControlNets trained with a plain DDPM loss on top of the
frozen Stage-1 LoRA. Depth maps come from precompute_depth.py as {idx:06d}.png."""

import os
import shutil
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, set_peft_model_state_dict
from datasets import load_from_disk

MODEL_ID = "runwayml/stable-diffusion-v1-5"
WS = "/data/horse/ws/faqa581h-dsad_workspace"

DATA = f"{WS}/multi_dsad"
DEPTH_DIR = f"{WS}/depth_maps/train"
LABEL_MAP_DIR = None

LORA_DIR = f"{WS}/model_checkpoints/lora_selfflow_c1/checkpoints/final"
RANK = 8

INIT = "lllyasviel/control_v11p_sd15_seg"
DEPTH_INIT = "lllyasviel/control_v11f1p_sd15_depth"
DEPTH_SCALE = 0.5
TRAIN_DEPTH = False
PROMPT = ""

OUT = f"{WS}/model_checkpoints/cnet_seg_depth_frozen_lora"

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-5
NUM_WORKERS = 8
SEED = 42
MAX_TRAIN_SAMPLES = None

SAVE_EVERY = 1
KEEP_LAST_N = 0
LOG_EVERY = 50

RES = 512

CLASS_INDICES = {
    "mask_abdominal_wall": 1, "mask_colon": 2, "mask_liver": 3, "mask_pancreas": 4,
    "mask_small_intestine": 5, "mask_spleen": 6, "mask_stomach": 7,
}
PALETTE = np.array([
    [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0],
    [255, 255, 0], [128, 128, 128], [255, 0, 255], [0, 255, 255],
], dtype=np.uint8)
PAINT_ORDER = ["mask_abdominal_wall", "mask_stomach", "mask_liver",
               "mask_colon", "mask_small_intestine", "mask_spleen", "mask_pancreas"]


def build_label_map_from_columns(item):
    label = np.zeros((RES, RES), dtype=np.uint8)
    for key in PAINT_ORDER:
        if key not in item or item[key] is None:
            continue
        mask = item[key].convert("L").resize((RES, RES), Image.NEAREST)
        label[np.array(mask) > 127] = CLASS_INDICES[key]
    return label


def colorize(label_map):
    rgb = PALETTE[label_map]
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def to_three_channels(img):
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    if img.shape[0] == 4:
        return img[:3, :, :]
    return img


class ColorMapDepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, depth_dir, label_map_dir=None, max_samples=None):
        self.dataset = load_from_disk(data_path)["train"]
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.depth_dir = depth_dir
        self.label_map_dir = label_map_dir
        self.image_transforms = transforms.Compose([
            transforms.Resize((RES, RES)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def _load_depth(self, idx):
        path = os.path.join(self.depth_dir, f"{idx:06d}.png")
        depth = Image.open(path).convert("L").resize((RES, RES), Image.BILINEAR)
        depth = torch.from_numpy(np.array(depth)).float() / 255.0
        return depth.unsqueeze(0).repeat(3, 1, 1)

    def _load_label_map(self, item, idx):
        if not self.label_map_dir:
            return build_label_map_from_columns(item)
        path = os.path.join(self.label_map_dir, f"{idx:04d}.png")
        return np.array(Image.open(path).convert("L").resize((RES, RES), Image.NEAREST))

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = self.image_transforms(item["image"].convert("RGB"))
        return {
            "pixel_values": to_three_channels(img),
            "conditioning_pixel_values": colorize(self._load_label_map(item, idx)),
            "depth_pixel_values": self._load_depth(idx),
        }


def load_frozen_lora(unet, lora_dir, rank):
    from safetensors.torch import load_file
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))

    fp = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(fp):
        fp = os.path.join(lora_dir, "pytorch_lora_weights.bin")
    state = load_file(fp) if fp.endswith(".safetensors") else torch.load(fp, map_location="cpu")
    unet_sd = {k[len("unet."):]: v for k, v in state.items() if k.startswith("unet.")}
    set_peft_model_state_dict(unet, unet_sd, adapter_name="default")

    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.requires_grad_(False)


def get_logger(out_dir):
    logger = logging.getLogger("cnet_seg_depth")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(out_dir, "train.log")); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def prune_checkpoints(ckpt_dir, keep_last_n):
    if keep_last_n <= 0:
        return
    epoch_dirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("epoch_")],
                        key=lambda d: int(d.split("_")[1]))
    for d in epoch_dirs[:-keep_last_n]:
        shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)


def main():
    ckpt_dir = os.path.join(OUT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log = get_logger(OUT)
    log.info(f"Stage 2 [seg+depth] | colour_init={INIT} | depth_init={DEPTH_INIT} "
             f"| depth_scale={DEPTH_SCALE} | train_depth={TRAIN_DEPTH} | lora_dir={LORA_DIR} (FROZEN)")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", dtype=torch.bfloat16)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    colour_cn = ControlNetModel.from_pretrained(INIT, dtype=torch.bfloat16)
    depth_cn = ControlNetModel.from_pretrained(DEPTH_INIT, dtype=torch.bfloat16)

    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    load_frozen_lora(unet, LORA_DIR, RANK)
    unet.set_adapter("default")
    unet.to(dtype=torch.bfloat16)
    unet.eval()

    colour_cn.train(); colour_cn.enable_gradient_checkpointing()
    if TRAIN_DEPTH:
        depth_cn.train(); depth_cn.enable_gradient_checkpointing()
    else:
        depth_cn.requires_grad_(False); depth_cn.eval()

    controlnet = MultiControlNetModel([colour_cn, depth_cn])

    for module in (vae, text_encoder, unet, colour_cn, depth_cn):
        module.to(device)

    lora_n = sum(p.numel() for n, p in unet.named_parameters() if "lora_" in n)
    cn_n = sum(p.numel() for p in colour_cn.parameters() if p.requires_grad)
    depth_n = sum(p.numel() for p in depth_cn.parameters() if p.requires_grad)
    log.info(f"trainable colour_cn: {cn_n:,} | depth_cn: {depth_n:,} "
             f"({'trainable' if TRAIN_DEPTH else 'frozen'}) | LoRA (frozen): {lora_n:,}")

    param_groups = [{"params": colour_cn.parameters(), "lr": LR}]
    if TRAIN_DEPTH:
        param_groups.append({"params": depth_cn.parameters(), "lr": LR})
    optimizer = torch.optim.AdamW(param_groups)

    def prompt_embeds(bsz):
        ids = tokenizer([PROMPT] * bsz, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return text_encoder(ids)[0]

    def training_step(batch):
        images = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        colour = batch["conditioning_pixel_values"].to(device, dtype=torch.bfloat16)
        depth = batch["depth_pixel_values"].to(device, dtype=torch.bfloat16)
        bsz = images.shape[0]
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        embeds = prompt_embeds(bsz)

        down, mid = controlnet(
            noisy, timesteps, encoder_hidden_states=embeds,
            controlnet_cond=[colour, depth],
            conditioning_scale=[1.0, DEPTH_SCALE],
            return_dict=False,
        )
        pred = unet(noisy, timesteps, encoder_hidden_states=embeds,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid).sample

        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    loader = DataLoader(
        ColorMapDepthDataset(DATA, DEPTH_DIR, LABEL_MAP_DIR, MAX_TRAIN_SAMPLES),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    log.info(f"dataset: {len(loader.dataset)} samples | {len(loader)} steps/epoch | {EPOCHS} epochs")

    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for step, batch in enumerate(loader):
            total += training_step(batch)
            if step % LOG_EVERY == 0:
                log.info(f"E{epoch} S{step}/{len(loader)} | loss {total/(step+1):.4f}")
        log.info(f"Epoch {epoch} done | avg loss {total/len(loader):.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
            colour_cn.save_pretrained(epoch_dir)
            if TRAIN_DEPTH:
                depth_cn.save_pretrained(os.path.join(epoch_dir, "depth_cn"))
            prune_checkpoints(ckpt_dir, KEEP_LAST_N)
            log.info(f"  saved {epoch_dir}")

    final = os.path.join(ckpt_dir, "final")
    colour_cn.save_pretrained(final)
    if TRAIN_DEPTH:
        depth_cn.save_pretrained(os.path.join(final, "depth_cn"))
    log.info(f"Done. Checkpoints in {ckpt_dir}. LoRA unchanged -- reuse LORA_DIR={LORA_DIR} at inference. "
             f"Depth branch {'trained, saved under depth_cn/' if TRAIN_DEPTH else f'kept frozen at {DEPTH_INIT}'}.")


if __name__ == "__main__":
    main()
