"""
train_controlnet_colormap.py
============================
Fine-tune the ControlNet on PER-CLASS color maps, warm-started from a checkpoint
you already have. Now wired to exp_utils so every run lands in its own
timestamped folder on the horse node (no overwrites across runs).

Switches (flags, so ablations need no code edits):
  --init             checkpoint to warm-start from (default: your multi-organ epoch-100)
  --prompt           text prompt, identical in train & inference (default: "" = text-free)
  --label_map_dir    optional dir of Dong's integer 0-7 label maps (else built from columns)
  --run_name         descriptive tag (default: auto-built from the flags above)
  --keep_last_n      keep only the newest N epoch checkpoints to save disk (default 3)
  --use_lora         ALSO adapt the UNet via LoRA (supervisor: "fine-tune the diffusion model")
  --lora_lr          LoRA learning rate (default 1e-4)
  --rank             LoRA rank (default 8)
  --max_train_samples  cap the dataset (for smoke tests; default = full split)

NOTE on --use_lora: it is OPT-IN. Without the flag the script behaves exactly as
before (ControlNet-only training), so old runs reproduce unchanged.

Run examples:
  python train_controlnet_colormap.py
  python train_controlnet_colormap.py --prompt "laparoscopic surgery, abdominal cavity"
  python train_controlnet_colormap.py --init lllyasviel/control_v11p_sd15_seg --use_lora
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    ControlNetModel, UNet2DConditionModel, AutoencoderKL, DDPMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms

from peft import LoraConfig, get_peft_model_state_dict

import exp_utils

MODEL_ID = "runwayml/stable-diffusion-v1-5"  # cached on Capella; mirror: stable-diffusion-v1-5/stable-diffusion-v1-5
RES = 512

# ---- OFFICIAL DSAD class order (from Dong's repo README). MUST match his. -----
CLASS_INDICES = {
    "mask_abdominal_wall": 1, "mask_colon": 2, "mask_liver": 3, "mask_pancreas": 4,
    "mask_small_intestine": 5, "mask_spleen": 6, "mask_stomach": 7,
}
PALETTE = np.array([
    [0,0,0], [0,0,255], [0,255,0], [255,0,0],
    [255,255,0], [128,128,128], [255,0,255], [0,255,255],
], dtype=np.uint8)
# Overlap rule (columns path only): rare organs painted last so they win overlaps.
PAINT_ORDER = ["mask_abdominal_wall", "mask_stomach", "mask_liver",
               "mask_colon", "mask_small_intestine", "mask_spleen", "mask_pancreas"]


def build_label_map_from_columns(item):
    label = np.zeros((RES, RES), dtype=np.uint8)
    for key in PAINT_ORDER:
        if key not in item or item[key] is None:
            continue
        m = item[key].convert("L").resize((RES, RES), Image.NEAREST)
        label[np.array(m) > 127] = CLASS_INDICES[key]
    return label


def colorize(label_map):
    rgb = PALETTE[label_map]
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


class ColorMapDSADDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_map_dir=None, max_samples=None):
        self.dataset = load_from_disk(data_path)["train"]
        if max_samples is not None:                                  # smoke-test cap
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.label_map_dir = label_map_dir
        self.image_transforms = transforms.Compose([
            transforms.Resize((RES, RES)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = self.image_transforms(item["image"])
        if img.shape[0] == 1:   img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4: img = img[:3, :, :]
        if self.label_map_dir:
            p = os.path.join(self.label_map_dir, f"{idx:04d}.png")
            label = np.array(Image.open(p).convert("L").resize((RES, RES), Image.NEAREST))
        else:
            label = build_label_map_from_columns(item)
        return {"pixel_values": img, "conditioning_pixel_values": colorize(label)}


def auto_run_name(args):
    init = ("multiorgan" if "multiorgan" in args.init else
            "single" if "single" in args.init else
            "seg" if "seg" in args.init else
            "canny" if "canny" in args.init else "ckpt")
    prompt = "textfree" if args.prompt == "" else "prompted"
    mask = "donglabels" if args.label_map_dir else "colormap"
    tag = f"{init}_{prompt}_{mask}"
    if args.use_lora:
        tag += "_lora"
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/data/horse/ws/faqa581h-dsad_workspace/multi_dsad")
    p.add_argument("--init", default="/data/horse/ws/faqa581h-dsad_workspace/syna_generative_baseline/multiorgan_baseline/checkpoints/epoch_100")
    p.add_argument("--prompt", default="")
    p.add_argument("--label_map_dir", default=None)
    p.add_argument("--run_name", default=None, help="Defaults to an auto tag from the flags.")
    p.add_argument("--exp_root", default=exp_utils.DEFAULT_EXP_ROOT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--keep_last_n", type=int, default=3, help="0 = keep every checkpoint.")
    # --- LoRA (opt-in; supervisor point 2: also fine-tune the diffusion model) ---
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap dataset size; use 2 for a smoke test.")
    args = p.parse_args()

    run_name = args.run_name or auto_run_name(args)
    run_dir = exp_utils.create_run_dir(run_name, vars(args), args.exp_root)
    log = exp_utils.get_logger(run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log.info(f"Run dir: {run_dir}")
    log.info(f"init={args.init}")
    log.info(f"prompt={args.prompt!r} | label_map_dir={args.label_map_dir}")
    log.info(f"use_lora={args.use_lora} | lora_lr={args.lora_lr} | rank={args.rank}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.bfloat16)
    controlnet = ControlNetModel.from_pretrained(args.init, torch_dtype=torch.bfloat16)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)

    # ---- optional UNet LoRA --------------------------------------------------
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet.add_adapter(lora_cfg)
        # script runs pure bf16 with NO autocast -> adapters must be bf16 too,
        # else bf16 activations x fp32 adapter weights raises a dtype error.
        unet.to(dtype=torch.bfloat16)
        unet.train()
        lora_params = [p for p in unet.parameters() if p.requires_grad]
        log.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")

    controlnet.train(); controlnet.enable_gradient_checkpointing()
    for m in (vae, text_encoder, unet, controlnet): m.to(device)

    if args.use_lora:
        optimizer = torch.optim.AdamW([
            {"params": controlnet.parameters(), "lr": args.lr},
            {"params": [p for p in unet.parameters() if p.requires_grad], "lr": args.lora_lr},
        ])
    else:
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    def save_lora(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionControlNetPipeline.save_lora_weights(save_directory=save_dir, unet_lora_layers=sd)

    def prompt_embeds(bsz):
        tok = tokenizer([args.prompt] * bsz, padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return text_encoder(tok)[0]

    def training_step(batch):
        optimizer.zero_grad()
        images = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        masks = batch["conditioning_pixel_values"].to(device, dtype=torch.bfloat16)
        bsz = images.shape[0]
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, ts)
        ehs = prompt_embeds(bsz)
        down, mid = controlnet(noisy, ts, encoder_hidden_states=ehs,
                               controlnet_cond=masks, return_dict=False)
        pred = unet(noisy, ts, encoder_hidden_states=ehs,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid).sample
        loss = F.mse_loss(pred, noise)
        loss.backward(); optimizer.step()
        return loss.item()

    loader = DataLoader(
        ColorMapDSADDataset(args.data_path, args.label_map_dir, args.max_train_samples),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    for epoch in range(args.epochs):
        total = 0.0
        for step, batch in enumerate(loader):
            total += training_step(batch)
            if step % 250 == 0:
                log.info(f"Epoch {epoch+1} | Step {step} | Loss {total/(step+1):.4f}")
        log.info(f"Epoch {epoch+1} done | Avg {total/len(loader):.4f}")
        if (epoch + 1) == args.epochs or (epoch + 1) % args.save_every == 0:
            out = os.path.join(ckpt_dir, f"epoch_{epoch+1}")
            controlnet.save_pretrained(out)
            if args.use_lora:
                save_lora(os.path.join(out, "lora"))
            exp_utils.prune_checkpoints(run_dir, args.keep_last_n)
            log.info(f"  saved {out}")

    controlnet.save_pretrained(os.path.join(ckpt_dir, "final"))
    if args.use_lora:
        save_lora(os.path.join(ckpt_dir, "final", "lora"))
    log.info(f"Done. Final checkpoint + config.json in {run_dir}")


if __name__ == "__main__":
    main()
