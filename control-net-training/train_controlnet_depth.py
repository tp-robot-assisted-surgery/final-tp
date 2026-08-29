"""
train_controlnet_depth.py
=========================
DEPTH ARM (job 2). Same as train_controlnet_colormap.py + LoRA, but with a SECOND
ControlNet branch that reads precomputed monocular depth maps (MultiControlNet).

Branches (diffusers MultiControlNetModel):
  [0] colour branch  -> init lllyasviel/control_v11p_sd15_seg   (TRAINABLE, like job 1)
  [1] depth  branch  -> init lllyasviel/control_v11f1p_sd15_depth (FROZEN to start)

Why frozen depth: MiDaS/DPT depth is out-of-domain on surgical scenes, so we use it
as a fixed geometric prior first and only unfreeze if it clearly helps. The ONLY
variable vs job 1 is the added depth branch -> clean ablation.

Depth maps must already exist (precompute_depth.py), named {idx:06d}.png keyed to the
SAME enumeration index as the dataset rows. Verified: 863 maps, 0 missing, sizes match.

Flags mirror train_controlnet_colormap.py, plus:
  --depth_dir          dir of precomputed depth PNGs for the train split
  --depth_init         depth ControlNet checkpoint
  --depth_scale        conditioning scale for the depth branch (default 0.5; low = gentle)
  --train_depth        unfreeze the depth branch (default: frozen)
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
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms

from peft import LoraConfig, get_peft_model_state_dict

import exp_utils

MODEL_ID = "runwayml/stable-diffusion-v1-5"
RES = 512

CLASS_INDICES = {
    "mask_abdominal_wall": 1, "mask_colon": 2, "mask_liver": 3, "mask_pancreas": 4,
    "mask_small_intestine": 5, "mask_spleen": 6, "mask_stomach": 7,
}
PALETTE = np.array([
    [0,0,0], [0,0,255], [0,255,0], [255,0,0],
    [255,255,0], [128,128,128], [255,0,255], [0,255,255],
], dtype=np.uint8)
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
        # keyed by the SAME index precompute_depth.py wrote: {idx:06d}.png
        p = os.path.join(self.depth_dir, f"{idx:06d}.png")
        d = Image.open(p).convert("L").resize((RES, RES), Image.BILINEAR)  # depth is continuous
        d = torch.from_numpy(np.array(d)).float() / 255.0          # [H,W] in [0,1]
        return d.unsqueeze(0).repeat(3, 1, 1)                       # depth ControlNet wants 3ch

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
        return {
            "pixel_values": img,
            "conditioning_pixel_values": colorize(label),   # colour branch
            "depth_pixel_values": self._load_depth(idx),    # depth branch
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/data/horse/ws/faqa581h-dsad_workspace/multi_dsad")
    p.add_argument("--depth_dir", default="/data/horse/ws/faqa581h-dsad_workspace/depth_maps/train")
    p.add_argument("--init", default="lllyasviel/control_v11p_sd15_seg")
    p.add_argument("--depth_init", default="lllyasviel/control_v11f1p_sd15_depth")
    p.add_argument("--depth_scale", type=float, default=0.5)
    p.add_argument("--train_depth", action="store_true", help="unfreeze depth branch (default frozen)")
    p.add_argument("--prompt", default="")
    p.add_argument("--label_map_dir", default=None)
    p.add_argument("--run_name", default="multiorgan_colormap_lora_depth")
    p.add_argument("--exp_root", default=exp_utils.DEFAULT_EXP_ROOT)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--keep_last_n", type=int, default=3)
    p.add_argument("--max_train_samples", type=int, default=None)
    args = p.parse_args()

    run_dir = exp_utils.create_run_dir(args.run_name, vars(args), args.exp_root)
    log = exp_utils.get_logger(run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log.info(f"Run dir: {run_dir}")
    log.info(f"colour_init={args.init} | depth_init={args.depth_init} | depth_scale={args.depth_scale}")
    log.info(f"train_depth={args.train_depth}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.bfloat16)

    colour_cn = ControlNetModel.from_pretrained(args.init, torch_dtype=torch.bfloat16)
    depth_cn = ControlNetModel.from_pretrained(args.depth_init, torch_dtype=torch.bfloat16)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)

    # UNet LoRA (same as job 1)
    lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                          target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(lora_cfg)
    unet.to(dtype=torch.bfloat16)   # bf16 adapters; script has no autocast
    unet.train()

    # colour branch trainable; depth branch frozen unless --train_depth
    colour_cn.train(); colour_cn.enable_gradient_checkpointing()
    if args.train_depth:
        depth_cn.train(); depth_cn.enable_gradient_checkpointing()
    else:
        depth_cn.requires_grad_(False); depth_cn.eval()

    controlnet = MultiControlNetModel([colour_cn, depth_cn])

    for m in (vae, text_encoder, unet, colour_cn, depth_cn): m.to(device)

    trainable = [{"params": colour_cn.parameters(), "lr": args.lr},
                 {"params": [q for q in unet.parameters() if q.requires_grad], "lr": args.lora_lr}]
    if args.train_depth:
        trainable.append({"params": depth_cn.parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(trainable)

    lora_n = sum(q.numel() for q in unet.parameters() if q.requires_grad)
    log.info(f"LoRA trainable params: {lora_n:,} | depth branch {'TRAINABLE' if args.train_depth else 'frozen'}")

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
        colour = batch["conditioning_pixel_values"].to(device, dtype=torch.bfloat16)
        depth = batch["depth_pixel_values"].to(device, dtype=torch.bfloat16)
        bsz = images.shape[0]
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, ts)
        ehs = prompt_embeds(bsz)
        # MultiControlNet: lists for cond images and scales; it sums the branch residuals
        down, mid = controlnet(
            noisy, ts, encoder_hidden_states=ehs,
            controlnet_cond=[colour, depth],
            conditioning_scale=[1.0, args.depth_scale],
            return_dict=False,
        )
        pred = unet(noisy, ts, encoder_hidden_states=ehs,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid).sample
        loss = F.mse_loss(pred, noise)
        loss.backward(); optimizer.step()
        return loss.item()

    loader = DataLoader(
        ColorMapDepthDataset(args.data_path, args.depth_dir, args.label_map_dir, args.max_train_samples),
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
            colour_cn.save_pretrained(out)                       # colour branch is the trained one
            save_lora(os.path.join(out, "lora"))
            if args.train_depth:
                depth_cn.save_pretrained(os.path.join(out, "depth_cn"))
            exp_utils.prune_checkpoints(run_dir, args.keep_last_n)
            log.info(f"  saved {out}")

    final = os.path.join(ckpt_dir, "final")
    colour_cn.save_pretrained(final)
    save_lora(os.path.join(final, "lora"))
    if args.train_depth:
        depth_cn.save_pretrained(os.path.join(final, "depth_cn"))
    log.info(f"Done. Final checkpoint + config.json in {run_dir}")


if __name__ == "__main__":
    main()
