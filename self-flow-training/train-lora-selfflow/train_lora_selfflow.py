"""Stage 1: UNet LoRA trained with the Self-Flow objective. REP_GAMMA = 0 gives the
matched plain-DDPM baseline arm."""

import os
import shutil
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from datasets import load_from_disk

MODEL_ID = "runwayml/stable-diffusion-v1-5"
WS = "/data/horse/ws/faqa581h-dsad_workspace"

DATA = f"{WS}/multi_dsad"
INIT_LORA = ""
OUT = f"{WS}/model_checkpoints/lora_selfflow_c1"

RESOLUTION = 512
PROMPT = ""
MAX_TRAIN_SAMPLES = None
AUGMENT_FLIP = True

EPOCHS = 20
BATCH_SIZE = 16
GRAD_ACCUM = 1
LORA_LR = 1e-4
PROJ_LR = 1e-4
WEIGHT_DECAY = 1e-2
RANK = 8
NUM_WORKERS = 8
SEED = 42
MIXED_PRECISION = "bf16"

MASK_RATIO = 0.25
REP_GAMMA = 0.5
EMA_DECAY = 0.999

SAVE_EVERY = 1
KEEP_LAST_N = 0
LOG_EVERY = 10

STUDENT_ADAPTER = "default"
TEACHER_ADAPTER = "ema"
SELF_FLOW_ENABLED = REP_GAMMA > 0.0


def to_three_channels(img):
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    if img.shape[0] == 4:
        return img[:3, :, :]
    return img


class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_samples=None, augment_flip=True):
        self.dataset = load_from_disk(data_path)["train"]
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        tf = [transforms.Resize((RESOLUTION, RESOLUTION))]
        if augment_flip:
            tf.append(transforms.RandomHorizontalFlip(p=0.5))
        tf += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.image_transforms = transforms.Compose(tf)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.image_transforms(self.dataset[idx]["image"])
        return {"pixel_values": to_three_channels(img)}


def load_lora_into_adapter(unet, lora_dir, adapter_name=STUDENT_ADAPTER):
    from safetensors.torch import load_file
    fp = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(fp):
        fp = os.path.join(lora_dir, "pytorch_lora_weights.bin")
    state = load_file(fp) if fp.endswith(".safetensors") else torch.load(fp, map_location="cpu")
    unet_sd = {k[len("unet."):]: v for k, v in state.items() if k.startswith("unet.")}
    set_peft_model_state_dict(unet, unet_sd, adapter_name=adapter_name)


def save_lora_weights(unet, save_dir, adapter_name=STUDENT_ADAPTER):
    os.makedirs(save_dir, exist_ok=True)
    unet.set_adapter(adapter_name)
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name=adapter_name))
    StableDiffusionControlNetPipeline.save_lora_weights(save_directory=save_dir, unet_lora_layers=sd)


def prune_checkpoints(ckpt_dir, keep_last_n):
    if keep_last_n <= 0:
        return
    epoch_dirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("epoch_")],
                        key=lambda d: int(d.split("_")[1]))
    for d in epoch_dirs[:-keep_last_n]:
        shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)


def lora_adapter_params(unet, adapter_name):
    """Student and teacher are two adapters on the same UNet, selected by set_adapter."""
    params = dict(unet.named_parameters())
    return {name: p for name, p in params.items()
            if "lora_" in name and f".{adapter_name}." in name}


def paired_lora_params(unet, student_name=STUDENT_ADAPTER, teacher_name=TEACHER_ADAPTER):
    params = dict(unet.named_parameters())
    pairs = []
    for name, student in lora_adapter_params(unet, student_name).items():
        teacher = params.get(name.replace(f".{student_name}.", f".{teacher_name}."))
        if teacher is not None:
            pairs.append((student, teacher))
    return pairs


@torch.no_grad()
def sync_teacher_to_student(unet):
    for student, teacher in paired_lora_params(unet):
        teacher.copy_(student)


@torch.no_grad()
def ema_update_lora(unet, decay):
    for student, teacher in paired_lora_params(unet):
        teacher.mul_(decay).add_(student.detach(), alpha=1.0 - decay)


@torch.no_grad()
def ema_drift(unet):
    diffs = [(student.detach() - teacher.detach()).abs().mean().item()
             for student, teacher in paired_lora_params(unet)]
    return sum(diffs) / len(diffs) if diffs else 0.0


def add_noise(latents, noise, alpha_bar):
    alpha_bar = alpha_bar.to(latents.dtype)
    return (alpha_bar.sqrt() * latents + (1.0 - alpha_bar).sqrt() * noise).to(latents.dtype)


def dual_timestep_noise(latents, noise, alphas_cumprod, num_timesteps, mask_ratio):
    """Student sees x_tau, a mix of two noise levels; teacher sees the cleaner x_tmin."""
    b, _, h, w = latents.shape
    t_ctx = torch.randint(0, num_timesteps, (b,), device=latents.device)
    t_cor = torch.randint(0, num_timesteps, (b,), device=latents.device)

    corrupted = torch.rand(b, 1, h, w, device=latents.device) < mask_ratio
    tau = torch.where(corrupted, t_cor.view(b, 1, 1, 1), t_ctx.view(b, 1, 1, 1))
    t_min = torch.minimum(t_ctx, t_cor)

    noisy = add_noise(latents, noise, alphas_cumprod[tau])
    noisy_min = add_noise(latents, noise, alphas_cumprod[t_min].view(b, 1, 1, 1))
    return noisy, t_ctx, noisy_min, t_min


class ProjectionHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.SiLU(), nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        return self.net(x)


def representation_loss(f_student, f_teacher, proj_head):
    fs = proj_head(f_student.float())
    b, c, h, w = fs.shape
    fs = F.normalize(fs.permute(0, 2, 3, 1).reshape(b, h * w, c).float(), dim=-1)
    ft = F.normalize(f_teacher.permute(0, 2, 3, 1).reshape(b, h * w, c).float().detach(), dim=-1)
    return -(fs * ft).sum(-1).mean()


def get_logger(out_dir):
    logger = logging.getLogger("selfflow_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(out_dir, "train.log")); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    ckpt_dir = os.path.join(OUT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log = get_logger(OUT)
    t_start = time.perf_counter()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[MIXED_PRECISION]
    use_amp = autocast_dtype is not None and device.type == "cuda"

    tag = "Self-Flow C1 (LoRA, gamma={:.2f}, ema_decay={})".format(REP_GAMMA, EMA_DECAY) if SELF_FLOW_ENABLED \
        else "Plain-DDPM LoRA baseline (Arm-B, REP_GAMMA=0)"
    log.info(f"{tag} | device={device} | amp={use_amp} ({MIXED_PRECISION})")

    train_dataset = ImageOnlyDataset(DATA, MAX_TRAIN_SAMPLES, AUGMENT_FLIP)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    steps_per_epoch = len(train_loader)
    log.info(f"dataset: {len(train_dataset)} samples | {steps_per_epoch} steps/epoch | {EPOCHS} epochs")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    lora_config = LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian",
                             target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(lora_config, adapter_name=STUDENT_ADAPTER)
    if os.path.isdir(INIT_LORA):
        load_lora_into_adapter(unet, INIT_LORA, adapter_name=STUDENT_ADAPTER)

    proj_head = None
    if SELF_FLOW_ENABLED:
        unet.add_adapter(lora_config, adapter_name=TEACHER_ADAPTER)
        sync_teacher_to_student(unet)
        for param in lora_adapter_params(unet, TEACHER_ADAPTER).values():
            param.requires_grad_(False)

        proj_head = ProjectionHead(unet.config.block_out_channels[-1]).to(device)
        feat = {}
        unet.mid_block.register_forward_hook(lambda _m, _i, out: feat.update(mid=out))
    unet.set_adapter(STUDENT_ADAPTER)

    vae.to(device); text_encoder.to(device); unet.to(device)
    vae.eval(); text_encoder.eval(); unet.train()

    lora_params = list(lora_adapter_params(unet, STUDENT_ADAPTER).values())
    param_groups = [{"params": lora_params, "lr": LORA_LR}]
    if SELF_FLOW_ENABLED:
        param_groups.append({"params": list(proj_head.parameters()), "lr": PROJ_LR})
    log.info(f"trainable LoRA(default): {sum(p.numel() for p in lora_params):,}"
             + (f" | proj: {sum(p.numel() for p in proj_head.parameters()):,}" if SELF_FLOW_ENABLED else ""))
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    num_timesteps = noise_scheduler.config.num_train_timesteps

    with torch.no_grad():
        ids = tokenizer([PROMPT], padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        empty_embed = text_encoder(ids)[0]

    log.info(f"setup done in {time.perf_counter()-t_start:.1f}s -- starting training")

    def predict_noise(noisy, timesteps, prompt_embeds):
        return unet(noisy, timesteps, encoder_hidden_states=prompt_embeds).sample

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_gen, epoch_rep = 0.0, 0.0
        epoch_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            images = batch["pixel_values"].to(device, non_blocking=True)
            bsz = images.shape[0]
            prompt_embeds = empty_embed.expand(bsz, -1, -1)

            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)

            unet.set_adapter(STUDENT_ADAPTER)
            t_gen = torch.randint(0, num_timesteps, (bsz,), device=device)
            noisy_gen = add_noise(latents, noise, alphas_cumprod[t_gen].view(bsz, 1, 1, 1))
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                l_gen = F.mse_loss(predict_noise(noisy_gen, t_gen, prompt_embeds).float(), noise.float())

            if SELF_FLOW_ENABLED:
                noisy_het, t_ctx, noisy_min, t_min = dual_timestep_noise(
                    latents, noise, alphas_cumprod, num_timesteps, MASK_RATIO)
                unet.set_adapter(STUDENT_ADAPTER)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                    predict_noise(noisy_het, t_ctx, prompt_embeds)
                f_student = feat["mid"]

                unet.set_adapter(TEACHER_ADAPTER)
                with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                    predict_noise(noisy_min, t_min, prompt_embeds)
                    f_teacher = feat["mid"].detach()
                unet.set_adapter(STUDENT_ADAPTER)

                l_rep = representation_loss(f_student, f_teacher, proj_head)
            else:
                l_rep = torch.zeros((), device=device)

            loss = l_gen + REP_GAMMA * l_rep
            (loss / GRAD_ACCUM).backward()
            epoch_gen += l_gen.item()
            epoch_rep += l_rep.item()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if SELF_FLOW_ENABLED:
                    ema_update_lora(unet, EMA_DECAY)
                global_step += 1

            if step % LOG_EVERY == 0:
                log.info(f"E{epoch} S{step}/{steps_per_epoch} g{global_step} | "
                         f"Lgen {l_gen.item():.4f} Lrep {l_rep.item():.4f}")

        drift_line = f" | ema_drift {ema_drift(unet):.5f}" if SELF_FLOW_ENABLED else ""
        log.info(f"Epoch {epoch} done in {time.perf_counter()-epoch_t0:.1f}s | "
                 f"avg Lgen {epoch_gen/steps_per_epoch:.4f} | avg Lrep {epoch_rep/steps_per_epoch:.4f}{drift_line}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
            save_lora_weights(unet, epoch_dir)
            prune_checkpoints(ckpt_dir, KEEP_LAST_N)
            log.info(f"  saved {epoch_dir}")

    final = os.path.join(ckpt_dir, "final")
    save_lora_weights(unet, final)
    log.info(f"finished in {time.perf_counter()-t_start:.1f}s total. Final checkpoint in {final}")


if __name__ == "__main__":
    main()
