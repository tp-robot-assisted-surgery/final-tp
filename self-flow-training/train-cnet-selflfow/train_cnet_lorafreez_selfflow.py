"""Self-Flow training of a segmentation-mask ControlNet over a frozen, warm-started UNet LoRA."""

import os
import sys
import copy
import shutil
import time
import logging
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
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
LABEL_MAP_DIR = None

CNET_RUN = f"{WS}/experiments/20260629_202928_multiorgan_colormap_lora_depth_trained/checkpoints/final"
INIT_CNET = "__init__"
INIT_LORA = os.path.join(CNET_RUN, "lora")
CNET_INIT_FALLBACK = "lllyasviel/control_v11p_sd15_seg"

OUT = f"{WS}/model_checkpoints/cnet_selfflow_frozen_lora"

RESOLUTION = 512
PROMPT = ""
MAX_TRAIN_SAMPLES = None

EPOCHS = 20
BATCH_SIZE = 16
GRAD_ACCUM = 1
CONTROLNET_LR = 1e-5
PROJ_LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 8
SEED = 42
MIXED_PRECISION = "bf16"

MASK_RATIO = 0.25
REP_GAMMA = 0.5
EMA_DECAY = 0.9999

SAVE_EVERY = 1
KEEP_LAST_N = 0
LOG_EVERY = 10

LORA_RANK = 8
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

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
    label = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
    for key in PAINT_ORDER:
        if key not in item or item[key] is None:
            continue
        mask = item[key].convert("L").resize((RESOLUTION, RESOLUTION), Image.NEAREST)
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


class ColorMapDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_map_dir=None, max_samples=None):
        self.dataset = load_from_disk(data_path)["train"]
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.label_map_dir = label_map_dir
        self.image_transforms = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def _load_label_map(self, item, idx):
        if not self.label_map_dir:
            return build_label_map_from_columns(item)
        path = os.path.join(self.label_map_dir, f"{idx:04d}.png")
        return np.array(Image.open(path).convert("L").resize((RESOLUTION, RESOLUTION), Image.NEAREST))

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "pixel_values": to_three_channels(self.image_transforms(item["image"])),
            "conditioning_pixel_values": colorize(self._load_label_map(item, idx)),
        }


def load_lora_into_adapter(unet, lora_dir, adapter_name="default"):
    from safetensors.torch import load_file
    try:
        from diffusers.utils import convert_unet_state_dict_to_peft
    except Exception:
        convert_unet_state_dict_to_peft = None

    fp = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(fp):
        fp = os.path.join(lora_dir, "pytorch_lora_weights.bin")
    state = load_file(fp) if fp.endswith(".safetensors") else torch.load(fp, map_location="cpu")

    unet_sd = {k[len("unet."):]: v for k, v in state.items() if k.startswith("unet.")}
    if convert_unet_state_dict_to_peft is not None:
        unet_sd = convert_unet_state_dict_to_peft(unet_sd)
    res = set_peft_model_state_dict(unet, unet_sd, adapter_name=adapter_name)

    missing = list(getattr(res, "missing_keys", []) or [])
    model_lora_keys = [n for n, _ in unet.named_parameters()
                       if "lora_" in n and f".{adapter_name}." in n]
    missing_lora = [k for k in missing if "lora_" in k and f".{adapter_name}." in k]
    loaded = len(model_lora_keys) - len(missing_lora)
    print(f"  loaded LoRA '{adapter_name}' from {lora_dir}: "
          f"{loaded}/{len(model_lora_keys)} tensors set (missing_lora={len(missing_lora)})")
    if missing_lora:
        print(f"  WARNING: {len(missing_lora)} LoRA tensors NOT in checkpoint — warm-start PARTIAL. "
              f"First few: {missing_lora[:5]}")


def load_controlnet(init, fallback):
    try:
        print(f"Loading ControlNet from: {init}")
        return ControlNetModel.from_pretrained(init)
    except Exception as e:
        print(f"WARNING: could not load {init} ({e}) — falling back to {fallback}")
        return ControlNetModel.from_pretrained(fallback)


def freeze_lora(unet):
    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.requires_grad_(False)


def save_lora_weights(unet, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    unet.set_adapter("default")
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name="default"))
    StableDiffusionControlNetPipeline.save_lora_weights(save_directory=save_dir, unet_lora_layers=sd)


def prune_checkpoints(ckpt_dir, keep_last_n):
    if keep_last_n <= 0:
        return
    epoch_dirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("epoch_")],
                        key=lambda d: int(d.split("_")[1]))
    for d in epoch_dirs[:-keep_last_n]:
        shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)


@torch.no_grad()
def ema_update(student_cn, teacher_cn, decay):
    for teacher_param, student_param in zip(teacher_cn.parameters(), student_cn.parameters()):
        teacher_param.mul_(decay).add_(student_param.detach(), alpha=1.0 - decay)


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


class ElapsedFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.t0 = time.perf_counter()

    def filter(self, record):
        record.relsec = time.perf_counter() - self.t0
        return True


def get_logger(out_dir):
    logger = logging.getLogger("selfflow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | +%(relsec).1fs | %(message)s", "%H:%M:%S")
    fmt.converter = time.localtime
    elapsed = ElapsedFilter()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(out_dir, "train.log"))):
        handler.setFormatter(fmt)
        handler.addFilter(elapsed)
        logger.addHandler(handler)
    return logger


@contextlib.contextmanager
def timed(log, label):
    t0 = time.perf_counter()
    yield
    log.info(f"  {label:<22} {time.perf_counter() - t0:.1f}s")


def sync_now():
    """perf_counter after a CUDA sync, since unsynchronized kernel timings are meaningless."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def lap(mark):
    now = sync_now()
    return now - mark, now


def main():
    os.makedirs(OUT, exist_ok=True)
    ckpt_dir = os.path.join(OUT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log = get_logger(OUT)
    t_start = time.perf_counter()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[MIXED_PRECISION]
    use_amp = autocast_dtype is not None and device.type == "cuda"
    log.info("Self-Flow C2 | single ControlNet (segmentation mask) trainable | LoRA FROZEN (warm-started)")
    log.info(f"device={device} | cuda={torch.cuda.is_available()} | amp={use_amp} ({MIXED_PRECISION})")
    if device.type == "cpu":
        log.info("WARNING: running on CPU — expect this to be VERY slow; use a GPU node for real runs.")

    t0 = time.perf_counter()
    train_dataset = ColorMapDataset(DATA, LABEL_MAP_DIR, MAX_TRAIN_SAMPLES)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    steps_per_epoch = len(train_loader)
    log.info(f"dataset ready: {len(train_dataset)} samples | {steps_per_epoch} steps/epoch "
             f"| load_from_disk {time.perf_counter()-t0:.1f}s")

    log.info("loading backbone + checkpoints...")
    with timed(log, "tokenizer"):
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    with timed(log, "text_encoder"):
        text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    with timed(log, "vae"):
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    with timed(log, "unet"):
        unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    with timed(log, "controlnet"):
        controlnet = load_controlnet(INIT_CNET, CNET_INIT_FALLBACK)

    with timed(log, "EMA teacher copy"):
        ema_controlnet = copy.deepcopy(controlnet)
        ema_controlnet.requires_grad_(False)
        ema_controlnet.eval()

    with timed(log, "LoRA adapter (frozen)"):
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        unet.add_adapter(LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_RANK, init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        ))
        if os.path.isdir(INIT_LORA):
            load_lora_into_adapter(unet, INIT_LORA, adapter_name="default")
        else:
            log.info(f"WARNING: {INIT_LORA} not found — LoRA starts from fresh gaussian init")
        freeze_lora(unet)
        unet.set_adapter("default")

    controlnet.train()
    controlnet.enable_gradient_checkpointing()

    proj_head = ProjectionHead(unet.config.block_out_channels[-1])
    feat = {}
    unet.mid_block.register_forward_hook(lambda _m, _i, out: feat.update(mid=out))

    with timed(log, f"moved to {device}"):
        for module in (vae, text_encoder, unet, controlnet, ema_controlnet, proj_head):
            module.to(device)
        vae.eval()
        text_encoder.eval()
        unet.train()
        sync_now()
    if device.type == "cuda":
        log.info(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    cnet_params = [p for p in controlnet.parameters() if p.requires_grad]
    proj_params = list(proj_head.parameters())
    lora_frozen = sum(p.numel() for n, p in unet.named_parameters() if "lora_" in n)
    log.info(f"trainable  controlnet: {sum(p.numel() for p in cnet_params):,} | "
             f"proj: {sum(p.numel() for p in proj_params):,} | "
             f"LoRA (frozen): {lora_frozen:,}")
    optimizer = torch.optim.AdamW(
        [
            {"params": cnet_params, "lr": CONTROLNET_LR},
            {"params": proj_params, "lr": PROJ_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    num_timesteps = noise_scheduler.config.num_train_timesteps

    with torch.no_grad():
        ids = tokenizer([PROMPT], padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        empty_embed = text_encoder(ids)[0]

    log.info(f"SETUP DONE in {time.perf_counter()-t_start:.1f}s — starting training "
             f"({EPOCHS} epochs x {steps_per_epoch} steps)")

    def predict_noise(cn_module, noisy, timesteps, prompt_embeds, cond):
        down, mid = cn_module(
            noisy, timesteps, encoder_hidden_states=prompt_embeds,
            controlnet_cond=cond,
            return_dict=False,
        )
        return unet(
            noisy, timesteps, encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        ).sample

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_gen, epoch_rep = 0.0, 0.0
        epoch_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()

        for step, batch in enumerate(train_loader):
            images = batch["pixel_values"].to(device, non_blocking=True)
            cond = batch["conditioning_pixel_values"].to(device, non_blocking=True)
            bsz = images.shape[0]
            prompt_embeds = empty_embed.expand(bsz, -1, -1)
            dt_data = sync_now() - step_start

            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            mark = sync_now()

            t_gen = torch.randint(0, num_timesteps, (bsz,), device=device)
            noisy_gen = add_noise(latents, noise, alphas_cumprod[t_gen].view(bsz, 1, 1, 1))
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                model_pred = predict_noise(controlnet, noisy_gen, t_gen, prompt_embeds, cond)
                l_gen = F.mse_loss(model_pred.float(), noise.float())
            dt_gen, mark = lap(mark)

            noisy_het, t_ctx, noisy_min, t_min = dual_timestep_noise(
                latents, noise, alphas_cumprod, num_timesteps, MASK_RATIO)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                predict_noise(controlnet, noisy_het, t_ctx, prompt_embeds, cond)
            f_student = feat["mid"]
            dt_student, mark = lap(mark)

            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                predict_noise(ema_controlnet, noisy_min, t_min, prompt_embeds, cond)
                f_teacher = feat["mid"].detach()
            dt_teacher, mark = lap(mark)

            l_rep = representation_loss(f_student, f_teacher, proj_head)
            loss = l_gen + REP_GAMMA * l_rep

            (loss / GRAD_ACCUM).backward()
            epoch_gen += l_gen.item()
            epoch_rep += l_rep.item()
            dt_backward, mark = lap(mark)

            dt_optimizer = 0.0
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema_update(controlnet, ema_controlnet, EMA_DECAY)
                global_step += 1
                dt_optimizer, mark = lap(mark)

            if step % LOG_EVERY == 0:
                dt_step = time.perf_counter() - step_start
                log.info(f"E{epoch} S{step}/{steps_per_epoch} g{global_step} | "
                         f"Lgen {l_gen.item():.4f} Lrep {l_rep.item():.4f} | "
                         f"step {dt_step:.2f}s [data {dt_data:.2f} gen {dt_gen:.2f} "
                         f"repS {dt_student:.2f} repT {dt_teacher:.2f} "
                         f"bwd {dt_backward:.2f} opt {dt_optimizer:.2f}]")
            step_start = time.perf_counter()

        log.info(f"Epoch {epoch} done in {time.perf_counter()-epoch_t0:.1f}s | "
                 f"avg L_gen {epoch_gen/steps_per_epoch:.4f} | avg L_rep {epoch_rep/steps_per_epoch:.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            t0 = time.perf_counter()
            epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
            unet.set_adapter("default")
            controlnet.save_pretrained(epoch_dir)
            save_lora_weights(unet, os.path.join(epoch_dir, "lora"))
            prune_checkpoints(ckpt_dir, KEEP_LAST_N)
            log.info(f"  saved {epoch_dir}  ({time.perf_counter()-t0:.1f}s)")

    final = os.path.join(ckpt_dir, "final")
    unet.set_adapter("default")
    controlnet.save_pretrained(final)
    save_lora_weights(unet, os.path.join(final, "lora"))
    log.info(f"Self-Flow (single CNet) finished in {time.perf_counter()-t_start:.1f}s total. "
             f"Final checkpoint in {final}")


if __name__ == "__main__":
    main()
