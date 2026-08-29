import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ==========================================
# 1. INITIALIZATION & HARDWARE SETUP
# ==========================================
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-canny" # Baseline architecture

print("Loading models in 16-bit precision...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.bfloat16)
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.bfloat16)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# ==========================================
# 2. MEMORY OPTIMIZATION (24GB GPU LIMIT)
# ==========================================
print("Locking Stable Diffusion backbone...")
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
controlnet.train()

# Enable Gradient Checkpointing to save VRAM
controlnet.enable_gradient_checkpointing()

# Move everything to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)
text_encoder.to(device)
unet.to(device)
controlnet.to(device)

# Optimizer strictly targets the trainable path
optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

# ==========================================
# 3. TEXT-DROP LOGIC (100% SPATIAL LEARNING)
# ==========================================
def get_empty_prompt_embeds(batch_size):
    """Generates empty language embeddings to force reliance on surgical masks."""
    empty_tokens = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    with torch.no_grad():
        empty_embeds = text_encoder(empty_tokens)[0]
    return empty_embeds

# ==========================================
# 4. THE TRAINING STEP
# ==========================================
def training_step(batch):
    """
    Executes one forward pass using ONLY spatial conditioning.
    """
    optimizer.zero_grad()

    # Extract tensors from Dataloader
    images = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    masks = batch["conditioning_pixel_values"].to(device, dtype=torch.bfloat16)
    batch_size = images.shape[0]

    if masks.shape[1] == 1:
        masks = masks.repeat(1, 3, 1, 1)

    # 1. Encode target surgical frames into latent space
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # 2. Sample noise and timesteps (Phase 1: Epsilon Prediction)
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 3. Amputate language conditioning
    encoder_hidden_states = get_empty_prompt_embeds(batch_size)

    # 4. Forward pass through ControlNet
    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=masks,
        return_dict=False,
    )

    # 5. Forward pass through locked UNet
    # 5. Forward pass through locked UNet
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    ).sample

    # 6. Calculate MSE Loss (predicting the noise)
    loss = F.mse_loss(model_pred, noise, reduction="mean")
    
    # 7. Backpropagate
    loss.backward()
    optimizer.step()

    return loss.item()

print("Generative Baseline Architecture successfully initialized!")
# ==========================================
# 5. DATALOADER & LOCAL REAL-DATA DRY RUN
# ==========================================
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms

class RealDSADControlNetDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Pointing directly to Fardeen's downloaded dataset on the scratch drive
        data_path = "/data/cat/ws/faqa581h-dsad_workspace/dsad_dataset"
        self.dataset = load_from_disk(data_path)["train"]
        
        self.image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() 
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Apply base transforms
        img = self.image_transforms(item["image"])
        mask = self.mask_transforms(item["mask"])
        
        # 2. Force images to strictly 3 channels (RGB)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4: # Fallback just in case of RGBA pngs
            img = img[:3, :, :]
            
        # 3. Force masks to strictly 3 channels
        if mask.shape[0] == 1:
            mask = mask.repeat(3, 1, 1)
        elif mask.shape[0] == 4:
            mask = mask[:3, :, :]
            
        return {
            "pixel_values": img,
            "conditioning_pixel_values": mask
        }

if __name__ == "__main__":
    import os
    import time  # <-- Add this import
    
    print("\n--- Starting Full Training (100 Epochs) ---")
    
    # 1. Initialize dataset and dataloader
    dataset = RealDSADControlNetDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    
    num_epochs = 100  # <-- Updated to 100
    
    # Start the total training timer
    total_start_time = time.time() 
    
    # 2. The Full Training Loop
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_start_time = time.time()  # <-- Start epoch timer
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            loss = training_step(batch)
            epoch_loss += loss
            
            # Print an update every 500 steps (reduced frequency for long runs)
            if step % 500 == 0:
                print(f"Epoch {epoch + 1} | Step {step}/{len(dataloader)} | MSE Loss: {loss:.4f}")
                
        # Calculate average loss and time for the epoch
        avg_loss = epoch_loss / len(dataloader)
        epoch_end_time = time.time()
        epoch_duration_mins = (epoch_end_time - epoch_start_time) / 60
        
        print(f"✅ Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f} | Time: {epoch_duration_mins:.2f} mins")
        
        # 3. Save the model checkpoint safely
        save_path = f"./controlnet_checkpoints/epoch_{epoch + 1}"
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving ControlNet weights to {save_path}...")
        controlnet.save_pretrained(save_path)
        
    # Calculate and print the total training time
    total_end_time = time.time()
    total_duration_hours = (total_end_time - total_start_time) / 3600
    print(f"\n🎉 Training Complete! Total time for {num_epochs} Epochs: {total_duration_hours:.2f} hours")
