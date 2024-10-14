import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import time
from src.diffusers.utils.torch_utils import randn_tensor
from src.diffusers import DDPMScheduler, UNet2DConditionModel, Transformer2DModel


class AudioDiffusion(nn.Module):
    def __init__(
        self,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        uncondition=False,

    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.projection = nn.Embedding(1024,1024)

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained("/apdcephfs_cq8/private_sedricksong/temp", subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained("/apdcephfs_cq8/private_sedricksong/temp", subfolder="scheduler")

        if unet_model_config_path:
            #unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
            #print("unet_config", unet_config)
            #self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
            unet_config = Transformer2DModel.load_config(unet_model_config_path)
            self.unet = Transformer2DModel.from_config(unet_config, subfolder="unet")
            self.set_from = "random"
        else:
            self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
            self.set_from = "pre-trained"
            self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward(self, latents, encoder_hidden_states, boolean_encoder_mask, mel_mask, validation_mode=False):
        device = self.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        #encoder_hidden_states = self.projection(encoder_hidden_states)

        #print("noisy_latents shape", noisy_latents.size())
        if self.set_from == "random":
            model_pred = self.unet(
                hidden_states=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=boolean_encoder_mask
            ).sample

        elif self.set_from == "pre-trained":
            compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            model_pred = self.unet(
                hidden_states=compressed_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=boolean_encoder_mask
            ).sample
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        #print("model_pred shape", model_pred.size())
        #model_pred = model_pred.reshape(-1, 8, 256, 16)

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float() * mel_mask, target.float() * mel_mask, reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float() * mel_mask, target.float() * mel_mask, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def inference(self, prompt_embeds, boolean_prompt_mask, inference_scheduler, num_steps=20, guidance_scale=1, num_samples_per_prompt=1, 
                  disable_progress=True):
        start = time.time()
        device = self.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1

        #print("ldm time 1", time.time()-start)
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        #print("ldm time 2", time.time()-start, timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            tim = torch.tensor([t]*batch_size*2).to(device) if classifier_free_guidance else torch.tensor([t]*batch_size).to(device)
            #print("ldm emu", i, time.time()-start)
            noise_pred = self.unet(
                hidden_states=latent_model_input, timestep=tim, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        #print("ldm time 3", time.time()-start)
        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 750, 100)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
                
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]
                
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask



