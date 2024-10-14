import torch
from data.meldataset import MAX_WAV_VALUE

#Inference pipeline

def decode_one_audio(encodec, model, bigvgan, batch, device):
    waveforms, spectrogram, audio_mask, mel_mask = batch[0], batch[1], batch[2], batch[3]  # wavform B*T
    mel_mask = mel_mask.unsqueeze(1)
    waveforms = waveforms.unsqueeze(1) # waveform B*1*T
    waveforms = waveforms.to(device)
    discrete_token = encodec.encode(waveforms)
    latent_embedding = encodec.get_latent_embedding(discrete_token)
    latent_embedding = latent_embedding.permute(0,2,1).to(device)
    audio_mask = audio_mask.to(device)
    mel_mask = mel_mask.to(device)
    with torch.no_grad():
        pred_mel = model.inference(prompt_embeds=latent_embedding, boolean_prompt_mask=audio_mask, inference_scheduler=model.inference_scheduler, num_steps=50)
    pred_mel = pred_mel.squeeze(1)
    pred_mel = pred_mel.permute(0,2,1)
    pred_mel = pred_mel[:,:,mel_mask[0][0][0]]
    y_g_hat = bigvgan(pred_mel)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.detach().cpu().numpy().astype('int16')
    return audio

    
    

            
