import torch
import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from data.meldataset import mel_spectrogram
from model.tokenizer import convert_audio
import torch.nn.functional as F

N_FFT = 1024
NUM_MELS = 100
SAMPLING_RATE = 24000
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = None

# Construct train, dev, test dataset

class Chinesedataset(torch.utils.data.Dataset):
    def __init__(self, audio_file):
        with open(audio_file, "r") as f_read:
            lines = f_read.readlines()

        self.data = []
        for line in lines:
            wav_path, text = line.strip().split("\t")
            #if not debug:
                #wav_path = wav_path.replace("apdcephfs_cq10", "apdcephfs_cq10_1297902")
            #wav_path = wav_path.replace("apdcephfs_cq10", "apdcephfs_cq10_1297902")
            self.data.append((wav_path, text))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        wav_path, text = self.data[index]
        waveform, sr = librosa.load(wav_path, sr=16000)
        target_sr = 24000
        waveform = torch.tensor(waveform)
        waveform = waveform.unsqueeze(0)
        convert_waveform = convert_audio(waveform, sr, target_sr, 1)
        target_length = target_sr * 8
        convert_waveform = convert_waveform.squeeze()
        if target_length > len(waveform):
            padding_length = target_length - len(convert_waveform)
            #padded_audio = np.pad(waveform, (0, padding_length), 'constant')
            padded_audio = F.pad(convert_waveform, (0, padding_length), mode='constant', value=0)
        else:
            padding_length = 0
            padded_audio = convert_waveform[:target_length]
        return padded_audio, text, padding_length
        
def custom_collate_fn(batch):
    waveforms = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    padding_length_list = [item[2] for item in batch]

    padded_sequence = pad_sequence(waveforms, batch_first=True, padding_value=0)

    spectrograms = [mel_spectrogram(wav.unsqueeze(0), N_FFT, NUM_MELS, SAMPLING_RATE, HOP_SIZE, WIN_SIZE, FMIN, FMAX) for wav in waveforms]
    spectrograms = torch.stack(spectrograms)
    spectrograms = spectrograms.permute(0,1,3,2)

    audio_embedding_mask = []
    for padding_length in padding_length_list:
        n = round(padding_length / 24000 * 75)
        audio_embedding_mask.append([True] * (600 - n) + [False] * n)
    audio_embedding_mask = torch.tensor(audio_embedding_mask)

    mel_mask = []
    for padding_length in padding_length_list:
        n = round(padding_length / 24000 * 93.75)
        mel_mask.append([[True] * (750 - n) + [False] * n] * 100)
    mel_mask = torch.tensor(mel_mask)


    return padded_sequence, spectrograms, audio_embedding_mask, mel_mask
        
