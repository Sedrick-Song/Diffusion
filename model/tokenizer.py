# cp from https://github.com/lifeiteng/vall-e/blob/main/valle/data/tokenizer.py
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import torch
import torchaudio
# from lhotse.features import FeatureExtractor
# from lhotse.utils import Seconds, compute_num_frames

class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
        signature = None
    ) -> None:
        from src.audiocraft.audiocraft.solvers import CompressionSolver
        model = CompressionSolver.model_from_checkpoint(signature)
        self.sample_rate = model.sample_rate
        self.channels = model.channels
        
        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self.codec.encode(wav.to(self.device))
        return [(codes[0], None)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames[0][0] # [1,4,T]
        return self.codec.decode(frames)
    
    def get_latent_embedding(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames[0][0] # [1,4,T]
        return self.codec.decode_latent(frames)
    
def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str, offset = -1, num_frames=-1):
    # Load and pre-process the audio waveform
    if offset != -1 and num_frames!=-1:
        wav, sr = torchaudio.load(audio_path, frame_offset=offset, num_frames=num_frames)
    else:
        wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames
