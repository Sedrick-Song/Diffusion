import os
import argparse
import numpy as np
import torch
import torch.distributed
import torch.utils
import json
from scipy.io.wavfile import write

from utils.decoding import decode_one_audio
from data.chinese import Chinesedataset
from data.chinese import custom_collate_fn
from model.tokenizer import AudioTokenizer
from model.model import AudioDiffusion
from model.bigvgan import BigVGAN as Generator
from src.diffusers.models.transformer_2d import Transformer2DModel
from encodec.model import EncodecModel

from BigVGAN.env import AttrDict

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/apdcephfs_cq8/private_sedricksong/exp/encodec_DiT/epoch-3.pt", help="The checkpoint file")
    parser.add_argument('--bigvgan_ckpt', default='/apdcephfs_cq8/private_sedricksong/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt')
    parser.add_argument("--test_file", type=str, default="/apdcephfs_cq8/private_sedricksong/data/zh_magic_data/test_file_temp", help="The test file path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for decoding")
    parser.add_argument("--bandwidth", type=int, default=6, help="Encodec bandwidth")
    parser.add_argument("--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1", help="Scheduler identifier.")
    parser.add_argument("--unet_model_name", type=str, default=None, help="UNet model identifier from huggingface.co/models.")
    parser.add_argument("--unet_model_config", type=str, default="/apdcephfs_cq8/private_sedricksong/Encodec_DiT/configs/diffusion_model_config_dit_L_4.json", help="UNet model config json path.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--uncondition", action="store_true", default=False, help="10% uncondition for training.")
    args = parser.parse_args()
    return args

def decode(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Encodec model
    #encodec_fn = "/apdcephfs_cq10/share_1297902/user/sedricksong/pretrained_models/VoiceCraft/encodec_4cb2048_giga.th"
    #audio_tokenizer = AudioTokenizer(signature=encodec_fn)
    audio_tokenizer = EncodecModel.encodec_model_24khz()
    audio_tokenizer.set_target_bandwidth(args.bandwidth)
    # DiT model
    #dit_config = Transformer2DModel.load_config(args.dit_config)
    #model = Transformer2DModel.from_config(dit_config, subfolder="unet")
    model = AudioDiffusion(
        args.scheduler_name, args.unet_model_name, args.unet_model_config, args.snr_gamma, args.uncondition
    )

    model_ckpt = torch.load(args.ckpt)
    model.load_state_dict(model_ckpt["model"])

    config_file = os.path.join(os.path.split(args.bigvgan_ckpt)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    state_dict_g = torch.load(args.bigvgan_ckpt)
    generator = Generator(h, use_cuda_kernel=False).to(device)
    generator.load_state_dict(state_dict_g['generator'])

    audio_tokenizer.to(device)
    generator.to(device)
    model.to(device)

    dataset_test = Chinesedataset(args.test_file)

    # define dataloder
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers = 4,
        pin_memory = True,
        batch_size = args.batch_size,
        collate_fn = custom_collate_fn,
        shuffle = False
    )

    # decoding process
    for batch_idx, batch in enumerate(test_dataloader):
        generate_audio = decode_one_audio(
            encodec=audio_tokenizer,
            model=model,
            bigvgan=generator,
            batch=batch,
            device=device
        )
        output_file = os.path.join("/apdcephfs_cq8/private_sedricksong/data/encodec_dit_generate", f'{batch_idx}_generated.wav')
        write(output_file, h.sampling_rate, generate_audio)
    

def main():
    args = get_parse()
    decode(args=args)

if __name__ == "__main__":
    main()