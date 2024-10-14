import soundfile as sf
import argparse

def get_audio_duration(file_path): 
    with sf.SoundFile(file_path) as audio_file: 
        duration = len(audio_file) / audio_file.samplerate 
        return duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_list", type=str, help="Path of str list")
    parser.add_argument("--output_wav_list", type=str, help="Path of str list")
    args = parser.parse_args()
    with open(args.wav_list, "r") as f_read, open(args.output_wav_list, "w") as f_write:
        for line in f_read.readlines():
            file_path, text = line.strip().split()
            if len(text) >= 5:
                duration = get_audio_duration(file_path)
                if duration >= 2 and duration <= 8:
                    f_write.write(line)


if __name__ == "__main__":
    main()