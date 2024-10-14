import soundfile as sf
import argparse

def get_audio_duration(file_path): 
    with sf.SoundFile(file_path) as audio_file: 
        duration = len(audio_file) / audio_file.samplerate 
        return duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_list", type=str, help="Path of str list")
    args = parser.parse_args()
    total_duration = 0
    with open(args.wav_list, "r") as f_read:
        for line in f_read.readlines():
            if len(line.split()) != 2:
                print(line)
                break
            file_path, text = line.split()
            total_duration += get_audio_duration(file_path)

    total_duration /= 3600
    print("total duration: {} hours".format(total_duration))

if __name__ == "__main__":
    main()