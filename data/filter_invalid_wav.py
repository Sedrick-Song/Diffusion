import soundfile as sf

def check_wav_file(file_path):
    """Checks if a wav file is valid."""
    try:
        with sf.SoundFile(file_path) as file:
            return True
    except RuntimeError:
        return False

def process_wav_list(input_file, output_file):
    """Processes a wav list file and writes valid entries to an output file."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue

            wav_path, transcription = parts
            if check_wav_file(wav_path):
                outfile.write(line)
            else:
                print(f"Invalid WAV file: {wav_path}")

# Replace 'wav_list.txt' and 'valid_wav_list.txt' with your actual file names
process_wav_list('wav_text', 'valid_wav_text')