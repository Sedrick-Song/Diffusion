import os
import re
import string

dir_list = ["MDT-ASR-D025", "MDT-ASR-E001", "MDT-ASR-F055", "MDT-ASR-F056", "MDT-ASR-F057", "MDT-ASR-F062", "MDT-ASR-F063", "MDT-ASR-F064", "MDT-ASR-G028"]
data_path = "/apdcephfs_cq10/share_1297902/data/speech_data/single_data/tts/medium_quality/magic_data"
output_file = "/data/home/sedricksong/data/zh_magic_data/wav_text"

def remove_punctuation_and_whitespace(input_string):
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    result = re.sub(pattern, '', input_string)
    return result

with open(output_file, 'w') as f_write:
    for dir in dir_list:
        if "UTTERANCEINFO.txt" in os.listdir(os.path.join(data_path,dir)):
            file = os.path.join(data_path, dir, "UTTERANCEINFO.txt")
        else:
            file = os.path.join(data_path, dir, "UTTRANSINFO.txt")
        with open(file, "r") as f_read:
            next(f_read)
            for line in f_read.readlines():
                info_list = line.split("\t")
                id, speaker, text = info_list[1], info_list[2], info_list[4]
                audio_path = os.path.join(data_path, dir, "WAV", speaker, id)
                clean_text = remove_punctuation_and_whitespace(text).strip()
                if clean_text == "":
                    continue
                f_write.write(audio_path + "\t" + clean_text + "\n")


            