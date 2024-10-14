from pydub import AudioSegment
import random
import os

def add_sound_to_speech(speech_path, sound_path, output_path, sound_reduction_db):
    speech = AudioSegment.from_file(speech_path)
    sound = AudioSegment.from_file(sound_path)

    insert_position = random.randint(len(speech) // 3, len(speech) // 2)
    random_duration = random.randint(1000,min(2000, len(speech)-insert_position))
    sound = sound[:random_duration]
    sound = sound - sound_reduction_db

    pre_insert = speech[:insert_position]
    post_insert = speech[insert_position:]

    mix_part = post_insert[:len(sound)].overlay(sound)
    mixed_part = pre_insert + mix_part + post_insert[len(sound):]

    mixed_part.export(output_path, format="wav")

    start = insert_position
    end = start + random_duration
    return start, end

Audioset_path = "/apdcephfs/private_sedricksong/AudioSet/audio"
audio_list = os.listdir(Audioset_path)
speech_file = "/apdcephfs/private_sedricksong/data/zh_magic_data/train_file_temp"
output_dir = "/apdcephfs/private_sedricksong/data/mix_data"
mix_data_file = "/apdcephfs/private_sedricksong/data/mix_data/data_info"
with open(speech_file, "r") as f, open(mix_data_file, "w") as f_write:
    for line in f.readlines():
        speech_path, text = line.strip().split("\t")
        temp = speech_path.split("/")
        sub_dir1, sub_dir2, filename = temp[-4], temp[-2], temp[-1]
        modified_filename = filename[:-4] + "_mix" + ".wav"
        os.makedirs(os.path.join(output_dir, sub_dir1, sub_dir2), exist_ok=True)
        output_path = os.path.join(output_dir, sub_dir1, sub_dir2, modified_filename)
        random_index = random.randint(0, len(audio_list)-1)
        sound_path = os.path.join(Audioset_path, audio_list[random_index])
        start, end = add_sound_to_speech(speech_path, sound_path, output_path, 15)
        newline = output_path + "\t" + str(start) + "\t" + str(end) + "\n"
        f_write.write(newline)



