import os
import random

# split valid_wav_text into train split, dev split, test split

input_file = "/apdcephfs/private_sedricksong/data/zh_magic_data/valid_wav_text_filter_by_duration"
train_file = "/apdcephfs/private_sedricksong/data/zh_magic_data/train_file"
dev_file = "/apdcephfs/private_sedricksong/data/zh_magic_data/dev_file"
test_file = "/apdcephfs/private_sedricksong/data/zh_magic_data/test_file"

with open (input_file, "r") as f_read, open(train_file, "w") as f_train, open(dev_file, "w") as f_dev, open(test_file, "w") as f_test:
    for line in f_read.readlines():
        number = random.random()
        if number <= 0.002:
            f_dev.write(line)
        elif number >= (1-0.002):
            f_test.write(line)
        else:
            f_train.write(line)
