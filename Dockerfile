FROM mirrors.tencent.com/voicecraft/voicecraft:v5 
RUN pip install \
    numpy==1.23.5 \
    librosa==0.8.1 \
    scipy \
    soundfile \
    matplotlib \
    pesq \
    auraloss \
    tqdm \
    nnAudio \
    ninja \ 
    huggingface_hub==0.23.5

WORKDIR /Encodec_DiT
RUN cd /Encodec_DiT && \
    pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft

RUN apt-get install -y --no-install-recommends rpm
RUN apt-get install -y --no-install-recommends bash
COPY cuda-compat-11-2-460.91.03-1.x86_64.rpm /Encodec_DiT/
RUN rpm -ivh --force --nodeps cuda-compat-11-2-460.91.03-1.x86_64.rpm