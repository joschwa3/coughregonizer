import keras
import youtube_dl
from pydub import AudioSegment
import os
import sys

sound_class = "/m/01b_21"
neg_path = r'/Users/kylegood/Desktop/MastersProgram/data/cough/neg/'
pos_path = r'/Users/kylegood/Desktop/MastersProgram/data/cough/pos/'
cwd = os.getcwd()
paths = [neg_path, pos_path]


def fifteen_sec_chunks(wav_file):
    count=1
    for i in range(1,1000,15):
        t1 = i * 1000  # This is in millieconds..
        t2 = (i+15) * 1000
        newAudio = AudioSegment.from_wav(wav_file)
        newAudio = newAudio[t1:t2]
        newAudio.export('chunks/'+str(count)+'.wav', format="wav")  # Exports to a wav file in the current path.
        print(count)
        count += 1

mp3s = []
for directories in paths:
    for files in os.listdir(directories):
        mp3s.append(files)

for clips in mp3s:
    if 'neg' in clips:
        sound = AudioSegment.from_mp3(neg_path + clips)
        split_name = clips.split('.')[0]
        sound.export(split_name + ".wav", format="wav")
    else:
        sound = AudioSegment.from_mp3(pos_path + clips)
        split_name = clips.split('.')[0]
        sound.export(split_name + ".wav", format="wav")

for files in os.listdir(cwd):
    if '.wav' in files:
        fifteen_sec_chunks(files)


