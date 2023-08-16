import os
import sys
import scipy.io.wavfile

sys.path.append('./bark/')
from bark import SAMPLE_RATE, generate_audio, preload_models
from get_model import get_model


#Bark Processing
preload_models()

model_name = sys.argv[1]

voice_preset = "v2/en_speaker_6"

text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt, voice_preset)

scipy.io.wavfile.write("barkrvc_out.wav", SAMPLE_RATE, audio_array)


#RVC Processing
get_model()

os.chdir('./RVC')
os.system(f"python oneclickprocess.py --name {model_name}")
