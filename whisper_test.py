import openai



print('hello')

openai.api_key = ''

audio_file= open("/Users/benkinsella/Documents/Work/University of Toronto/2022-2023/Semester 2/ALC/Interview Files/trimmed_raspberry_test.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

print(transcript)


'''
# HOW TO CHOP UP CLIPS

from pydub import AudioSegment

song = AudioSegment.from_mp3("good_morning.mp3")

# PyDub handles time in milliseconds
ten_minutes = 10 * 60 * 1000

first_10_minutes = song[:ten_minutes]

first_10_minutes.export("good_morning_10.mp3", format="mp3")

'''

'''


#USES OLD WHISPER PACKAGE

import whisper


model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("/Users/benkinsella/Downloads/10_4231_FT0S-1715/bundle/PRESTO-R/List_A/Male/CARL LIVES in a LIVELY HOME.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

'''
