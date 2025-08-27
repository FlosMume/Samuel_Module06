# test_asr.py - Test Speech Recognition with Specific Microphone

import os

# Get the current working directory
cwd = os.getcwd()
print("Current Working Directory:", cwd)

import speech_recognition as sr

# === STEP 1: List all available microphones ===
print("🎤 Available Microphones:")
print("------------------------")
mic_list = sr.Microphone.list_microphone_names()
for index, name in enumerate(mic_list):
    print(f"  [{index}] {name}")

print("\n")

# === STEP 2: Choose a microphone (set index manually) ===
# 👇 Change this number to the index of your desired mic
TARGET_MIC_INDEX = 11  # ← Change this to your preferred mic index

# Optional: Validate index
if TARGET_MIC_INDEX >= len(mic_list):
    print(f"❌ Error: Microphone index {TARGET_MIC_INDEX} is out of range!")
    exit(1)

print(f"🎙️  Using Microphone [{TARGET_MIC_INDEX}]: {mic_list[TARGET_MIC_INDEX]}")
print("Say something...")

# === STEP 3: Use the selected mic ===
r = sr.Recognizer()
with sr.Microphone(device_index=TARGET_MIC_INDEX, sample_rate = 16000, chunk_size=1024) as source:
    print("Say something...")    
    r.adjust_for_ambient_noise(source)  # Reduce background noise
    # Audio captured in memory
    audio = r.listen(source, timeout=5, phrase_time_limit=300) # Listen with timeout and phrase limit

# === 🔽 SAVE THE AUDIO TO A FILE 🔽 ===
with open("recorded_audio.wav", "wb") as f:
    f.write(audio.get_wav_data())

print("✅ Audio saved as 'recorded_audio.wav'")


with open("Conference.wav", "rb") as f:
    audio_data = sr.AudioFile(f)
    with audio_data as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    print(text)
print("✅ Audio 'Conference.wav'") 

import wave
with wave.open("recorded_audio.wav", "r") as f:
    print("Channels:", f.getnchannels())
    print("Sample width (bytes):", f.getsampwidth())
    print("Frame rate (sample rate):", f.getframerate())
    print("Number of frames:", f.getnframes())

from pydub import AudioSegment

# Load the recorded audio
audio_file = AudioSegment.from_wav("recorded_audio.wav")

# Convert to mono and set frame rate to 16kHz
audio_file = audio_file.set_frame_rate(16000).set_channels(1)

# Export as raw audio data for Google
wav_data = audio_file.raw_data
sample_rate = audio_file.frame_rate
sample_width = audio_file.sample_width  # Usually 2 for 16-bit

# Create new AudioData object
audio = sr.AudioData(wav_data, sample_rate, sample_width)


# === STEP 4: Recognize speech ===
try:
    text = r.recognize_google(audio)
    print("✅ You said:", text)
except sr.UnknownValueError:
    print("❌ Could not understand the audio")
except sr.RequestError as e:
    print(f"❌ Could not request results from Google Speech Recognition service; {e}")