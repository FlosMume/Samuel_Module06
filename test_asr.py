# test_asr.py - Test Speech Recognition with Specific Microphone

import os
import wave
import speech_recognition as sr
from pydub import AudioSegment


def print_intro():
    """Print current working directory and available microphones."""
    print("Current Working Directory:", os.getcwd())
    print("\n🎤 Available Microphones:")
    print("------------------------")
    mic_list = sr.Microphone.list_microphone_names()
    for index, name in enumerate(mic_list):
        print(f"  [{index}] {name}")
    return mic_list


def select_microphone(mic_list, target_index):
    """Validate and return the selected microphone."""
    if target_index >= len(mic_list):
        print(f"❌ Error: Microphone index {target_index} is out of range!")
        exit(1)

    print(f"🎙️  Using Microphone [{target_index}]: {mic_list[target_index]}")
    return target_index


def record_audio(mic_index, output_file="recorded_audio.wav"):
    """Record audio from the specified microphone and save to file."""
    print("Say something...")
    recognizer = sr.Recognizer()

    recognizer.phrase_threshold = 1.0  # Wait up to 1 second of silence before ending phrase
    recognizer.non_speaking_duration = 1.0  # Treat <1s silence as part of same phrase

    with sr.Microphone(device_index=mic_index, sample_rate=16000, chunk_size=1024) as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("🎤 Say something (I'm listening for up to 10 seconds)...")
        try:
            audio_data = recognizer.listen(source, timeout=None, phrase_time_limit=None, phrase_threshold=1.0)
        except sr.WaitTimeoutError:
            print("❌ Listening timed out.")
            exit(1)

        print("✅ Recording finished. Saving audio...")
        with open(output_file, "wb") as f:
            f.write(audio_data.get_wav_data())

    print(f"✅ Audio saved as '{output_file}'")
    return output_file


def analyze_wav_file(file_path):
    """Print basic WAV file info."""
    print("\n🔍 Analyzing WAV file...")
    with wave.open(file_path, "r") as f:
        print("Channels:", f.getnchannels())
        print("Sample width (bytes):", f.getsampwidth())
        print("Frame rate (sample rate):", f.getframerate())
        print("Number of frames:", f.getnframes())


def clean_audio_for_google(input_file, output_file="clean_for_google.wav"):
    """Convert audio to mono, 16kHz for optimal Google ASR compatibility."""
    print(f"🧹 Cleaning audio: converting {input_file} to mono 16kHz...")
    audio = AudioSegment.from_wav(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_file, format="wav")
    print(f"✅ Cleaned audio saved as '{output_file}'")
    return output_file


def transcribe_with_google(audio_file):
    """Transcribe cleaned audio using Google Web Speech API."""
    print("🧠 Sending to Google Speech Recognition...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("✅ Google heard:", text)
            return text
        except sr.UnknownValueError:
            print("❌ Google couldn't understand the audio.")
        except sr.RequestError as e:
            print(f"❌ Could not request results from Google service; {e}")
    return None


def main():
    # === STEP 1: List microphones and select one ===
    mic_list = print_intro()

    # 👇 Set your desired microphone index here
    TARGET_MIC_INDEX = 3  # ← Change this to your preferred mic index

    # === STEP 2: Validate and select microphone ===
    selected_index = select_microphone(mic_list, TARGET_MIC_INDEX)

    # === STEP 3: Record audio ===
    raw_audio_file = record_audio(selected_index)

    # === STEP 4: Analyze raw recording ===
    analyze_wav_file(raw_audio_file)

    # === STEP 5: Clean audio for better ASR performance ===
    cleaned_audio_file = clean_audio_for_google(raw_audio_file)

    # === STEP 6: Transcribe using Google API ===
    transcribe_with_google(cleaned_audio_file)


if __name__ == "__main__":
    main()