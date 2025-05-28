import os
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf

# 설정
input_directory = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\mp3_data"
temp_wav_dir = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\split_wavs"
feature_dir = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\features"

os.makedirs(temp_wav_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)

keywords = ['mosquito', 'flies']

# 지원하는 오디오 확장자
supported_formats = ['.mp3', '.wav']

# 파일 처리
for audio_file in os.listdir(input_directory):
    file_ext = os.path.splitext(audio_file)[1].lower()
    if file_ext not in supported_formats:
        continue  # mp3나 wav가 아니면 스킵

    audio_path = os.path.join(input_directory, audio_file)
    print(f"처리 중: {audio_path}")

    # 확장자에 따라 불러오기
    if file_ext == '.mp3':
        audio = AudioSegment.from_mp3(audio_path)
    elif file_ext == '.wav':
        audio = AudioSegment.from_wav(audio_path)

    duration_sec = int(audio.duration_seconds)
    print(f"총 길이: {duration_sec}초")

    # 키워드 기반 카테고리 결정
    category = next((k for k in keywords if k in audio_file.lower()), "others")

    # 디렉토리 설정
    category_wav_dir = os.path.join(temp_wav_dir, category)
    category_feature_dir = os.path.join(feature_dir, category)

    os.makedirs(category_wav_dir, exist_ok=True)
    os.makedirs(category_feature_dir, exist_ok=True)

    # 1초 단위로 잘라서 저장
    base_name = os.path.splitext(audio_file)[0]
    for i in range(duration_sec):
        chunk = audio[i*1000:(i+1)*1000]
        chunk_path = os.path.join(category_wav_dir, f"{base_name}_chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

    print(f"1초 단위로 자르기 완료: {audio_file}")

    # Mel spectrogram 변환
    sr_desired = 22050
    n_mels = 128

    for fname in os.listdir(category_wav_dir):
        if fname.endswith(".wav") and fname.startswith(base_name):
            fpath = os.path.join(category_wav_dir, fname)
            y, sr = librosa.load(fpath, sr=sr_desired)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = mel_db[np.newaxis, ...]  # 채널 추가 (1, 128, time)

            save_path = os.path.join(category_feature_dir, fname.replace(".wav", ".npy"))
            np.save(save_path, mel_db)

    print(f"mel spectrogram 추출 및 저장 완료: {audio_file}")

print("모든 오디오 파일 처리 완료.")
