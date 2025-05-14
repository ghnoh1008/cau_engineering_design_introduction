import os
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf

# 설정
input_directory = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\mp3_data"  # mp3 파일들이 있는 디렉토리
temp_wav_dir = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\split_wavs"  # 자른 .wav 파일 저장 폴더
feature_dir = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\features"  # CNN 입력용 numpy 파일 저장 폴더

# 폴더가 없다면 생성
os.makedirs(temp_wav_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)

# 키워드 목록 (파일명에서 확인할 키워드)
keywords = ['mosquito', 'flies']

# 1. 디렉토리 내의 모든 mp3 파일을 처리
for mp3_file in os.listdir(input_directory):
    if mp3_file.endswith(".mp3"):
        mp3_path = os.path.join(input_directory, mp3_file)
        print(f"처리 중: {mp3_path}")

        # mp3 불러오기
        audio = AudioSegment.from_mp3(mp3_path)
        duration_sec = int(audio.duration_seconds)
        print(f"총 길이: {duration_sec}초")

        # 키워드를 기반으로 카테고리 찾기
        category = None
        for keyword in keywords:
            if keyword in mp3_file.lower():
                category = keyword
                break  # 첫 번째 매칭된 키워드로 카테고리 결정

        if not category:
            category = "others"  # 키워드에 매칭되지 않으면 "others"로 처리

        # 키워드별로 카테고리 폴더 생성
        category_wav_dir = os.path.join(temp_wav_dir, category)
        category_feature_dir = os.path.join(feature_dir, category)

        # 해당 카테고리 폴더가 없으면 생성
        os.makedirs(category_wav_dir, exist_ok=True)
        os.makedirs(category_feature_dir, exist_ok=True)

        # 2. 1초 단위로 자르기 + wav 저장
        for i in range(duration_sec):
            chunk = audio[i*1000:(i+1)*1000]
            chunk_path = os.path.join(category_wav_dir, f"{mp3_file[:-4]}_chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")

        print(f"1초 단위로 자르기 완료: {mp3_file}")

        # 3. 각 wav 파일 -> mel spectrogram 으로 변환
        sr_desired = 22050  # 표준 샘플레이트
        n_mels = 128        # mel feature 수

        for fname in os.listdir(category_wav_dir):
            if fname.endswith(".wav") and fname.startswith(mp3_file[:-4]):
                fpath = os.path.join(category_wav_dir, fname)
                y, sr = librosa.load(fpath, sr=sr_desired)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # shape: (128, time) -> CNN에 맞게 (1, 128, time) 또는 (128, time, 1)
                mel_db = mel_db[np.newaxis, ...]  # 채널 추가

                save_path = os.path.join(category_feature_dir, fname.replace(".wav", ".npy"))
                np.save(save_path, mel_db)

        print(f"mel spectrogram 추출 및 저장 완료: {mp3_file}")

print("모든 mp3 파일 처리 완료.")