import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, f1_score

# feature 디렉토리
feature_dir = r"C:\Users\kikio\Desktop\전자전기공학\2학년\공설입\project\features"

# 1. 데이터 불러오기
X = []
y = []

# 하위 디렉토리 목록 (모기, 파리, 기타)
subdirs = ['mosquito', 'flies', 'others']

for label, subdir in enumerate(subdirs):
    subdir_path = os.path.join(feature_dir, subdir)
    for fname in os.listdir(subdir_path):
        if fname.endswith(".npy"):
            fpath = os.path.join(subdir_path, fname)
            feature = np.load(fpath)  # shape: (1, 128, time)
            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

# (N, 1, 128, time) → (N, 128, time, 1)
X = np.squeeze(X, axis=1)
X = X[..., np.newaxis]

# 레이블을 one-hot 인코딩
y_cat = to_categorical(y, num_classes=3)

print("입력 데이터 shape:", X.shape)  # 예: (N, 128, time, 1)
print("레이블 shape (one-hot):", y_cat.shape)

# 2. train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 3. CNN + BiLSTM 모델 정의
model = models.Sequential([
    # [1] Conv2D로 주파수-시간 패턴 추출
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # [2] CNN 출력 → 시퀀스로 변환
    layers.Reshape((X_train.shape[1] // 4, -1)),  # 2번 maxpool → 128 → 32

    # [3] BiLSTM으로 시간 흐름 학습
    layers.Bidirectional(layers.LSTM(128, return_sequences=False)),

    # [4] FC
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4-1. 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# 4-2. 모델 저장
model.save('your_model.h5')

# 5. 평가
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print("F1-score (macro):", f1_macro)
print("F1-score (weighted):", f1_weighted)
print("\n[분류 리포트]")
print(classification_report(y_true, y_pred, target_names=subdirs))
