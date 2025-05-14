import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

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

            # 레이블을 추출 (0: others, 1: mosquito, 2: flies)
            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

# (N, 1, 128, time) → (N, 128, time, 1)
X = np.squeeze(X, axis=1)
X = X[..., np.newaxis]

# 레이블을 one-hot 인코딩
y_cat = to_categorical(y, num_classes=3)

print("입력 데이터 shape:", X.shape)
print("레이블 shape (one-hot):", y_cat.shape)

# 2. train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 3. 다중 클래스 CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 다중 클래스 (3개)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# 5. 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {acc:.4f}")