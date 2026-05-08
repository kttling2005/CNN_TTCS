import tensorflow as tf
from tensorflow import keras
from tensorflow.python.layers.normalization import normalization
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2

#chuan hoa du lieu vi model MovbileNetV2 chay tren dai gia tri [-1,1]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
val_data = val_data.map(lambda x, y: (preprocess_input(x), y))

base_model = MobileNetV2(input_shape=(150, 150, 3), weights='imagenet', include_top=False)

# 2. Đóng băng base model để giữ lại kiến thức cũ
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), # Chuyển tensor 4D về 2D
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(38, activation='softmax') # n_classes là số lớp của bạn
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("plant_disease_model_mobileNetV2_v2.keras")

model.save("plant_disease_model_mobilenet.keras")
