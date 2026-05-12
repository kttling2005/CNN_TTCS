#import thu vien
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.layers.normalization import normalization

#load du lieu
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/training",
    image_size=(256,256),
    batch_size=32,
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/validation",
    image_size=(256,256),
    batch_size=32,
)

#chuan hoa du lieu
normalization_layer = layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

#xay dung mo hinh CNN
model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(38,activation='softmax') # so class
])

#compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

#luu model
model.save("plant_disease_model.h5")

