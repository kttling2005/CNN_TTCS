from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False)

history2 = model.fit(train_data, validation_data=val_data, epochs=10)

model.save("plant_disease_model_mobilenet.keras")
