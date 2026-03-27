import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. Đường dẫn (Sửa cho đúng tên file của bạn)
model_path = 'plant_disease_model.h5'
img_path = 'dataset/test/Potato___Early_blight/3b1e3161-b02e-4aa0-a490-8a50c242d10e___RS_Early.B 8553.JPG'

# 2. Load model
model = tf.keras.models.load_model(model_path)

# 3. Xử lý ảnh (Quan trọng: Kích thước phải giống lúc train, ví dụ 150x150)
img_size = (256, 256)
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Chia 255 nếu lúc train có dùng rescale

# 4. Dự đoán
result = model.predict(img_array)

#mapping result -> tên class
class_name = [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_apple_rust",
    "Apple_healthy",
    "Blueberry_healthy",
    "Cherry_healthy",
    "Cherry_powdery_mildew",
    "Corn_leaf_spot",
    "Corn_rust",
    "Corn_healthy",
    "Corn_leaf_blight",
    "Grape_black_rot",
    "Grape_black_measles",
    "Grape_healthy",
    "Grape_leaf_blight",
    "Orange_citrus_greening",
    "peach_bacterial_spot",
    "peach_healthy",
    "pepper_bacterial_spot",
    "pepper_healthy",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_powdery_mildew",
    "Strawberry_healthy",
    "Strawberry_leaf_scorch",
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_healthy",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites",
    "Tomato_target_spot",
    "Tomato_moisaic_virus",
    "Tomato_yellow_leaf_curl_virus"
]
predict_index = np.argmax(result)
predicted_class = class_name[predict_index]
# 5. In kết quả
print("Mảng dự đoán (Raw output):", result)
print("Lớp (Class) có tỉ lệ cao nhất:", predicted_class)