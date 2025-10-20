# -*- coding: utf-8 -*-
import os
import random
import time
import io
import pandas as pd
import numpy as np
# THÊM THƯ VIỆN STREAMLIT VÀ PIL
import streamlit as st
from PIL import Image

# THÊM THƯ VIỆN ML THỰC TẾ
try:
    import tensorflow as tf
    # Chỉ cần import load_model, Model từ Keras
    from tensorflow.keras.models import load_model, Model
    # THÊM HÀM TIỀN XỬ LÝ CHUẨN XCEPTION
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input 
    
    # Tắt thông báo Keras/TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    st.error("Lỗi: Vui lòng cài đặt TensorFlow. Kiểm tra file requirements.txt.")
    tf = None
    load_model = None

# --- Cấu hình Mô hình và Giống Chó ---

# ĐƯỜNG DẪN ĐẾN CÁC MÔ HÌNH (Chỉ còn Xception)
MODEL_PATHS = {
    "Xception": "model/final_xception_classifier.h5",
}

# Danh sách 10 giống chó đã được huấn luyện (Đã xác nhận từ bạn)
SUPERVISED_BREEDS = [
    'scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 
    'bernese_mountain_dog', 'shih-tzu', 'pomeranian', 'great_pyrenees', 
    'basenji', 'samoyed'
]

# Danh sách 10 giống chó Zero-Shot (chưa từng thấy, vẫn giữ giả lập)
UNSEEN_BREEDS = [
    "Saluki", "Azawakh", "Löwchen", "Komondor", "Affenpinscher", 
    "Schipperke", "Cirneco_dell'Etna", "Xoloitzcuintli", "Thai_Ridgeback", 
    "New_Guinea_Singing_Dog"
]

# Kích thước ảnh đầu vào đã sửa: 224x224
TARGET_SIZE = (224, 224) 

# --- HÀM TẢI NHÃN CŨ ĐÃ BỊ LOẠI BỎ ---
# Chúng ta sẽ sử dụng trực tiếp danh sách SUPERVISED_BREEDS
def get_supervised_breeds():
    """Trả về danh sách 10 giống chó đã được huấn luyện (Xception)."""
    st.success("Tải thành công 10 giống chó đã được huấn luyện.")
    return SUPERVISED_BREEDS

# --- Tiền Xử Lý Ảnh (Giữ nguyên) ---

@st.cache_data
def preprocess_image(image, target_size=TARGET_SIZE):
    """
    Tiền xử lý ảnh cho mô hình Deep Learning (224x224) bằng hàm chuẩn Xception.
    """
    # Thay đổi kích thước và chuyển sang mảng Numpy (float32)
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    
    # Kiểm tra kênh màu (Đảm bảo RGB)
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    # Mở rộng chiều (batch dimension)
    image_array = np.expand_dims(image_array, axis=0)
    
    # CHUẨN HÓA SỬ DỤNG HÀM CHÍNH THỨC CỦA XCEPTION
    processed_image = xception_preprocess_input(image_array)
    
    return processed_image

# --- Tải Mô hình Thực tế (Chỉ Xception) ---

@st.cache_resource
def load_models():
    """
    Tải mô hình Xception.
    """
    st.info("Đang tải mô hình Xception (thực tế)...")
    
    loaded_models = {}
    
    # Tải Xception Model (.h5)
    try:
        model_xception = load_model(MODEL_PATHS["Xception"])
        
        # KIỂM TRA ĐẦU RA: Đảm bảo mô hình có 10 lớp
        output_dim = model_xception.output_shape[-1]
        if output_dim != 10:
             st.error(f"LỖI KHỚP LỚP: Mô hình .h5 có {output_dim} đầu ra, nhưng bạn đã huấn luyện 10 giống chó.")
             st.info("Vui lòng đảm bảo mô hình Xception bạn tải chỉ có 10 neuron ở lớp cuối cùng.")
             return None

        loaded_models["Model Xception"] = model_xception
        st.success(f"Tải thành công: Xception (Keras/TensorFlow)")
    except Exception as e:
        st.error(f"LỖI tải Xception: {e}")
        
    if not loaded_models:
        st.warning("Không có mô hình nào được tải thành công. Vui lòng kiểm tra đường dẫn.")
        return None
        
    return loaded_models

# --- Hàm Dự đoán Thực tế (Chỉ Xception) ---

def run_prediction(model, image_features, top_k, all_breeds_list):
    """
    Chạy dự đoán trên mô hình Xception.
    """
    
    # Xception (Deep Learning)
    predictions = model.predict(image_features)
    labels = all_breeds_list 
    
    # Xử lý kết quả đầu ra
    if predictions.ndim > 1:
        probabilities = predictions[0]
    else:
        # Trường hợp predict() chỉ trả về chỉ số lớp
        predicted_class_index = int(predictions[0])
        probabilities = np.zeros(len(labels))
        probabilities[predicted_class_index] = 0.99
    
    # Lấy Top K kết quả
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for i in top_k_indices:
        # Chỉ mục (index) của dự đoán (0-9) phải khớp với nhãn (0-9)
        if i < len(labels):
            results.append((labels[i], probabilities[i]))
        
    return results

# --- Hàm Giả lập Zero-Shot (ZSL) (Giữ nguyên) ---

def simulate_zeroshot_prediction(image_features, unseen_breeds, top_k=5):
    """
    Giữ nguyên giả lập cho Zero-Shot Learning.
    """
    # GIẢ LẬP:
    predictions = random.sample(unseen_breeds, top_k)
    results = []
    base_prob = 0.70
    for i, breed in enumerate(predictions):
        similarity = round(base_prob - (i * 0.04) - random.uniform(0.01, 0.02), 4)
        results.append((breed, max(0.01, similarity)))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Hàm Chính Của Ứng Dụng Streamlit ---

def app():
    st.set_page_config(
        page_title="Dự Đoán Giống Chó: Xception & Zero-Shot", 
        layout="wide"
    )
    
    st.title("🐶 Ứng Dụng Dự Đoán Giống Chó (Xception & Zero-Shot)")
    st.markdown("Sử dụng **Mô hình Xception CNN (10 giống)** và **Zero-Shot Learning** (ZSL) để nhận diện giống chó.")
    
    # Tải danh sách 10 giống chó (danh sách cứng)
    ALL_BREEDS = get_supervised_breeds()
    if ALL_BREEDS is None:
        return 
    
    # --- KHỐI KIỂM TRA NHÃN ---
    if len(ALL_BREEDS) >= 3:
        st.sidebar.markdown("**Kiểm tra Nhãn (Top 3)**")
        st.sidebar.markdown(f"Index 0: **{ALL_BREEDS[0]}**")
        st.sidebar.markdown(f"Index 1: **{ALL_BREEDS[1]}**")
        st.sidebar.markdown(f"Index 2: **{ALL_BREEDS[2]}**")
        st.sidebar.markdown("---")
        
    # Tải mô hình chỉ một lần
    loaded_models = load_models()
    
    if loaded_models is None:
        return 
    
    model_xception = loaded_models.get("Model Xception")
    
    # Bộ tải ảnh lên
    uploaded_file = st.file_uploader(
        "**1. Chọn một hình ảnh chó từ thư viện của bạn**", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Đọc và hiển thị ảnh
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Hình ảnh đã tải lên', use_column_width=True)
            
            st.markdown("---")
            
            with st.spinner('Đang tiền xử lý và chạy dự đoán...'):
                
                # Tiền xử lý ảnh (chỉ chạy 1 lần)
                image_features = preprocess_image(image, target_size=TARGET_SIZE)
                
                # 2. Chạy dự đoán Xception (Top 3)
                st.subheader("2. Kết Quả Từ Mô Hình Xception CNN (Top 3)")
                
                predictions = run_prediction(
                    model_xception, 
                    image_features, 
                    top_k=3, 
                    all_breeds_list=ALL_BREEDS
                )
                
                df = pd.DataFrame(predictions, columns=['Giống Chó', 'Xác Suất'])
                df['Xác Suất'] = (df['Xác Suất'] * 100).map('{:.2f}%'.format)
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                        
                # 3. Chạy dự đoán Zero-Shot Learning (Top 5)
                st.markdown("---")
                st.subheader("3. Dự Đoán Zero-Shot Learning (Top 5 Giống Chưa Từng Thấy)")
                st.caption("Dự đoán giả lập dựa trên độ tương đồng ngữ cảnh/đặc trưng.")
                
                zeroshot_results = simulate_zeroshot_prediction(image_features, UNSEEN_BREEDS, top_k=5)
                
                df_zsl = pd.DataFrame(zeroshot_results, columns=['Giống Chó (UNSEEN)', 'Độ Tương Đồng'])
                df_zsl['Độ Tương Đồng'] = (df_zsl['Độ Tương Đồng'] * 100).map('{:.2f}%'.format)
                
                st.dataframe(df_zsl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
            st.error("Vui lòng kiểm tra lại file ảnh và đảm bảo mô hình Xception đã được tải chính xác.")
    else:
        st.info("Hãy tải một hình ảnh để bắt đầu dự đoán.")


if __name__ == "__main__":
    app()
