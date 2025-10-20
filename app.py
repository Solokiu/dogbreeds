# -*- coding: utf-8 -*-
import os
import random
import time
import io
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# THÊM MỚI: Thư viện để tải mô hình XGBoost và Random Forest
import joblib 

# THÊM THƯ VIỆN ML THỰC TẾ
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    st.error("Lỗi: Vui lòng cài đặt TensorFlow. Kiểm tra file requirements.txt.")
    tf = None
    load_model = None

# --- Cấu hình Mô hình và Giống Chó ---

# CHỈNH SỬA: Đường dẫn đến TẤT CẢ các mô hình
MODEL_PATHS = {
    "Xception": "model/final_xception_classifier.h5",
    "FeatureExtractor": "model/feature_extractor_only.h5",
    "XGBoost": "model/xgboost_model.joblib", # Giả định tên file là .joblib
    "RandomForest": "model/rf_model.joblib"  # Giả định tên file là .joblib
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

# --- HÀM TẢI NHÃN (Giữ nguyên) ---
def get_supervised_breeds():
    """Trả về danh sách 10 giống chó đã được huấn luyện."""
    st.success("Tải thành công 10 giống chó đã được huấn luyện.")
    return SUPERVISED_BREEDS

# --- Tiền Xử Lý Ảnh (Giữ nguyên) ---

@st.cache_data
def preprocess_image(image, target_size=TARGET_SIZE):
    """
    Tiền xử lý ảnh cho mô hình Deep Learning (224x224) bằng hàm chuẩn Xception.
    """
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    image_array = np.expand_dims(image_array, axis=0)
    processed_image = xception_preprocess_input(image_array)
    
    return processed_image

# --- Tải Mô hình (Đã cập nhật) ---

@st.cache_resource
def load_models():
    """
    Tải tất cả các mô hình: Xception (CNN), Feature Extractor (CNN), 
    XGBoost, và Random Forest.
    """
    st.info("Đang tải các mô hình...")
    
    loaded_models = {}
    
    # 1. Tải Xception Model (.h5)
    try:
        model_xception = load_model(MODEL_PATHS["Xception"])
        output_dim = model_xception.output_shape[-1]
        if output_dim != 10:
            st.error(f"LỖI KHỚP LỚP: Mô hình Xception .h5 có {output_dim} đầu ra, nhưng bạn đã huấn luyện 10 giống chó.")
        else:
            loaded_models["Model Xception"] = model_xception
            st.success(f"Tải thành công: Xception (Keras/TensorFlow)")
    except Exception as e:
        st.error(f"LỖI tải Xception: {e}")
        
    # 2. THÊM MỚI: Tải Feature Extractor (.h5)
    try:
        model_extractor = load_model(MODEL_PATHS["FeatureExtractor"])
        loaded_models["FeatureExtractor"] = model_extractor
        st.success(f"Tải thành công: Trình trích xuất đặc trưng (Keras/TensorFlow)")
    except Exception as e:
        st.error(f"LỖI tải FeatureExtractor: {e}. Đảm bảo file '{MODEL_PATHS['FeatureExtractor']}' tồn tại.")

    # 3. THÊM MỚI: Tải XGBoost Model (.joblib)
    try:
        model_xgb = joblib.load(MODEL_PATHS["XGBoost"])
        loaded_models["XGBoost"] = model_xgb
        st.success(f"Tải thành công: XGBoost (Joblib)")
    except Exception as e:
        st.error(f"LỖI tải XGBoost: {e}. Bạn đã tạo file '{MODEL_PATHS['XGBoost']}' chưa?")

    # 4. THÊM MỚI: Tải Random Forest Model (.joblib)
    try:
        model_rf = joblib.load(MODEL_PATHS["RandomForest"])
        loaded_models["RandomForest"] = model_rf
        st.success(f"Tải thành công: Random Forest (Joblib)")
    except Exception as e:
        st.error(f"LỖI tải Random Forest: {e}. Bạn đã tạo file '{MODEL_PATHS['RandomForest']}' chưa?")
        
    if not loaded_models:
        st.warning("Không có mô hình nào được tải thành công.")
        return None
        
    return loaded_models

# --- Hàm Dự đoán (CNN) ---

# CHỈNH SỬA: Đổi tên biến (image_features -> preprocessed_image) cho rõ ràng
def run_prediction(model, preprocessed_image, top_k, all_breeds_list):
    """
    Chạy dự đoán trên mô hình Xception (CNN).
    """
    # Xception (Deep Learning)
    predictions = model.predict(preprocessed_image)
    labels = all_breeds_list
    
    if predictions.ndim > 1:
        probabilities = predictions[0]
    else:
        predicted_class_index = int(predictions[0])
        probabilities = np.zeros(len(labels))
        probabilities[predicted_class_index] = 0.99
    
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for i in top_k_indices:
        if i < len(labels):
            results.append((labels[i], probabilities[i]))
            
    return results

# --- THÊM MỚI: Hàm Dự đoán (XGBoost/Random Forest) ---

def run_tree_prediction(model, feature_vector, top_k, all_breeds_list):
    """
    Chạy dự đoán trên các mô hình tree-based (XGB, RF) sử dụng feature vector.
    """
    # Tree models (XGB, RF) cần .predict_proba()
    # feature_vector đã có dạng (1, N_features) từ extractor
    probabilities = model.predict_proba(feature_vector)
    
    labels = all_breeds_list
    
    # Lấy hàng đầu tiên (batch size là 1)
    if probabilities.ndim > 1:
        probabilities = probabilities[0]
    
    # Lấy Top K kết quả
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for i in top_k_indices:
        if i < len(labels):
            results.append((labels[i], probabilities[i]))
            
    return results

# --- Hàm Giả lập Zero-Shot (ZSL) ---

# CHỈNH SỬA: Nhận feature_vector (mặc dù hàm giả lập không dùng)
def simulate_zeroshot_prediction(feature_vector, unseen_breeds, top_k=5):
    """
    Giữ nguyên giả lập cho Zero-Shot Learning.
    (Trong tương lai, bạn có thể dùng feature_vector thật ở đây)
    """
    predictions = random.sample(unseen_breeds, top_k)
    results = []
    base_prob = 0.70
    for i, breed in enumerate(predictions):
        similarity = round(base_prob - (i * 0.04) - random.uniform(0.01, 0.02), 4)
        results.append((breed, max(0.01, similarity)))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Hàm Chính Của Ứng Dụng Streamlit (Đã cập nhật) ---

def app():
    st.set_page_config(
        # CHỈNH SỬA: Tiêu đề
        page_title="Dự Đoán Giống Chó (CNN, XGB, RF)", 
        layout="wide"
    )
    
    # CHỈNH SỬA: Tiêu đề
    st.title("🐶 Ứng Dụng Dự Đoán Giống Chó (Xception, XGBoost, Random Forest)")
    st.markdown("Sử dụng 3 mô hình (CNN, XGB, RF) cho 10 giống chó và Zero-Shot Learning (ZSL).")
    
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
        st.error("Một số mô hình quan trọng không thể tải. Vui lòng kiểm tra log.")
        return 
    
    # CHỈNH SỬA: Lấy tất cả các mô hình
    model_xception = loaded_models.get("Model Xception")
    model_extractor = loaded_models.get("FeatureExtractor") # Mới
    model_xgb = loaded_models.get("XGBoost")               # Mới
    model_rf = loaded_models.get("RandomForest")           # Mới
    
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
            
            with st.spinner('Đang tiền xử lý, trích xuất đặc trưng và chạy dự đoán...'):
                
                # Tiền xử lý ảnh (chỉ chạy 1 lần)
                # CHỈNH SỬA: Đổi tên biến
                preprocessed_image = preprocess_image(image, target_size=TARGET_SIZE)
                
                # THÊM MỚI: Trích xuất đặc trưng (chỉ chạy 1 lần)
                feature_vector = None
                if model_extractor:
                    feature_vector = model_extractor.predict(preprocessed_image)
                    st.success("Trích xuất đặc trưng thành công.")
                else:
                    st.warning("Thiếu mô hình Trích xuất Đặc trưng, không thể chạy XGBoost/RF.")
                
                # 2. Chạy dự đoán Xception (Top 3)
                st.subheader("2. Kết Quả Từ Mô Hình Xception CNN (Top 3)")
                if model_xception:
                    predictions = run_prediction(
                        model_xception, 
                        preprocessed_image, # Đã đổi tên
                        top_k=3, 
                        all_breeds_list=ALL_BREEDS
                    )
                    
                    df = pd.DataFrame(predictions, columns=['Giống Chó', 'Xác Suất'])
                    df['Xác Suất'] = (df['Xác Suất'] * 100).map('{:.2f}%'.format)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Mô hình Xception chưa được tải.")
                
                
                # 3. THÊM MỚI: Chạy dự đoán XGBoost (Top 3)
                st.markdown("---")
                st.subheader("3. Kết Quả Từ Mô Hình XGBoost (Top 3)")
                
                if model_xgb and feature_vector is not None:
                    xgb_predictions = run_tree_prediction(
                        model_xgb,
                        feature_vector,
                        top_k=3,
                        all_breeds_list=ALL_BREEDS
                    )
                    df_xgb = pd.DataFrame(xgb_predictions, columns=['Giống Chó', 'Xác Suất'])
                    df_xgb['Xác Suất'] = (df_xgb['Xác Suất'] * 100).map('{:.2f}%'.format)
                    st.dataframe(df_xgb, use_container_width=True, hide_index=True)
                elif feature_vector is None:
                    st.info("Cần có Trình trích xuất đặc trưng để chạy XGBoost.")
                else:
                    st.warning("Mô hình XGBoost chưa được tải.")


                # 4. THÊM MỚI: Chạy dự đoán Random Forest (Top 3)
                st.markdown("---")
                st.subheader("4. Kết Quả Từ Mô Hình Random Forest (Top 3)")
                
                if model_rf and feature_vector is not None:
                    rf_predictions = run_tree_prediction(
                        model_rf,
                        feature_vector,
                        top_k=3,
                        all_breeds_list=ALL_BREEDS
                    )
                    df_rf = pd.DataFrame(rf_predictions, columns=['Giống Chó', 'Xác Suất'])
                    df_rf['Xác Suất'] = (df_rf['Xác Suất'] * 100).map('{:.2f}%'.format)
                    st.dataframe(df_rf, use_container_width=True, hide_index=True)
                elif feature_vector is None:
                    st.info("Cần có Trình trích xuất đặc trưng để chạy Random Forest.")
                else:
                    st.warning("Mô hình Random Forest chưa được tải.")

                                    
                # 5. Chạy dự đoán Zero-Shot Learning (Top 5)
                # CHỈNH SỬA: Đổi số thứ tự
                st.markdown("---")
                st.subheader("5. Dự Đoán Zero-Shot Learning (Top 5 Giống Chưa Từng Thấy)")
                st.caption("Dự đoán giả lập dựa trên độ tương đồng ngữ cảnh/đặc trưng.")
                
                # CHỈNH SỬA: Truyền feature_vector
                zeroshot_results = simulate_zeroshot_prediction(feature_vector, UNSEEN_BREEDS, top_k=5)
                
                df_zsl = pd.DataFrame(zeroshot_results, columns=['Giống Chó (UNSEEN)', 'Độ Tương Đồng'])
                df_zsl['Độ Tương Đồng'] = (df_zsl['Độ Tương Đồng'] * 100).map('{:.2f}%'.format)
                
                st.dataframe(df_zsl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
            st.error("Vui lòng kiểm tra lại file ảnh và đảm bảo các mô hình đã được tải chính xác.")
    else:
        st.info("Hãy tải một hình ảnh để bắt đầu dự đoán.")


if __name__ == "__main__":
    app()
