import os
import joblib
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

app = Flask(__name__)
CORS(app)

MODEL_DIR = 'models'

model_1 = None
model_2 = None
model_3 = None

try:
    model_1 = joblib.load(os.path.join(MODEL_DIR, 'rf_model_top10.joblib'))
    model_2 = joblib.load(os.path.join(MODEL_DIR, 'xgb_model_top10.joblib'))
    model_3 = load_model(os.path.join(MODEL_DIR, 'model_3.h5'))
    print("All 3 models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

def preprocess_image(file, target_size=(224, 224)):
    img = keras_image.load_img(file, target_size=target_size)
    img_arr = keras_image.img_to_array(img)
    x = np.expand_dims(img_arr, axis=0)
    x_processed = xception_preprocess(x)
    return x_processed, img_arr.flatten().reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = {}

    if request.method == 'POST':
        try:
            if 'image' not in request.files or request.files['image'].filename == '':
                prediction_results['Error'] = "Vui lòng chọn ảnh!"
            else:
                img_file = request.files['image']
                x_processed, img_features_flat = preprocess_image(img_file)

                # Model 1: RF
                pred_1 = model_1.predict(img_features_flat)[0]
                prediction_results['Model RF'] = f"Dự đoán: {pred_1}"

                # Model 2: XGBoost
                pred_2 = model_2.predict(img_features_flat)[0]
                prediction_results['Model XGBOOST'] = f"Dự đoán: {pred_2}"

                # Model 3: CNN
                pred_3_proba = model_3.predict(x_processed)[0]
                pred_3_idx = np.argmax(pred_3_proba)
                prediction_results['Model CNN'] = f"Class index: {pred_3_idx}, Xác suất: {pred_3_proba[pred_3_idx]:.4f}"

        except Exception as e:
            prediction_results['Error'] = f"Lỗi dự đoán: {e}. Vui lòng kiểm tra đầu vào."

    return render_template('index.html', results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)