import sys
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

def load_all_models(model_folder):
    cnn_model = load_model(os.path.join(model_folder, "final_xception_classifier.keras"))
    xgb_model = joblib.load(os.path.join(model_folder, "xgboost.joblib"))
    rf_model = joblib.load(os.path.join(model_folder, "randomforest.joblib"))
    attribute_predictor = load_model(os.path.join(model_folder, "attribute_predictor_zsl.keras"))
    base_model_layer = cnn_model.layers[1]
    feature_extractor = Model(inputs=base_model_layer.input, outputs=base_model_layer.output)
    return cnn_model, xgb_model, rf_model, attribute_predictor, feature_extractor

def predict_single(image_path, model_cnn, feature_extractor, xgb_model, rf_model, attribute_predictor, SIZE):
    img = keras_image.load_img(image_path, target_size=SIZE)
    img_arr = keras_image.img_to_array(img)
    x = np.expand_dims(img_arr, axis=0)
    x_processed = xception_preprocess(x)
    img_features = feature_extractor.predict(x_processed, verbose=0)
    img_features_flat = img_features.reshape(1, -1)
    probs_cnn = model_cnn.predict(x_processed, verbose=0)
    probs_rf = rf_model.predict_proba(img_features_flat)
    probs_xgb = xgb_model.predict_proba(img_features_flat)
    zsl_pred = attribute_predictor.predict(img_features_flat, verbose=0)
    return {
        "cnn_probs": probs_cnn,
        "rf_probs": probs_rf,
        "xgb_probs": probs_xgb,
        "zsl_pred": zsl_pred
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_folder = os.path.join(os.path.dirname(__file__), "model")
    SIZE = (224, 224)

    # Load breed labels
    labels_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "labels.csv"))
    # Assuming labels.csv has columns: 'breed', 'class_id'
    breed_names = labels_df.sort_values("id")["breed"].tolist()

    cnn_model, xgb_model, rf_model, attribute_predictor, feature_extractor = load_all_models(model_folder)

    results = predict_single(
        image_path=image_path,
        model_cnn=cnn_model,
        feature_extractor=feature_extractor,
        xgb_model=xgb_model,
        rf_model=rf_model,
        attribute_predictor=attribute_predictor,
        SIZE=SIZE
    )

    # Get predicted breed for each model
    cnn_pred_idx = np.argmax(results["cnn_probs"])
    rf_pred_idx = np.argmax(results["rf_probs"])
    xgb_pred_idx = np.argmax(results["xgb_probs"])

    print(f"CNN (Xception) prediction: {breed_names[cnn_pred_idx]}")
    print(f"Random Forest prediction: {breed_names[rf_pred_idx]}")
    print(f"XGBoost prediction: {breed_names[xgb_pred_idx]}")
    print("Zero-Shot raw output:", results["zsl_pred"])