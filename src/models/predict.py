import pandas as pd
import mlflow.xgboost

def load_model(model_uri: str):
    model = mlflow.xgboost.load_model(model_uri)
    print("Model loaded")
    return model

def predict(model, input_data: pd.DataFrame) -> pd.DataFrame:
    """Run prediction on new patient data"""
    probs = model.predict_proba(input_data)[:, 1]
    preds = model.predict(input_data)
    input_data['predicted_risk'] = probs
    input_data['predicted_label'] = preds
    return input_data

if __name__ == "__main__":
    model = load_model("models:/healthcare-risk/latest")
    df = pd.read_csv("../../data/processed/patient_data_features.csv")
    results = predict(model, df)
    print(results[['predicted_risk', 'predicted_label']].head())
