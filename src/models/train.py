import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.xgboost

def train(file_path: str, target_col: str = 'readmitted'):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("healthcare-risk")

    with mlflow.start_run():
        model = XGBClassifier(n_estimators=100, max_depth=5,
                               learning_rate=0.1, use_label_encoder=False,
                               eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {auc:.4f}")

        mlflow.log_metric("auc_roc", auc)
        mlflow.xgboost.log_model(model, "model")

    return model

if __name__ == "__main__":
    train("../../data/processed/patient_data_features.csv")
