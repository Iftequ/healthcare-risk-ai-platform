import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values"""
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values handled")
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical columns"""
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    print("Categorical columns encoded")
    return df

def scale_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Standard scale numeric features"""
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Features scaled")
    return df

def preprocess(df: pd.DataFrame, target_col: str = 'readmitted') -> pd.DataFrame:
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    df = scale_features(df, target_col)
    return df

if __name__ == "__main__":
    from ingestion import load_data
    df = load_data("../../data/raw/patient_data.csv")
    df = preprocess(df)
    df.to_csv("../../data/processed/patient_data_processed.csv", index=False)
    print(df.head())
