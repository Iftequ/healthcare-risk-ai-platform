import pandas as pd

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into groups"""
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 40, 60, 80, 120],
                                  labels=['child', 'young', 'middle', 'senior', 'elderly'])
    return df

def add_visit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from visit history"""
    if 'num_visits' in df.columns:
        df['high_utilizer'] = (df['num_visits'] > df['num_visits'].median()).astype(int)
    return df

def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Simple composite risk score"""
    cols = [c for c in ['num_diagnoses', 'num_medications', 'num_procedures'] if c in df.columns]
    if cols:
        df['risk_score'] = df[cols].sum(axis=1)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_age_group(df)
    df = add_visit_features(df)
    df = add_risk_score(df)
    print("Feature engineering complete")
    return df

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/patient_data_processed.csv")
    df = engineer_features(df)
    df.to_csv("../../data/processed/patient_data_features.csv", index=False)
    print(df.head())
