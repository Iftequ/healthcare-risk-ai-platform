import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    df = load_data("../../data/raw/patient_data.csv")
    print(df.head())
