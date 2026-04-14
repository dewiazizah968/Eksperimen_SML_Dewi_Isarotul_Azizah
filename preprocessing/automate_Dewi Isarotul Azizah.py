import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    # Memuat dataset dari file CSV.
    df = pd.read_csv(filepath)
    print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Menghapus kolom yang tidak relevan untuk model.
    df = df.drop(columns=["customerID"])
    print(f"Kolom customerID dihapus. Shape: {df.shape}")
    return df

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # Mengkonversi TotalCharges ke numerik dan menghapus missing values.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    after = len(df)
    print(f"TotalCharges dikonversi. Baris dihapus: {before - after}")
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Menghapus data duplikat.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Duplikat dihapus: {before - after} baris")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # Encoding fitur kategorikal menggunakan Label Encoding.
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Churn"]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Encode target: Yes=1, No=0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"Encoding selesai untuk {len(cat_cols)} kolom kategorikal.")
    return df

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalisasi fitur numerik menggunakan StandardScaler.
    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"Normalisasi selesai untuk kolom: {cols_to_scale}")
    return df

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    
    print("=" * 50)
    print("Memulai proses preprocessing...")
    print("=" * 50)

    df = load_data(input_path)
    df = drop_unnecessary_columns(df)
    df = fix_total_charges(df)
    df = drop_duplicates(df)
    df = encode_categorical(df)
    df = normalize_features(df)

    # Simpan hasil
    df.to_csv(output_path, index=False)

    print("=" * 50)
    print(f"Preprocessing selesai!")
    print(f"Shape akhir     : {df.shape}")
    print(f"Missing values  : {df.isnull().sum().sum()}")
    print(f"Distribusi target:")
    print(df["Churn"].value_counts().to_string())
    print(f"File disimpan ke: {output_path}")
    print("=" * 50)

    return df

if __name__ == "__main__":
    INPUT_PATH  = "Telco-Customer-Churn.csv"
    OUTPUT_PATH = "telco_churn_preprocessing.csv"
    preprocess(INPUT_PATH, OUTPUT_PATH)
