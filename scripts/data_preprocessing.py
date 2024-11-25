import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV.
    """
    df = pd.read_csv('data\\Cardiovascular_Disease_Dataset.csv')
    return df

def preprocess_data(df):
    """
    Tiền xử lý dữ liệu: tách biến đầu vào (X) và nhãn (y), chia dữ liệu train/test, chuẩn hóa dữ liệu.
    """
    X = df.drop(columns=['patientid', 'target'])
    y = df['target']
    
    # Chia tập dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
