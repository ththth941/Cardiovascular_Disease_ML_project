import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    X = df.drop(["patientid", "target"], axis=1)
    y = df["target"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling the data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mean_std_values = {'mean': scaler.mean_, 'std': scaler.scale_}
    with open('models/mean_std_values_ML.pkl', 'wb') as f:
        pickle.dump(mean_std_values, f)
    
    return X_train, X_test, y_train, y_test, scaler
