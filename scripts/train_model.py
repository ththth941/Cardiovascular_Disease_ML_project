import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train_scaled, y_train):
    """
    Huấn luyện mô hình Logistic Regression.
    """
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    return lr

def train_knn(X_train_scaled, y_train, n_neighbors=5):
    """
    Huấn luyện mô hình K-Nearest Neighbors.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    return knn

def train_random_forest(X_train_scaled, y_train, n_estimators=100):
    """
    Huấn luyện mô hình Random Forest.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train_scaled, y_train)
    return rf

def save_model(model, model_name):
    """
    Lưu mô hình đã huấn luyện vào file bằng pickle.
    """
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
