from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def initialize_models():
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    return models

def train_models(models, X_train, y_train):
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
