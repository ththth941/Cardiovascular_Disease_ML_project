from src.data_preprocessing import load_data, preprocess_data
from src.model_training import initialize_models, train_models
from src.model_tuning import tune_log_reg, tune_random_forest
from src.model_evaluation import evaluate_model
from src.utils import save_model

# Load and preprocess data
df = load_data("data\\Cardiovascular_Disease_Dataset.csv")
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Initialize and train models
models = initialize_models()
trained_models = train_models(models, X_train, y_train)

# Tune models
rs_log_reg = tune_log_reg(X_train, y_train)
rs_rf = tune_random_forest(X_train, y_train)

# Evaluate models
evaluate_model(rs_log_reg.best_estimator_, X_test, y_test)
evaluate_model(rs_rf.best_estimator_, X_test, y_test)

# Save the best model
save_model(rs_rf.best_estimator_, "model/model_ML.pkl")
