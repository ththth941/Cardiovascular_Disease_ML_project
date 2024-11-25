from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def tune_log_reg(X_train, y_train):
    log_reg_grid = {"C": np.logspace(-4, 4, 30), "solver": ["liblinear"]}
    rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, cv=5, n_iter=20)
    rs_log_reg.fit(X_train, y_train)
    return rs_log_reg

def tune_random_forest(X_train, y_train):
    rf_grid = {"n_estimators": np.arange(10, 100, 50), "max_depth": [None, 3, 5, 10], "min_samples_split": np.arange(2, 20, 2)}
    rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=100)
    rs_rf.fit(X_train, y_train)
    return rs_rf
