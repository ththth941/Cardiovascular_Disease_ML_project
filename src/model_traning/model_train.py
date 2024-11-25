
# Import các thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# mô hình từ thư viện scikitlearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# thư viện đánh giá mô hình
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("src\\data\\Cardiovascular_Disease_Dataset.csv")
print(df.head(10))

nRow, nCol = df.shape
print("Shape of dataset {}".format(df.shape))
print(f"Rows: {nRow} \nColumns: {nCol}")

print(df.columns)

# Split data into X and y
X = df.drop(["patientid", "target"], axis=1)

y = df["target"]

# Chia data thành tập train và tập test
np.random.seed(42)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mean_std_values = {'mean': scaler.mean_, 'std': scaler.scale_}
with open('src\\model\\mean_std_values_ML.pkl', 'wb') as f:
    pickle.dump(mean_std_values, f)
    
# Đưa các model vào dictionary

models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}


# Tạo hàm để fit và chấm điểm model

def fit_and_score(models, X_train, X_test, y_train, y_test):
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    for name, model in models.items():
        # Fit model vào data
        model.fit(X_train, y_train)
        # Chấm điểm model
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)
print(model_scores)

# So Sánh Model 

model_compare = pd.DataFrame(model_scores, index=["Accuracy"])
model_compare.T.plot.bar()
plt.xticks(rotation=0);

# Hyperparameter Tuning

# Tuning KNN

train_scores = []
test_scores = []

# Tạo danh sách các giá trị khác nhau cho n_neighbors
neighbors = range(1, 21)

knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    knn.fit(X_train, y_train)

    train_scores.append(knn.score(X_train, y_train))

    # Đưa vào test score
    test_scores.append(knn.score(X_test, y_test))


knn.set_params(n_neighbors=np.argmax(test_scores) + 1)
knn.fit(X_train, y_train)

plt.figure(figsize=(15, 8))
plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model Scores")
plt.legend()
print(f"Maximum KNN score on the test data: {max(test_scores) * 100:.2f}%");

# Hyperparameter tuning with Randomized SearchCV for LogisticRegression, RandomForestCLassfier

# Tạo hyperparameter grid cho Logistic Regression

log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# Tạo hyperparameter grid cho RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 100, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

# Tune LogisticRegression

np.random.seed(42)

# Setup random hyperparameter search for logisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit Random hyperparameter search model for logisticRegression
rs_log_reg.fit(X_train, y_train)

# %%
print(rs_log_reg.best_params_)

# %%
rs_log_reg.score(X_test, y_test)

# %% [markdown]
# Now as we've tuned logisticRegression(), Lets do same with RandomForestClassfiers()

# %%
# Setup Random seed
print(np.random.seed(42))

# Setup random hyperparameters search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=100,
                           verbose=True)

# Fit the random Hyperprameter search mode for randomforestclassifier()
rs_rf.fit(X_train, y_train)

# Find the best hyperparameters
print(rs_rf.best_params_)

# Evaluate the randomized search RandomForestClassifier model
print(rs_rf.score(X_test, y_test))

sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Dùng seaborn's heatmap để tạo ma trận nhầm lẫn(
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")


y_preds = knn.predict(X_test)
plot_conf_mat(y_test, y_preds)
print("KNN Classification Report:")
print(classification_report(y_test, y_preds))

# Logistic Regression
# Evaluate the model with best parameters
rs_log_reg_model = rs_log_reg.best_estimator_
y_preds_rs_log_reg_model = rs_log_reg_model.predict(X_test)
plot_conf_mat(y_test, y_preds_rs_log_reg_model)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_preds_rs_log_reg_model))

# Random Forest
# Evaluate the model with best parameters
rs_rf_model = rs_rf.best_estimator_
y_preds_rs_rf_model = rs_rf_model.predict(X_test)
plot_conf_mat(y_test, y_preds_rs_rf_model)
print("Random Forest Classification Report for Random Forest:")
print(classification_report(y_test, y_preds_rs_rf_model))

knn.score(X_test, y_test)

# Cross-validated accuaracy
cv_acc_knn = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
cv_acc_knn = np.mean(cv_acc_knn)


# Cross-validated precision
cv_precision_knn = cross_val_score(knn, X, y, cv=5, scoring="precision")
cv_precision_knn = np.mean(cv_precision_knn)


# Cross-validated recall
cv_recall_knn = cross_val_score(knn, X, y, cv=5, scoring="recall")
cv_recall_knn = np.mean(cv_recall_knn)


# Cross-validated f1-score
cv_f1_knn = cross_val_score(knn, X, y, cv=5, scoring="f1")
cv_f1_knn = np.mean(cv_f1_knn)


# Visuzalize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc_knn,
                           "Precision": cv_precision_knn,
                           "Recall": cv_recall_knn,
                           "F1": cv_f1_knn},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics for KNN",
                      legend=False)
plt.xticks(rotation=0);

# LogististRegression
# Cross-validated accuracy
cv_acc_lr = cross_val_score(rs_log_reg_model, X, y, cv=5, scoring="accuracy")
cv_acc_lr = np.mean(cv_acc_lr)
# Cross-validated precision
cv_precision_lr = cross_val_score(rs_log_reg_model, X, y, cv=5, scoring="precision")
cv_precision_lr = np.mean(cv_precision_lr)
# Cross-validated recall
cv_recall_lr = cross_val_score(rs_log_reg_model, X, y, cv=5, scoring="recall")
cv_recall_lr = np.mean(cv_recall_lr)
# Cross-validated f1-score
cv_f1_lr = cross_val_score(rs_log_reg_model, X, y, cv=5, scoring="f1")
cv_f1_lr = np.mean(cv_f1_lr)
# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc_lr,
                           "Precision": cv_precision_lr,
                           "Recall": cv_recall_lr,
                           "F1": cv_f1_lr},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics for LR",
                      legend=False)
plt.xticks(rotation=0)
plt.show()

# RandomForestRandomForest
# Cross-validated accuracy
cv_acc_rf = cross_val_score(rs_rf_model, X, y, cv=5, scoring="accuracy")
cv_acc_rf = np.mean(cv_acc_rf)
# Cross-validated precision
cv_precision_rf = cross_val_score(rs_rf_model, X, y, cv=5, scoring="precision")
cv_precision_rf = np.mean(cv_precision_rf)
# Cross-validated recall
cv_recall_rf = cross_val_score(rs_rf_model, X, y, cv=5, scoring="recall")
cv_recall_rf = np.mean(cv_recall_rf)
# Cross-validated f1-score
cv_f1_rf = cross_val_score(rs_rf_model, X, y, cv=5, scoring="f1")
cv_f1_rf = np.mean(cv_f1_rf)
# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc_rf,
                           "Precision": cv_precision_rf,
                           "Recall": cv_recall_rf,
                           "F1": cv_f1_rf},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics for Random Forest",
                      legend=False)
plt.xticks(rotation=0)
plt.show()

# Lưu kết quả của tất cả các mô hình vào DataFrame
cv_results = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Random Forest'],
    'Accuracy': [cv_acc_knn, cv_acc_lr, cv_acc_rf],
    'Precision': [cv_precision_knn, cv_precision_lr, cv_precision_rf],
    'Recall': [cv_recall_knn, cv_recall_lr, cv_recall_rf],
    'F1': [cv_f1_knn, cv_f1_lr, cv_f1_rf]
})

# Vẽ biểu đồ so sánh các mô hình
cv_results.set_index('Model').plot.bar(title="Comparison of models with cross-validation", figsize=(10, 6))
plt.xticks(rotation=0)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Giả sử y_test là nhãn thật và y_preds là các dự đoán của mô hình
mae_knn_cv = np.round(mean_absolute_error(y_test, y_preds), 2)
mse_knn_cv = np.round(mean_squared_error(y_test, y_preds), 2)
rmse_knn_cv = np.round(np.sqrt(mean_squared_error(y_test, y_preds)), 2)

mae_lr_cv = np.round(mean_absolute_error(y_test, y_preds_rs_log_reg_model), 2)
mse_lr_cv = np.round(mean_squared_error(y_test, y_preds_rs_log_reg_model), 2)
rmse_lr_cv = np.round(np.sqrt(mean_squared_error(y_test, y_preds_rs_log_reg_model)), 2)

mae_rf_cv = np.round(mean_absolute_error(y_test, y_preds_rs_rf_model), 2)
mse_rf_cv = np.round(mean_squared_error(y_test, y_preds_rs_rf_model), 2)
rmse_rf_cv = np.round(np.sqrt(mean_squared_error(y_test, y_preds_rs_rf_model)), 2)

models = ['KNeighborsClassifier', 'LogisticRegression', 'RandomForestClassifier']
data = [[mae_knn_cv, mse_knn_cv, rmse_knn_cv], [mae_lr_cv, mse_lr_cv, rmse_lr_cv], [mae_rf_cv, mse_rf_cv, rmse_rf_cv]]
cols = ['mae', 'mse', 'rmse']
pd.DataFrame(data=data, index=models, columns=cols)

# Feature Importance

clf = LogisticRegression(C=0.20433597178569418,
                         solver='liblinear')
clf.fit(X_train, y_train);

# Check Coef_
print(clf.coef_)

# Match coef's of features to coloumns

feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
print(feature_dict)

# Visualize Feature Importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature importance", legend=False);


# Lưu mô hình có hiệu quả tốt nhất
model_filename = 'E:\\Python Projects\\Heart-Disease_Project\\model\\model_ML.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rs_rf_model, file)
print('Model Saved Succesfully!')


