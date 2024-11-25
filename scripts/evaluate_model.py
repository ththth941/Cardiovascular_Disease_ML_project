import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cross_val_score

def plot_conf_matrix(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")

def evaluate_model(model, X_test, y_test):
    y_preds = model.predict(X_test)
    plot_conf_matrix(y_test, y_preds)
    print(f"{model.__class__.__name__} Classification Report:")
    print(classification_report(y_test, y_preds))

def cross_validate_model(model, X, y):
    cv_acc = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return cv_acc.mean()
