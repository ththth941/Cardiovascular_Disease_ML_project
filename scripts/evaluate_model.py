from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def evaluate_accuracy(model, X_test_scaled, y_test):
    """
    Đánh giá độ chính xác của mô hình.
    """
    accuracy = model.score(X_test_scaled, y_test)
    return accuracy

def evaluate_confusion_matrix(model, X_test_scaled, y_test):
    """
    Đánh giá mô hình bằng ma trận nhầm lẫn.
    """
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    return cm, cr

def evaluate_cross_validation(model, X_scaled, y):
    """
    Đánh giá mô hình bằng cross-validation.
    """
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    return cv_scores
