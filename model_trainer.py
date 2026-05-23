from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains baseline Logistic Regression and advanced Random Forest models,
    generates detailed performance metrics, and extracts feature importances.
    
    Parameters:
        X_train (pd.DataFrame or np.ndarray): Preprocessed and resampled training features.
        X_test (pd.DataFrame or np.ndarray): Preprocessed test features.
        y_train (pd.Series or np.ndarray): Preprocessed and resampled training labels.
        y_test (pd.Series or np.ndarray): Test labels.
        
    Returns:
        lr_model (LogisticRegression): The fitted Logistic Regression baseline.
        rf_model (RandomForestClassifier): The fitted Random Forest ensemble.
        metrics (dict): Dictionary of performance reports and confusion matrices.
        feature_importances (np.ndarray): Random Forest feature importances.
    """
    # 1. Model 1: Logistic Regression (Baseline)
    # Using max_iter=1000 to ensure convergence on volatile market data
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 2. Model 2: RandomForestClassifier (Ensemble)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 3. Model Predictions on Stratified Holdout Test Set
    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    
    # 4. Generate Classification Reports as Dictionaries (read by UI)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    
    # 5. Generate Confusion Matrices
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    # Pack metrics for UI consumption
    metrics = {
        'report_lr': report_lr,
        'report_rf': report_rf,
        'cm_lr': cm_lr,
        'cm_rf': cm_rf
    }
    
    # 6. Extract Feature Importance from the Random Forest model
    feature_importances = rf_model.feature_importances_
    
    return lr_model, rf_model, metrics, feature_importances
