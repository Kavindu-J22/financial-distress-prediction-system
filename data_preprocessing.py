import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# List of the 15 financial ratio features requested
RATIOS = [
    'Return on Assets', 'Operating Profit Margin', 'Return on Equity', 
    'Current Ratio', 'Quick Ratio', 'Working Capital to Total Assets', 
    'Leverage', 'Debt Ratio', 'Asset Turnover Ratio', 'Capital Turnover Ratio', 
    'Cash Flow to Net Income', 'Cash Flow to Total Liabilities', 
    'Cash Flow to Total Assets', 'Cash Flow to Current Liabilities', 
    'Cash Flow to Total Debt'
]

# Helper columns used to determine distress under the dynamic rule
HELPER_COLS = ['Total Assets', 'Total Liabilities', 'Net Income (or Net Profit)']

def clean_numeric(val):
    """
    Cleans Excel/CSV data cells to ensure correct numeric parsing.
    Handles date/time fractions, commas, trailing spaces, and empty placeholders.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, datetime.time):
        total_seconds = val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
        return total_seconds / 86400.0  # Excel time representation as a fraction of a day
    if isinstance(val, datetime.datetime):
        total_seconds = val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
        return total_seconds / 86400.0
    if isinstance(val, str):
        # Remove common formatting characters
        val = val.replace(',', '').replace(' ', '').strip()
        if val == '-' or val == '' or val.lower() in ['nan', 'none', 'null']:
            return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def load_and_clean_data(file_path_or_buffer=None):
    """
    Loads raw Excel/CSV dataset, applies deep numeric cleaning to all features, 
    and handles missing target variables by applying robust dynamic labeling fallback rules.
    
    Returns:
        raw_df (pd.DataFrame): Loaded dataframe before cleaning.
        cleaned_df (pd.DataFrame): Sanitized dataframe with target variables populated.
        labeling_method (str): A description of the method used to define the targets.
    """
    if file_path_or_buffer is not None:
        if hasattr(file_path_or_buffer, 'name') and file_path_or_buffer.name.endswith('.csv'):
            df = pd.read_csv(file_path_or_buffer)
        else:
            df = pd.read_excel(file_path_or_buffer)
    else:
        # Default fallback to the workspace local dataset
        import os
        local_path = os.path.join("Dataset", "Final data set 2 .xlsx")
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local dataset not found at {local_path} and no buffer provided.")
        df = pd.read_excel(local_path)

    # Check for critical metadata columns
    required_metadata = ['Company Code', 'Financial year']
    for col in required_metadata:
        if col not in df.columns:
            raise KeyError(f"Critical Metadata Column '{col}' is missing in the dataset.")

    # Check target column
    target_col = 'Target variable (0.1)'
    if target_col not in df.columns:
        df[target_col] = np.nan

    cleaned_df = df.copy()
    
    # Identify all columns that need to be parsed numerically
    cols_to_clean = list(set(HELPER_COLS + RATIOS + [target_col]))
    for col in cols_to_clean:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(clean_numeric)

    # Determine if target column is empty/all NaN
    is_target_empty = (
        cleaned_df[target_col].isna().all() or 
        (cleaned_df[target_col] == 0.0).all() and len(cleaned_df[target_col].unique()) == 1
    )

    if is_target_empty:
        # Check if helper columns are present for the premium dynamic rule
        missing_helpers = [h for h in HELPER_COLS if h not in cleaned_df.columns]
        if not missing_helpers:
            cleaned_df[target_col] = np.where(
                (cleaned_df['Total Assets'] < cleaned_df['Total Liabilities']) | 
                ((cleaned_df['Net Income (or Net Profit)'] < 0) & (cleaned_df['Current Ratio'] < 1.0)),
                1.0, 0.0
            )
            labeling_method = "Dynamic Fallback Rule: (Assets < Liabilities) OR (Net Income < 0 AND Current Ratio < 1)"
        else:
            # If assets/liabilities are missing, fall back to simple negative ROA or high debt ratio
            cleaned_df[target_col] = np.where(
                (cleaned_df['Return on Assets'] < 0.0) | (cleaned_df['Debt Ratio'] > 0.8),
                1.0, 0.0
            )
            labeling_method = "Basic Fallback Rule: (ROA < 0) OR (Debt Ratio > 0.8)"
    else:
        labeling_method = "Original Dataset Target Labels"

    return df, cleaned_df, labeling_method

def preprocess_and_split_data(cleaned_df):
    """
    Separates features and targets, splits into 80% train and 20% test sets, 
    and applies strict leakage-free preprocessing pipelines (Imputation, SMOTE, and Scaling).
    
    Returns:
        X_train_res (pd.DataFrame): Scaled, SMOTE-balanced training features.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_train_res (pd.Series): SMOTE-balanced training targets.
        y_test (pd.Series): Test targets.
        scaler (StandardScaler): Fitted scaler object.
        imputer (SimpleImputer): Fitted imputer object.
    """
    # 1. Extract 15 features and 1 target variable
    X = cleaned_df[RATIOS].copy()
    y = cleaned_df['Target variable (0.1)'].copy()
    
    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Fit Imputer only on training set features, transform both
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=RATIOS, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=RATIOS, index=X_test.index)
    
    # 4. Apply SMOTE strictly on training set only (resolves class imbalance)
    smote = SMOTE(random_state=42)
    X_train_res_raw, y_train_res = smote.fit_resample(X_train_imputed, y_train)
    
    # 5. Fit StandardScaler only on resampled training set features, transform both
    scaler = StandardScaler()
    X_train_res_arr = scaler.fit_transform(X_train_res_raw)
    X_test_scaled_arr = scaler.transform(X_test_imputed)
    
    # Convert scaled features back to pandas DataFrames with correct feature columns
    X_train_res = pd.DataFrame(X_train_res_arr, columns=RATIOS)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=RATIOS)
    
    return X_train_res, X_test_scaled, y_train_res, y_test, scaler, imputer
