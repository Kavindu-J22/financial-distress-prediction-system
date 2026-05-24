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

# Helper columns used to determine distress under the correct rule
HELPER_COLS = ['Total Assets', 'Total Liabilities', 'Operating Profit (EBIT)', 'Operating Cash Flow']

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
    and handles missing target variables by applying the correct research methodology labeling rules.
    
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
        # Sort chronologically per company code before applying shifted/rolling conditions
        cleaned_df = cleaned_df.sort_values(by=['Company Code', 'Financial year']).reset_index(drop=True)
        
        # 1. Total Assets < Total Liabilities (Negative Net Assets) in Year t
        net_assets_neg = cleaned_df['Total Assets'] < cleaned_df['Total Liabilities']
        
        # 2. 3 consecutive years negative Operating Cash Flow (OCF < 0 in t, t-1, t-2)
        ocf = cleaned_df['Operating Cash Flow']
        ocf_t0 = ocf < 0
        ocf_t1 = cleaned_df.groupby('Company Code')['Operating Cash Flow'].shift(1) < 0
        ocf_t2 = cleaned_df.groupby('Company Code')['Operating Cash Flow'].shift(2) < 0
        three_years_neg_ocf = ocf_t0 & ocf_t1 & ocf_t2
        
        # 3. 3 consecutive years negative Operating Profit (EBIT < 0 in t, t-1, t-2)
        ebit = cleaned_df['Operating Profit (EBIT)']
        ebit_t0 = ebit < 0
        ebit_t1 = cleaned_df.groupby('Company Code')['Operating Profit (EBIT)'].shift(1) < 0
        ebit_t2 = cleaned_df.groupby('Company Code')['Operating Profit (EBIT)'].shift(2) < 0
        three_years_neg_ebit = ebit_t0 & ebit_t1 & ebit_t2
        
        # A firm is classified as distressed (1.0) if it satisfies any one of the three criteria
        cleaned_df[target_col] = np.where(
            net_assets_neg | three_years_neg_ocf | three_years_neg_ebit,
            1.0, 0.0
        )
        
        labeling_method = "Methodology Distress Rules: (Negative Net Assets in t) OR (3 Yrs Consecutive OCF < 0) OR (3 Yrs Consecutive Operating Losses)"
    else:
        labeling_method = "Original Dataset Target Labels"

    return df, cleaned_df, labeling_method

def preprocess_and_split_data(cleaned_df):
    """
    Separates features and targets, shifts features chronologically to use Year t-1 ratios to predict Year t distress,
    splits into 80% train and 20% test sets, and applies strict leakage-free preprocessing pipelines.
    
    Returns:
        X_train_res (pd.DataFrame): Scaled, SMOTE-balanced training features.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_train_res (pd.Series): SMOTE-balanced training targets.
        y_test (pd.Series): Test targets.
        scaler (StandardScaler): Fitted scaler object.
        imputer (SimpleImputer): Fitted imputer object.
    """
    # 1. Ensure cleaned_df is sorted chronologically per company code
    cleaned_df = cleaned_df.sort_values(by=['Company Code', 'Financial year']).reset_index(drop=True)
    
    # 2. Extract targets at Year t
    y_raw = cleaned_df['Target variable (0.1)'].copy()
    
    # 3. Lag the 15 financial ratios (shift by 1 year) within each company group
    # This represents X_{t-1} used to predict y_t
    X_lagged = cleaned_df.groupby('Company Code')[RATIOS].shift(1)
    
    # 4. We drop rows where the lagged features are completely missing (i.e. first year of data per company, like 2018)
    # The first year has no t-1 year data, so all its ratios will be NaN after shift.
    # We identify these rows where ALL 15 ratios are NaN.
    first_year_mask = X_lagged.isna().all(axis=1)
    
    # Keep only rows where lagged features are NOT completely missing
    # This drops 2018 records (the first year) completely from training/testing.
    X_aligned = X_lagged[~first_year_mask].copy()
    y_aligned = y_raw[~first_year_mask].copy()
    
    # 5. Stratified Train-Test Split (80% Train, 20% Test)
    # Using stratify=y_aligned ensures that both partitions have similar proportions of distress
    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned, y_aligned, test_size=0.20, random_state=42, stratify=y_aligned
    )
    
    # 6. Fit Imputer only on training set features, transform both
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=RATIOS, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=RATIOS, index=X_test.index)
    
    # 7. Apply SMOTE strictly on training set only (resolves class imbalance)
    smote = SMOTE(random_state=42)
    X_train_res_raw, y_train_res = smote.fit_resample(X_train_imputed, y_train)
    
    # 8. Fit StandardScaler only on resampled training set features, transform both
    scaler = StandardScaler()
    X_train_res_arr = scaler.fit_transform(X_train_res_raw)
    X_test_scaled_arr = scaler.transform(X_test_imputed)
    
    # Convert scaled features back to pandas DataFrames with correct feature columns
    X_train_res = pd.DataFrame(X_train_res_arr, columns=RATIOS)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=RATIOS)
    
    return X_train_res, X_test_scaled, y_train_res, y_test, scaler, imputer
