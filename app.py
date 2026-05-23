import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Import modular components and constants
from data_preprocessing import RATIOS, load_and_clean_data, preprocess_and_split_data
from model_trainer import train_and_evaluate_models

# Page Configuration
st.set_page_config(
    page_title="CSE Corporate Distress Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Glassmorphism Styling & Montserrat/Inter Typography Injection
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700;800&display=swap');
        
        /* Global typography */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3, .title-text {
            font-family: 'Montserrat', sans-serif;
        }
        
        /* Glassmorphic KPI Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            text-align: center;
            margin-bottom: 15px;
        }
        
        .metric-title {
            color: #64748b;
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        
        .metric-value {
            color: #1e3a8a;
            font-size: 2.2rem;
            font-weight: 800;
            font-family: 'Montserrat', sans-serif;
        }
        
        .metric-label-lr {
            border-bottom: 3px solid #636EFA;
            padding-bottom: 5px;
        }
        
        .metric-label-rf {
            border-bottom: 3px solid #00CC96;
            padding-bottom: 5px;
        }
        
        /* Prediction results banners */
        .prediction-distress {
            background-color: #fff1f0;
            border-left: 6px solid #ff4d4f;
            padding: 22px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(255, 77, 79, 0.08);
        }
        
        .prediction-healthy {
            background-color: #f6ffed;
            border-left: 6px solid #52c41a;
            padding: 22px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(82, 196, 26, 0.08);
        }
        
        /* Custom sidebars and tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            padding: 12px 24px;
            border-radius: 8px 8px 0 0;
            background-color: #f1f5f9;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #1e3a8a;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Streamlit Caching Wrappers -----------------

@st.cache_data
def load_cleaned_dataset(uploaded_file=None):
    """
    Wrapper function to cache the loading and cleaning of raw dataset.
    """
    try:
        raw_df, cleaned_df, labeling_method = load_and_clean_data(uploaded_file)
        return raw_df, cleaned_df, labeling_method
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None, None, str(e)

@st.cache_resource
def run_ml_pipeline(cleaned_df):
    """
    Wrapper function to cache the preprocessing split, SMOTE and model execution.
    Provides identical return structures as monolithic app.py to ensure zero KeyErrors.
    """
    if cleaned_df is None:
        return None
        
    # 1. Preprocess and Split Data (Strict Leakage Prevention)
    X_train_res, X_test_scaled, y_train_res, y_test, scaler, imputer = preprocess_and_split_data(cleaned_df)
    
    # 2. Train and Evaluate Models
    lr_model, rf_model, metrics, feature_importances = train_and_evaluate_models(
        X_train_res, X_test_scaled, y_train_res, y_test
    )
    
    # 3. Recalculate raw training and testing partitions for class distribution counts
    X = cleaned_df[RATIOS].copy()
    y = cleaned_df['Target variable (0.1)'].copy()
    X_train, X_test, y_train, y_test_raw = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    class_counts_before = y_train.value_counts().to_dict()
    class_counts_after = y_train_res.value_counts().to_dict()
    
    # Pack exactly in the form predicted by app.py UI
    pipeline_results = {
        'scaler': scaler,
        'imputer': imputer,
        'lr_model': lr_model,
        'rf_model': rf_model,
        'X_train_raw': X_train,
        'X_train_res': X_train_res,
        'y_train_res': y_train_res,
        'X_test': X_test,
        'y_test': y_test,
        'report_lr': metrics['report_lr'],
        'report_rf': metrics['report_rf'],
        'cm_lr': metrics['cm_lr'],
        'cm_rf': metrics['cm_rf'],
        'feature_importances': feature_importances,
        'class_counts_before': class_counts_before,
        'class_counts_after': class_counts_after
    }
    
    return pipeline_results

# ----------------- Dashboard Layout -----------------

# Header Section
st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 30px; border-radius: 14px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); border-left: 8px solid #00cc96;">
        <h1 style="color: white; margin: 0; font-family: 'Montserrat', sans-serif; font-size: 2.3rem; font-weight: 700; letter-spacing: -0.02em;">Predicting Corporate Financial Distress in Emerging Markets</h1>
        <p style="color: #94a3b8; margin: 8px 0 0 0; font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 400; max-width: 900px;">
            A Comparative analysis of <b>Logistic Regression</b> vs. <b>Ensemble Machine Learning</b> on the Colombo Stock Exchange (CSE) (2019–2024). Includes median imputation, stratified train-test splits, SMOTE balancing, and real-time predictor.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <h3 style="margin: 0; font-family: 'Montserrat', sans-serif; font-weight: 700; color: #1e3a8a;">⚙️ Dashboard Options</h3>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Custom Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (.xlsx) or CSV file containing corporate ratios.",
    type=["xlsx", "csv"]
)

# Load data and run models using cached loaders
raw_df, cleaned_df, labeling_method = load_cleaned_dataset(uploaded_file)
st.session_state['labeling_method'] = labeling_method

if cleaned_df is not None:
    # 2. Calculate dynamic sample statistics custom variables
    total_unique_firms = cleaned_df['Company Code'].nunique()
    total_firm_year_obs = len(cleaned_df)
    
    # Run modular ML pipeline
    pipeline = run_ml_pipeline(cleaned_df)
    
    # Initialize ratio inputs in Session State for Tab 3 if not present
    for r in RATIOS:
        if r not in st.session_state:
            # Default to median values of healthy companies to keep inputs realistic
            healthy_firms = cleaned_df[cleaned_df['Target variable (0.1)'] == 0]
            if len(healthy_firms) > 0:
                healthy_median = float(healthy_firms[r].median())
            else:
                healthy_median = 0.0
            st.session_state[r] = healthy_median if not np.isnan(healthy_median) else 0.0

    # Tabs Layout
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset Overview & Preprocessing", 
        "🧠 Model Evaluation & Feature Importance", 
        "🔮 Real-Time Corporate Distress Predictor"
    ])
    
    # ----------------- TAB 1: DATASET OVERVIEW & PREPROCESSING -----------------
    with tab1:
        st.markdown("### 📋 CSE Dataset Specifications & Cleaning Profile")
        
        # Upper KPI row
        col_shape, col_firms, col_years, col_lbl = st.columns(4)
        with col_shape:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Dataset Structure</div>
                    <div class="metric-value">{cleaned_df.shape[0]} × {cleaned_df.shape[1]}</div>
                    <div style="font-size:0.85rem; color:#64748b; margin-top:5px;">Rows × Columns</div>
                </div>
            """, unsafe_allow_html=True)
        with col_firms:
            num_firms = cleaned_df['Company Code'].nunique()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Unique Companies</div>
                    <div class="metric-value">{num_firms}</div>
                    <div style="font-size:0.85rem; color:#64748b; margin-top:5px;">Listed on CSE</div>
                </div>
            """, unsafe_allow_html=True)
        with col_years:
            min_yr = int(cleaned_df['Financial year'].min())
            max_yr = int(cleaned_df['Financial year'].max())
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Coverage Period</div>
                    <div class="metric-value">{min_yr}-{max_yr}</div>
                    <div style="font-size:0.85rem; color:#64748b; margin-top:5px;">Financial Years</div>
                </div>
            """, unsafe_allow_html=True)
        with col_lbl:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                    <div class="metric-title">Target Labeling Mode</div>
                    <div class="metric-value" style="font-size:1.15rem; font-weight:700; height:45px; display:flex; align-items:center; justify-content:center; color:#b45309;">
                        {"Fallback Dynamic Rule" if "Fallback" in st.session_state['labeling_method'] else "Preset Excel Labels"}
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:5px; font-style:italic;">
                        {st.session_state['labeling_method']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        
        # Class distribution charts (Before vs After SMOTE)
        st.markdown("### ⚖️ Target Class Balancing Analysis (SMOTE Pipeline)")
        st.info("SMOTE (Synthetic Minority Over-sampling Technique) is applied strictly on the **Training Set** to synthesize new minority instances, addressing severe class imbalance and ensuring unbiased training.")
        
        col_chart1, col_chart2 = st.columns(2)
        
        # 1. Before SMOTE Chart
        with col_chart1:
            counts_before = pipeline['class_counts_before']
            df_before = pd.DataFrame({
                'Status': ['Healthy (0)', 'Distressed (1)'],
                'Count': [counts_before.get(0.0, 0), counts_before.get(1.0, 0)]
            })
            fig_before = px.bar(
                df_before, x='Status', y='Count',
                title="Class Balance in Training Partition BEFORE SMOTE",
                color='Status',
                color_discrete_map={'Healthy (0)': '#636EFA', 'Distressed (1)': '#EF553B'},
                text='Count'
            )
            fig_before.update_layout(
                font_family="Inter",
                title_font_family="Montserrat",
                title_font_size=15,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=40)
            )
            fig_before.update_traces(textposition='outside')
            fig_before.update_yaxes(showgrid=True, gridcolor='#e2e8f0', zeroline=False)
            st.plotly_chart(fig_before, use_container_width=True)
            
        # 2. After SMOTE Chart
        with col_chart2:
            counts_after = pipeline['class_counts_after']
            df_after = pd.DataFrame({
                'Status': ['Healthy (0)', 'Distressed (1)'],
                'Count': [counts_after.get(0.0, 0), counts_after.get(1.0, 0)]
            })
            fig_after = px.bar(
                df_after, x='Status', y='Count',
                title="Class Balance in Training Partition AFTER SMOTE",
                color='Status',
                color_discrete_map={'Healthy (0)': '#636EFA', 'Distressed (1)': '#EF553B'},
                text='Count'
            )
            fig_after.update_layout(
                font_family="Inter",
                title_font_family="Montserrat",
                title_font_size=15,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=40)
            )
            fig_after.update_traces(textposition='outside')
            fig_after.update_yaxes(showgrid=True, gridcolor='#e2e8f0', zeroline=False)
            st.plotly_chart(fig_after, use_container_width=True)

        # 3. Display the exact sample statistics dynamically before the dataset preview
        st.write("")
        st.markdown("### 📊 Sample Statistics Summary")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Unique Firms (Companies)</div>
                    <div class="metric-value">{total_unique_firms}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_stat2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Firm-Year Observations</div>
                    <div class="metric-value">{total_firm_year_obs}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔍 Cleaned Dataset Preview")
        # Missing values profile
        with st.expander("📝 View Missing Values and Feature Processing Details"):
            col_miss_info, col_imput_info = st.columns([1, 1])
            with col_miss_info:
                st.markdown("**Original Missing Value Count (NaN) in the 15 Predictors:**")
                missing_counts = raw_df[RATIOS].isna().sum()
                st.write(missing_counts[missing_counts > 0])
                if missing_counts.sum() == 0:
                    st.success("No missing values in raw predictor ratios!")
            with col_imput_info:
                st.markdown("**Preprocessing Strategy Details (Modular Strict Architecture):**")
                st.write("1. **Imputation:** Missing predictor cells are filled using **Median Imputation** based on the training features distribution.")
                st.write("2. **Splitting:** 80% train and 20% test subsets are split using **Stratification** on the target class to maintain class balance in both partitions.")
                st.write("3. **Scaling:** Scaled using a **StandardScaler** to ensure all 15 indicators have a mean of 0 and unit variance, preventing high-leverage ratios from dominating.")
                st.write("4. **Zero-Leakage Guarantee:** `SimpleImputer` and `StandardScaler` are fitted strictly on the training partition to ensure zero test partition leakage.")

        st.dataframe(cleaned_df.head(100), use_container_width=True)

    # ----------------- TAB 2: MODEL EVALUATION & ANALYSIS -----------------
    with tab2:
        st.markdown("### ⚔️ Comparative Performance Analysis")
        st.write("Comparing Logistic Regression (Linear Baseline) vs. Random Forest Classifier (Advanced Decision Tree Ensemble) on the Stratified Test Partition.")
        
        rep_lr = pipeline['report_lr']
        rep_rf = pipeline['report_rf']
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision (Distressed)', 'Recall (Distressed)', 'F1-Score (Distressed)'],
            'Logistic Regression (Baseline)': [
                rep_lr['accuracy'], 
                rep_lr['1.0']['precision'], 
                rep_lr['1.0']['recall'], 
                rep_lr['1.0']['f1-score']
            ],
            'Random Forest (Ensemble)': [
                rep_rf['accuracy'], 
                rep_rf['1.0']['precision'], 
                rep_rf['1.0']['recall'], 
                rep_rf['1.0']['f1-score']
            ]
        })
        
        # Display side-by-side KPI cards
        col_lr_acc, col_rf_acc, col_lr_f1, col_rf_f1 = st.columns(4)
        with col_lr_acc:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title"><span class="metric-label-lr">Logistic Regression Accuracy</span></div>
                    <div class="metric-value" style="color: #636EFA;">{rep_lr['accuracy'] * 100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col_rf_acc:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title"><span class="metric-label-rf">Random Forest Accuracy</span></div>
                    <div class="metric-value" style="color: #00CC96;">{rep_rf['accuracy'] * 100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col_lr_f1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title"><span class="metric-label-lr">LR Distress F1-Score</span></div>
                    <div class="metric-value" style="color: #636EFA;">{rep_lr['1.0']['f1-score'] * 100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col_rf_f1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title"><span class="metric-label-rf">RF Distress F1-Score</span></div>
                    <div class="metric-value" style="color: #00CC96;">{rep_rf['1.0']['f1-score'] * 100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # Comparative Table
        st.markdown("#### 📊 Complete Stratified Test Set Evaluation Table")
        st.dataframe(
            metrics_df.style.format({
                'Logistic Regression (Baseline)': '{:.2%}',
                'Random Forest (Ensemble)': '{:.2%}'
            }),
            use_container_width=True
        )
        
        # Side-by-side Confusion Matrices
        st.markdown("### 🔲 Confusion Matrices (Holdout Set)")
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            cm_lr = pipeline['cm_lr']
            fig_cm1 = px.imshow(
                cm_lr,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted Label", y="Actual Label", color="Samples"),
                x=['Healthy (0)', 'Distressed (1)'],
                y=['Healthy (0)', 'Distressed (1)'],
                title="Logistic Regression Confusion Matrix",
                color_continuous_scale="Blues"
            )
            fig_cm1.update_layout(
                font_family="Inter",
                title_font_family="Montserrat",
                title_font_size=15,
                margin=dict(t=50, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_cm1, use_container_width=True)
            
        with col_cm2:
            cm_rf = pipeline['cm_rf']
            fig_cm2 = px.imshow(
                cm_rf,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted Label", y="Actual Label", color="Samples"),
                x=['Healthy (0)', 'Distressed (1)'],
                y=['Healthy (0)', 'Distressed (1)'],
                title="Random Forest Confusion Matrix",
                color_continuous_scale="Teal"
            )
            fig_cm2.update_layout(
                font_family="Inter",
                title_font_family="Montserrat",
                title_font_size=15,
                margin=dict(t=50, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_cm2, use_container_width=True)
            
        # Feature Importance Profile
        st.markdown("### ⚡ Predictor Ratios Importance Profile (Random Forest Model)")
        st.write("Feature Importance measures the contribution of each financial ratio to the decision-making process of the Ensemble Random Forest model.")
        
        importances = pipeline['feature_importances']
        fi_df = pd.DataFrame({
            'Ratio Name': RATIOS,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_fi = px.bar(
            fi_df, x='Importance', y='Ratio Name',
            orientation='h',
            title="Random Forest Feature Importance Analysis",
            color='Importance',
            color_continuous_scale=px.colors.sequential.Tealgrn
        )
        fig_fi.update_layout(
            font_family="Inter",
            title_font_family="Montserrat",
            title_font_size=15,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=40, l=40, r=40),
            height=500
        )
        fig_fi.update_xaxes(showgrid=True, gridcolor='#e2e8f0')
        st.plotly_chart(fig_fi, use_container_width=True)

    # ----------------- TAB 3: REAL-TIME CORPORATE DISTRESS PREDICTOR -----------------
    with tab3:
        st.markdown("### 🔮 Real-Time Corporate Distress Predictor")
        st.write("Enter financial ratios manually or load a real corporate profile from the CSE dataset below to generate an instant, scaled predictive evaluation using the trained Random Forest ensemble model.")

        st.markdown("#### 🏢 Option A: Load Profile from CSE Dataset")
        col_comp_sel, col_year_sel, col_btn_sel = st.columns([2, 1, 1])
        
        with col_comp_sel:
            comp_list = sorted(cleaned_df['Company Code'].unique())
            selected_comp = st.selectbox("Select Corporate Code (Ticker)", comp_list)
        
        with col_year_sel:
            year_list = sorted(cleaned_df[cleaned_df['Company Code'] == selected_comp]['Financial year'].unique())
            selected_yr = st.selectbox("Select Year", year_list)
            
        with col_btn_sel:
            st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔌 Load Corporate Profile", use_container_width=True):
                # Fetch row data
                profile_row = cleaned_df[
                    (cleaned_df['Company Code'] == selected_comp) & 
                    (cleaned_df['Financial year'] == selected_yr)
                ].iloc[0]
                
                # Load values to session state
                for r in RATIOS:
                    val = profile_row[r]
                    st.session_state[r] = float(val) if not pd.isna(val) else 0.0
                
                st.toast(f"Profile loaded successfully for {selected_comp} ({selected_yr})!", icon="✅")
                st.rerun()

        st.write("")
        st.markdown("#### 🖊️ Option B: Manually Adjust Ratios / Edit Input Form")
        
        # Form divided into logical categories
        with st.form("prediction_form"):
            col_p, col_l, col_c = st.columns(3)
            
            with col_p:
                st.markdown("##### 📈 Profitability & Leverage")
                roa_val = st.number_input("Return on Assets (ROA)", value=st.session_state['Return on Assets'], format="%.5f")
                opm_val = st.number_input("Operating Profit Margin (OPM)", value=st.session_state['Operating Profit Margin'], format="%.5f")
                roe_val = st.number_input("Return on Equity (ROE)", value=st.session_state['Return on Equity'], format="%.5f")
                lev_val = st.number_input("Leverage Ratio", value=st.session_state['Leverage'], format="%.5f")
                debt_val = st.number_input("Debt Ratio", value=st.session_state['Debt Ratio'], format="%.5f")

            with col_l:
                st.markdown("##### 💧 Liquidity & Efficiency")
                cr_val = st.number_input("Current Ratio (CR)", value=st.session_state['Current Ratio'], format="%.5f")
                qr_val = st.number_input("Quick Ratio (QR)", value=st.session_state['Quick Ratio'], format="%.5f")
                wcta_val = st.number_input("Working Capital to Total Assets", value=st.session_state['Working Capital to Total Assets'], format="%.5f")
                atr_val = st.number_input("Asset Turnover Ratio (ATR)", value=st.session_state['Asset Turnover Ratio'], format="%.5f")
                ctr_val = st.number_input("Capital Turnover Ratio (CTR)", value=st.session_state['Capital Turnover Ratio'], format="%.5f")

            with col_c:
                st.markdown("##### 💸 Operating Cash Flow Indicators")
                cffoni_val = st.number_input("Cash Flow to Net Income", value=st.session_state['Cash Flow to Net Income'], format="%.5f")
                cffotl_val = st.number_input("Cash Flow to Total Liabilities", value=st.session_state['Cash Flow to Total Liabilities'], format="%.5f")
                cffota_val = st.number_input("Cash Flow to Total Assets", value=st.session_state['Cash Flow to Total Assets'], format="%.5f")
                cffocl_val = st.number_input("Cash Flow to Current Liabilities", value=st.session_state['Cash Flow to Current Liabilities'], format="%.5f")
                cffotd_val = st.number_input("Cash Flow to Total Debt", value=st.session_state['Cash Flow to Total Debt'], format="%.5f")

            st.write("")
            submit_btn = st.form_submit_button("🔮 Predict Distress Risk Profile", use_container_width=True)

        if submit_btn:
            # Construct a row with the values matching the ratio keys exactly
            input_dict = {
                'Return on Assets': roa_val,
                'Operating Profit Margin': opm_val,
                'Return on Equity': roe_val,
                'Current Ratio': cr_val,
                'Quick Ratio': qr_val,
                'Working Capital to Total Assets': wcta_val,
                'Leverage': lev_val,
                'Debt Ratio': debt_val,
                'Asset Turnover Ratio': atr_val,
                'Capital Turnover Ratio': ctr_val,
                'Cash Flow to Net Income': cffoni_val,
                'Cash Flow to Total Liabilities': cffotl_val,
                'Cash Flow to Total Assets': cffota_val,
                'Cash Flow to Current Liabilities': cffocl_val,
                'Cash Flow to Total Debt': cffotd_val
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Impute and Scale using the fitted imputer and scaler from preprocessing
            input_imputed = pipeline['imputer'].transform(input_df)
            input_imputed_df = pd.DataFrame(input_imputed, columns=RATIOS)
            input_scaled = pipeline['scaler'].transform(input_imputed_df)
            
            # Predict with trained Random Forest model
            prediction = int(pipeline['rf_model'].predict(input_scaled)[0])
            prob_distress = float(pipeline['rf_model'].predict_proba(input_scaled)[0][1])
            
            # Results display columns
            col_res, col_gauge = st.columns([1.5, 1])
            
            with col_res:
                st.markdown("### 🏁 Model Evaluation Result")
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-distress">
                            <h2 style="color: #cf1322; margin:0; font-weight:700; font-family:'Montserrat',sans-serif;">⚠️ Financial Distress Risk Detected</h2>
                            <p style="color: #434343; font-size:1.02rem; margin:10px 0 0 0; line-height:1.5;">
                                The model has flagged this corporate financial structure as <b>Highly Vulnerable</b>. The calculated distress probability is <b>{prob_distress * 100:.1f}%</b>, which exceeds the classification threshold. Immediate financial restructuring and liquid preservation are recommended.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-healthy">
                            <h2 style="color: #389e0d; margin:0; font-weight:700; font-family:'Montserrat',sans-serif;">✅ Financially Healthy / Low Risk</h2>
                            <p style="color: #434343; font-size:1.02rem; margin:10px 0 0 0; line-height:1.5;">
                                The model evaluates this company as <b>Financially Healthy and Resilient</b>. The calculated distress probability is only <b>{prob_distress * 100:.1f}%</b>. The capital buffers, cash flows, and liquidity ratios are currently aligned with healthy firms.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("#### 📝 Input Ratio Summary")
                st.markdown(f"""
                    * **Profitability:** ROA: `{roa_val:.4%}` | OPM: `{opm_val:.4%}` | ROE: `{roe_val:.4%}`
                    * **Solvency/Leverage:** Debt Ratio: `{debt_val:.4%}` | Leverage: `{lev_val:.2f}`
                    * **Liquidity:** Current Ratio: `{cr_val:.2f}` | Quick Ratio: `{qr_val:.2f}`
                """)
                
            with col_gauge:
                # Plotly Gauge Chart for Distress Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_distress * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Distress Risk Probability (%)", 'font': {'size': 18, 'family': 'Montserrat'}},
                    number = {'suffix': "%", 'font': {'size': 36, 'family': 'Montserrat', 'color': '#0f172a'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#ef4444" if prediction == 1 else "#22c55e"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#cbd5e1",
                        'steps': [
                            {'range': [0, 30], 'color': '#f8fafc'},
                            {'range': [30, 70], 'color': '#f1f5f9'},
                            {'range': [70, 100], 'color': '#e2e8f0'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Inter",
                    height=280,
                    margin=dict(t=30, b=0, l=30, r=30)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.error("Failed to load dataset. Please check the file format or try uploading another one.")
