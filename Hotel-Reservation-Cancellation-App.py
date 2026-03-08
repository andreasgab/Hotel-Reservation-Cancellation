import pandas as pd
import streamlit as st
import pickle
import joblib
import os
import warnings
import plotly.express as px
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Hotel Cancellation Prediction")

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Prediction Strategy:",
    ("Balanced (Random Forest)", "High-Recall (CatBoost)")
)

pickle_folder = "pickle files"

dummies = joblib.load(os.path.join(pickle_folder, 'dummy_cols.pickle'))

if model_choice == "Balanced (Random Forest)":
    # Loading the RF specific parts
    model = joblib.load(os.path.join(pickle_folder, 'balanced_randomforest_final_model.pickle'))
    imputer = joblib.load(os.path.join(pickle_folder, 'balanced_randomforest_imputer.pickle'))
    scaler = joblib.load(os.path.join(pickle_folder, 'balanced_randomforest_scaler.pickle'))
    selector = joblib.load(os.path.join(pickle_folder, 'balanced_randomforest_feature_selector.pickle'))
    model_name = "Random Forest"
else:
    # Loading the CatBoost specific parts
    model = joblib.load(os.path.join(pickle_folder, 'high_recall_catboost_final_model.pickle'))
    imputer = joblib.load(os.path.join(pickle_folder, 'high_recall_catboost_imputer.pickle'))
    scaler = joblib.load(os.path.join(pickle_folder, 'high_recall_catboost_scaler.pickle'))
    selector = joblib.load(os.path.join(pickle_folder, 'high_recall_catboost_feature_selector.pickle'))
    model_name = "CatBoost"

# Get the feature names used by the selector (the 30 best columns)
# Note: This assumes X_train/X_full columns match the 'dummies' list order
fis = [dummies[i] for i in selector.get_support(indices=True)]

# ---------------- ENCODING FUNCTION ----------------
def smart_encode_hotels(df):
    hotels_encoded = df.copy()
    cat_cols = hotels_encoded.select_dtypes(include=['object']).columns
    for col in cat_cols:
        n_unique = hotels_encoded[col].nunique()
        if n_unique <= 2:
            hotels_encoded[col] = pd.factorize(hotels_encoded[col])[0]
        elif 2 < n_unique <= 6:
            hotels_encoded = pd.get_dummies(hotels_encoded, columns=[col], drop_first=True)
        else:
            top_6 = hotels_encoded[col].value_counts().nlargest(6).index
            hotels_encoded[col] = hotels_encoded[col].where(hotels_encoded[col].isin(top_6), 'Other')
            hotels_encoded = pd.get_dummies(hotels_encoded, columns=[col], drop_first=True)
    return hotels_encoded


uploaded_file = st.sidebar.file_uploader("Upload Hotels CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded using {model_name}")

    data_clean = data.drop(['company','arrival_date_year','arrival_date_month',
                            'arrival_date_week_number','arrival_date_day_of_month',
                            'reservation_status_date'], axis=1, errors='ignore')

    data_encoded = smart_encode_hotels(data_clean)
    data_final = data_encoded.drop(columns=['reservation_status_Check-Out','is_canceled'], errors='ignore')

    # Ensure columns match training dummy set
    data_final = data_final.reindex(columns=dummies, fill_value=0)

    # Transform through pipeline steps
    data_imputed = imputer.transform(data_final)
    data_scaled = scaler.transform(data_imputed)
    
    # Feature Selection (SelectKBest)
    data_selected = selector.transform(data_scaled)

    # ---------------- PREDICTION ----------------
    predictions = model.predict(data_selected)
    
    # Flatten predictions in case they are returned as a 2D array (common in CatBoost)
    if hasattr(predictions, "flatten"):
        predictions = predictions.flatten()

    cancellation_predictions = pd.Series(predictions).map({0: "No", 1: "Yes"})

    results = pd.DataFrame({
        "Cancellation Prediction": cancellation_predictions
    })

    filter_choice = st.sidebar.radio("Show bookings:", ("All", "Yes", "No"))
    if filter_choice != "All":
        results = results[results["Cancellation Prediction"] == filter_choice]

    st.subheader(f"Prediction Results ({model_name})")
    st.write(results)

    # -------- BARPLOT --------
    counts = cancellation_predictions.value_counts().reset_index()
    counts.columns = ['Cancel', 'Count']
    fig = px.bar(counts, x='Cancel', y='Count', text='Count', color='Cancel',
                 title='Cancellation Distribution',
                 color_discrete_map={'No': 'green', 'Yes': 'red'})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if 'is_canceled' in data.columns:
        st.subheader(f"Model Performance: {model_name}")
        actual_status = data['is_canceled'].map({0: "No", 1: "Yes"})
        
        f1_macro = f1_score(actual_status, cancellation_predictions, average='macro', pos_label="Yes")
        correct = (actual_status == cancellation_predictions).sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Bookings", len(data))
        m2.metric("Correct Predictions", correct)
        m3.metric("F1 Macro Score", f"{f1_macro:.2f}")

        col_eval1, col_eval2 = st.columns([1.2, 1])
        with col_eval1:
            cm_data = pd.crosstab(actual_status, cancellation_predictions, 
                                 colnames=['Predicted'], rownames=['Actual'])
            fig_cm = px.imshow(cm_data, text_auto=True, color_continuous_scale='Greens',
                              title="Confusion Matrix", aspect="auto")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_eval2:
            st.write("### Classification Report")
            report_dict = classification_report(actual_status, cancellation_predictions, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.iloc[:2, :3].style.format("{:.2f}"), use_container_width=True)
    
    st.divider()
    st.subheader(f"Key Factors ({model_name})")

    importances = model.feature_importances_
    
    feat_imp = pd.DataFrame({
        'Feature': fis, 
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                     title=f'Top {len(fis)} Features Influence',
                     color='Importance', color_continuous_scale='Reds')
    fig_imp.update_layout(height=800, margin=dict(l=200))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("Contextual Analytics")
    data['pred_status'] = cancellation_predictions
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(px.histogram(data, x='lead_time', color='pred_status', barmode='group',
                                     title='Lead Time Distribution', color_discrete_map={"No": "green", "Yes": "red"}), 
                        use_container_width=True)
    with r1c2:
        st.plotly_chart(px.histogram(data, x='total_of_special_requests', color='pred_status', barmode='group',
                                     title='Special Requests vs Prediction', color_discrete_map={"No": "green", "Yes": "red"}), 
                        use_container_width=True)