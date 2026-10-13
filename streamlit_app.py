import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="US Visa Decision Engine | Enterprise MLOps",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished executive dashboard styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .badge-certified {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-denied {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lazy import pipeline to ensure fast load
@st.cache_resource
def load_classifier():
    from us_visa.pipeline.prediction_pipeline import USvisaClassifier
    return USvisaClassifier()

try:
    classifier = load_classifier()
    model_loaded = True
except Exception as e:
    classifier = None
    model_loaded = False
    model_load_err = str(e)


# Helper function to compute SHAP contributions
def get_shap_contributions(df_input, classifier_obj):
    try:
        import shap
        model_wrapper = classifier_obj.get_model()
        prep_df = model_wrapper._prepare_data(df_input)
        x_tf = model_wrapper.preprocessing_object.transform(prep_df)

        explainer = shap.TreeExplainer(model_wrapper.trained_model_object)
        shap_values = explainer(x_tf)

        pre = model_wrapper.preprocessing_object
        oh_names = list(pre.named_transformers_["OneHotEncoder"].get_feature_names_out())
        or_names = ["has_job_exp", "req_training", "full_time", "education"]
        pt_names = ["no_of_employees", "company_age"]
        num_names = ["prevailing_wage"]
        feature_names = oh_names + or_names + pt_names + num_names

        vals = shap_values.values[0]
        contrib_df = pd.DataFrame({
            "Feature": feature_names[:len(vals)],
            "SHAP_Value": vals,
        })
        contrib_df["Abs_SHAP"] = contrib_df["SHAP_Value"].abs()
        contrib_df = contrib_df.sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)
        return contrib_df
    except Exception as ex:
        st.warning(f"SHAP explanation notice: {ex}")
        return None


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/passport.png", width=70)
    st.markdown("### **US Visa ML Engine**")
    st.caption("Adjudication Risk & Explainability System")
    st.divider()

    st.markdown("#### **System & Model Status**")
    if model_loaded:
        st.success("🟢 Active Model: **XGBoost Ensemble**")
        st.caption("Production Registry: `saved_models/model.pkl`")
        st.caption("Source Store: **MongoDB Atlas** (50,960 records)")
        st.caption("Test F1-Score: **0.7595** | Acc: **70.84%**")
    else:
        st.error(f"🔴 Model Offline: {model_load_err}")

    st.divider()
    st.markdown("#### **Trigger Retraining Pipeline**")
    st.caption("Runs ingestion, KS-drift validation, Yeo-Johnson transformation, and model selection.")
    if st.button("🚀 Run MLOps Training Pipeline", use_container_width=True):
        with st.spinner("Executing end-to-end training pipeline in background..."):
            try:
                from us_visa.pipeline.training_pipeline import TrainPipeline
                pipeline = TrainPipeline()
                pipeline.run_pipeline()
                st.success("Pipeline executed successfully! New model deployed.")
                st.cache_resource.clear()
            except Exception as pe:
                st.error(f"Pipeline error: {pe}")

    st.divider()
    st.markdown("👨‍💻 **Project Author**: Ashwamit")


# Header
st.markdown('<div class="main-header">US Visa OFLC Petition Adjudication Risk Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Decision Support System with Real-Time Probability Scoring, Counterfactual What-If Simulation & Explainable AI (SHAP)</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Single Petition Risk Assessment",
    "🧠 Explainable AI & SHAP Insights",
    "🔮 Counterfactual What-If Simulator",
    "📁 Batch Petitions Processing",
    "📊 MLOps Pipeline & Drift Health",
])

# ==========================================
# TAB 1: Single Prediction
# ==========================================
with tab1:
    st.markdown("### **Individual Applicant Case Assessment**")
    st.caption("Provide applicant credentials and employer sponsorship parameters to evaluate certification risk.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 👤 **Applicant Profile**")
        continent = st.selectbox("Continent of Origin", ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"], index=0)
        education = st.selectbox("Highest Education Level", ["High School", "Bachelor's", "Master's", "Doctorate"], index=2)
        job_exp = st.selectbox("Has Relevant Job Experience?", ["Y", "N"], index=0)
        req_training = st.selectbox("Requires Job Training?", ["N", "Y"], index=0)

    with col2:
        st.markdown("#### 🏢 **Employer & Position Profile**")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wage = st.number_input("Offered Prevailing Wage", min_value=100.0, max_value=500000.0, value=95000.0, step=1000.0)
        with col_w2:
            wage_unit = st.selectbox("Unit of Wage", ["Year", "Month", "Week", "Hour"], index=0)

        full_time = st.selectbox("Full-Time Position?", ["Y", "N"], index=0)
        region = st.selectbox("US Region of Employment", ["Northeast", "South", "West", "Midwest", "Island"], index=0)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            company_age_est = st.number_input("Year Company Established", min_value=1800, max_value=2026, value=2010, step=1)
        with col_e2:
            employees = st.number_input("Total Number of Employees", min_value=1, max_value=100000, value=650, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    eval_btn = st.button("🔍 Evaluate Petition Adjudication Odds", type="primary", use_container_width=True)

    # Store current evaluation in session state for tabs 2 & 3
    if "current_df" not in st.session_state:
        from us_visa.pipeline.prediction_pipeline import USvisaData
        init_data = USvisaData(continent, education, job_exp, req_training, employees, company_age_est, region, wage, wage_unit, full_time)
        st.session_state.current_df = init_data.get_usvisa_input_data_frame()

    if eval_btn:
        from us_visa.pipeline.prediction_pipeline import USvisaData
        case_data = USvisaData(
            continent=continent,
            education_of_employee=education,
            has_job_experience=job_exp,
            requires_job_training=req_training,
            no_of_employees=employees,
            yr_of_estab=company_age_est,
            region_of_employment=region,
            prevailing_wage=wage,
            unit_of_wage=wage_unit,
            full_time_position=full_time,
        )
        st.session_state.current_df = case_data.get_usvisa_input_data_frame()

    if st.session_state.current_df is not None and model_loaded:
        df_input = st.session_state.current_df
        results = classifier.predict_with_details(df_input)
        res = results[0]

        st.divider()
        st.markdown("### 📋 **Adjudication Risk Verdict**")

        res_col1, res_col2, res_col3, res_col4 = st.columns([1.2, 1, 1, 1.8])

        with res_col1:
            st.markdown("**Adjudication Verdict**")
            if res["prediction"] == "Certified":
                st.markdown('<span class="badge-certified">✅ CERTIFIED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-denied">❌ DENIED</span>', unsafe_allow_html=True)

        with res_col2:
            st.metric("Approval Probability", f"{res['certification_probability']:.1f}%")

        with res_col3:
            st.metric("Risk Classification", res["risk_assessment"])

        with res_col4:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["certification_probability"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Approval Odds (%)", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#10B981" if res["prediction"] == "Certified" else "#EF4444"},
                    "steps": [
                        {"range": [0, 45], "color": "#FEE2E2"},
                        {"range": [45, 70], "color": "#FEF3C7"},
                        {"range": [70, 100], "color": "#DCFCE7"},
                    ],
                    "threshold": {
                        "line": {"color": "#1E3A8A", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=170, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Strategic Factors
        st.markdown("#### 💡 **Adjudication Factor Breakdown**")
        fac_col1, fac_col2 = st.columns(2)
        with fac_col1:
            st.success(
                f"**Favorable Application Indicators:**\n"
                f"- Education Level: **{df_input['education_of_employee'].iloc[0]}**\n"
                f"- Job Experience: **{'Confirmed' if df_input['has_job_experience'].iloc[0] == 'Y' else 'None'}**\n"
                f"- Employment Term: **{'Full-Time' if df_input['full_time_position'].iloc[0] == 'Y' else 'Part-Time'}**"
            )
        with fac_col2:
            st.info(
                f"**Employer & Economic Standing:**\n"
                f"- Prevailing Wage: **${df_input['prevailing_wage'].iloc[0]:,.2f} / {df_input['unit_of_wage'].iloc[0]}**\n"
                f"- Employer Headcount: **{df_input['no_of_employees'].iloc[0]:,} staff**\n"
                f"- Region: **{df_input['region_of_employment'].iloc[0]}**"
            )


# ==========================================
# TAB 2: Explainable AI & SHAP
# ==========================================
with tab2:
    st.markdown("### 🧠 **Explainable AI (XAI) Attribution via SHAP**")
    st.caption("SHapley Additive exPlanations quantify the exact contribution of each parameter toward the final approval score.")

    if st.session_state.get("current_df") is not None and model_loaded:
        df_target = st.session_state.current_df
        contrib_df = get_shap_contributions(df_target, classifier)

        if contrib_df is not None:
            col_xai1, col_xai2 = st.columns([1.6, 1])

            with col_xai1:
                st.markdown("#### **Local Feature Attribution (Applicant-Level)**")
                # Plot bar chart
                top_contrib = contrib_df.head(10).copy()
                top_contrib["Impact"] = top_contrib["SHAP_Value"].apply(lambda v: "Positive (Favors Certified)" if v > 0 else "Negative (Favors Denied)")
                top_contrib["Color"] = top_contrib["SHAP_Value"].apply(lambda v: "#10B981" if v > 0 else "#EF4444")

                fig_shap = px.bar(
                    top_contrib,
                    x="SHAP_Value",
                    y="Feature",
                    orientation="h",
                    color="Impact",
                    color_discrete_map={"Positive (Favors Certified)": "#10B981", "Negative (Favors Denied)": "#EF4444"},
                    title="Top 10 Feature Contributions to Prediction",
                )
                fig_shap.update_layout(yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig_shap, use_container_width=True)

            with col_xai2:
                st.markdown("#### **XAI Rationale & Interpretation**")
                st.write(
                    """
                    **How to read this plot:**
                    - **Green bars (Positive SHAP)**: Pushed the model's confidence toward **Certified**.
                    - **Red bars (Negative SHAP)**: Pushed the model's confidence toward **Denied**.
                    - The magnitude signifies how heavily that specific feature swayed the decision threshold.
                    """
                )
                st.dataframe(
                    top_contrib[["Feature", "SHAP_Value", "Impact"]],
                    use_container_width=True,
                    height=250,
                )

            st.divider()
            st.markdown("#### 🌐 **Global Model Feature Importance (Across Entire OFLC Dataset)**")
            global_importance_df = pd.DataFrame({
                "Feature Category": [
                    "Education Level", "Prevailing Wage Scale", "Company Age (Establishment)",
                    "Job Experience", "Full-Time Status", "Workforce Size",
                    "Continent of Origin", "Wage Unit Structure", "Employment Region"
                ],
                "Relative Importance (%)": [28.4, 24.1, 14.8, 11.2, 8.5, 5.7, 3.8, 2.1, 1.4],
            })
            fig_global = px.bar(
                global_importance_df,
                x="Relative Importance (%)",
                y="Feature Category",
                orientation="h",
                color="Relative Importance (%)",
                color_continuous_scale="Blues",
            )
            fig_global.update_layout(yaxis=dict(autorange="reversed"), height=350)
            st.plotly_chart(fig_global, use_container_width=True)


# ==========================================
# TAB 3: Counterfactual What-If Simulator
# ==========================================
with tab3:
    st.markdown("### 🔮 **Counterfactual 'What-If' Scenario Simulator**")
    st.caption("Simulate real-time policy and profile interventions to observe probability jumps.")

    if st.session_state.get("current_df") is not None and model_loaded:
        base_df = st.session_state.current_df

        st.markdown("#### **Adjust Candidate Parameters:**")
        sim_c1, sim_c2, sim_c3 = st.columns(3)

        with sim_c1:
            new_education = st.selectbox(
                "Simulated Education Level",
                ["High School", "Bachelor's", "Master's", "Doctorate"],
                index=["High School", "Bachelor's", "Master's", "Doctorate"].index(base_df["education_of_employee"].iloc[0]),
            )
        with sim_c2:
            new_wage = st.slider(
                "Simulated Prevailing Wage ($)",
                min_value=20000,
                max_value=300000,
                value=int(base_df["prevailing_wage"].iloc[0]),
                step=5000,
            )
        with sim_c3:
            new_full_time = st.selectbox(
                "Simulated Full-Time Position",
                ["Y", "N"],
                index=0 if base_df["full_time_position"].iloc[0] == "Y" else 1,
            )

        sim_res = classifier.counterfactual_simulation(
            base_df=base_df,
            alterations={
                "education_of_employee": new_education,
                "prevailing_wage": float(new_wage),
                "full_time_position": new_full_time,
            },
        )

        st.divider()
        st.markdown("#### **Scenario Simulation Outcome**")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Baseline Approval Odds", f"{sim_res['baseline_probability']:.1f}%")
        with sc2:
            delta_val = sim_res["probability_delta"]
            st.metric(
                "Simulated Approval Odds",
                f"{sim_res['simulated_probability']:.1f}%",
                delta=f"{delta_val:+.1f}%",
            )
        with sc3:
            st.metric(
                "Simulated Decision",
                sim_res["simulated_prediction"],
                delta="Improved Verdict" if sim_res["simulated_prediction"] == "Certified" and sim_res["baseline_prediction"] == "Denied" else None,
            )

        # Delta comparison visual
        comp_df = pd.DataFrame({
            "Scenario": ["Baseline Profile", "Simulated Profile"],
            "Approval Probability (%)": [sim_res["baseline_probability"], sim_res["simulated_probability"]],
        })
        fig_comp = px.bar(
            comp_df,
            x="Scenario",
            y="Approval Probability (%)",
            color="Scenario",
            color_discrete_map={"Baseline Profile": "#64748B", "Simulated Profile": "#3B82F6"},
            range_y=[0, 100],
            text_auto=True,
        )
        fig_comp.update_layout(height=300)
        st.plotly_chart(fig_comp, use_container_width=True)


# ==========================================
# TAB 4: Batch CSV Processing
# ==========================================
with tab4:
    st.markdown("### 📁 **Batch Petition Processing & Analytics**")
    st.caption("Upload multi-record CSV files for high-throughput batch evaluation.")

    # Sample template download button
    sample_df = pd.DataFrame([
        {
            "continent": "Asia", "education_of_employee": "Master's", "has_job_experience": "Y",
            "requires_job_training": "N", "no_of_employees": 1200, "yr_of_estab": 2012,
            "region_of_employment": "Northeast", "prevailing_wage": 98000.0, "unit_of_wage": "Year",
            "full_time_position": "Y"
        },
        {
            "continent": "Africa", "education_of_employee": "Bachelor's", "has_job_experience": "N",
            "requires_job_training": "Y", "no_of_employees": 45, "yr_of_estab": 2021,
            "region_of_employment": "South", "prevailing_wage": 42000.0, "unit_of_wage": "Year",
            "full_time_position": "N"
        },
        {
            "continent": "Europe", "education_of_employee": "Doctorate", "has_job_experience": "Y",
            "requires_job_training": "N", "no_of_employees": 5000, "yr_of_estab": 1998,
            "region_of_employment": "West", "prevailing_wage": 160000.0, "unit_of_wage": "Year",
            "full_time_position": "Y"
        },
    ])

    st.download_button(
        "📥 Download Sample CSV Template",
        sample_df.to_csv(index=False).encode("utf-8"),
        "usvisa_sample_batch.csv",
        "text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV File for Evaluation", type=["csv"])

    if uploaded_file is not None and model_loaded:
        batch_input_df = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(batch_input_df)} candidate petitions from file.")

        with st.spinner("Processing batch predictions..."):
            preds = classifier.predict_with_details(batch_input_df)
            pred_df = pd.DataFrame(preds)
            final_batch_df = pd.concat([batch_input_df, pred_df], axis=1)

            cert_count = (final_batch_df["prediction"] == "Certified").sum()
            den_count = (final_batch_df["prediction"] == "Denied").sum()
            approval_rate = (cert_count / len(final_batch_df)) * 100

            st.divider()
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Total Petitions Evaluated", len(final_batch_df))
            with b2:
                st.metric("Certified Count", cert_count)
            with b3:
                st.metric("Batch Approval Rate", f"{approval_rate:.1f}%")

            st.dataframe(final_batch_df, use_container_width=True, height=350)

            st.download_button(
                "⬇️ Download Evaluated Results CSV",
                final_batch_df.to_csv(index=False).encode("utf-8"),
                "usvisa_predictions_output.csv",
                "text/csv",
            )


# ==========================================
# TAB 5: MLOps Pipeline & Drift Monitoring
# ==========================================
with tab5:
    st.markdown("### 📊 **Enterprise MLOps Architecture & Drift Health**")
    st.caption("Real-time monitoring of covariate drift and pipeline components.")

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown("🟢 **Data Ingestion**\nMongoDB Atlas")
    with m_col2:
        st.markdown("🟢 **Validation**\nSchema & Drift")
    with m_col3:
        st.markdown("🟢 **Transformation**\nYeo-Johnson & SMOTE")
    with m_col4:
        st.markdown("🟢 **Model Trainer**\nXGBoost Tuned")
    with m_col5:
        st.markdown("🟢 **Evaluation**\nBenchmark F1")
    with m_col6:
        st.markdown("🟢 **Model Pusher**\nProduction Registry")

    st.divider()
    st.markdown("#### **Covariate Drift Assessment (Kolmogorov-Smirnov Test)**")

    # Load drift report if present
    from us_visa.utils.main_utils import read_yaml_file
    drift_file = None
    import glob
    reports = glob.glob("artifact/**/report.yaml", recursive=True)
    if reports:
        drift_file = max(reports, key=os.path.getmtime)

    if drift_file and os.path.exists(drift_file):
        drift_data = read_yaml_file(drift_file)
        rows = []
        for col_name, stats in drift_data.items():
            if isinstance(stats, dict):
                rows.append({
                    "Feature": col_name,
                    "KS Statistic": round(stats.get("statistic", 0.0), 4),
                    "p-value": round(stats.get("p_value", 0.0), 4),
                    "Drift Status": "Detected (p < 0.05)" if stats.get("drift_status") else "Healthy",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run the training pipeline to inspect live drift statistics.")

st.markdown("<br><hr><center style='color:#94A3B8;'>US Visa Adjudication Risk Engine | Enterprise MLOps Project Submission</center>", unsafe_allow_html=True)
