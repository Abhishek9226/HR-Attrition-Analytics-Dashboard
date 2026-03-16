import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# ==============================
# CUSTOM UI STYLING
# ==============================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: radial-gradient(circle at top,#0f172a,#020617 70%);
color:white;
}

[data-testid="stSidebar"]{
background: rgba(255,255,255,0.05);
backdrop-filter: blur(10px);
border-right:1px solid rgba(255,255,255,0.1);
}

.kpi-card{
background: linear-gradient(135deg,#111827,#1f2937);
padding:25px;
border-radius:12px;
text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.6);
transition:0.3s;
}

.kpi-card:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================

st.markdown("""
<h1 style="text-align:center;font-size:52px;font-weight:800;margin-bottom:10px;">
🚀 Workforce Attrition Intelligence Platform
</h1>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================

df_full = pd.read_csv("data/Palo Alto Networks.csv")

# ==============================
# MACHINE LEARNING MODEL
# ==============================

ml_df = pd.get_dummies(df_full, drop_first=True)

X = ml_df.drop("Attrition", axis=1)
y = ml_df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ==============================
# SIDEBAR FILTERS
# ==============================

st.sidebar.markdown("## 📊 HR Dashboard Filters")
st.sidebar.divider()

df = df_full.copy()

department_filter = st.sidebar.multiselect(
"Department",
options=df["Department"].unique(),
default=df["Department"].unique()
)

gender_filter = st.sidebar.multiselect(
"Gender",
options=df["Gender"].unique(),
default=df["Gender"].unique()
)

overtime_filter = st.sidebar.multiselect(
"Overtime",
options=df["OverTime"].unique(),
default=df["OverTime"].unique()
)

df = df[
(df["Department"].isin(department_filter)) &
(df["Gender"].isin(gender_filter)) &
(df["OverTime"].isin(overtime_filter))
]

# ==============================
# WORKFORCE OVERVIEW
# ==============================

st.markdown("""
<h3 style='background:linear-gradient(90deg,#38bdf8,#22c55e);
padding:10px;border-radius:10px;color:white;'>
📊 Workforce Overview
</h3>
""", unsafe_allow_html=True)

total_employees = df.shape[0]
employees_left = df["Attrition"].sum()
attrition_rate = (employees_left / total_employees) * 100
avg_income = df["MonthlyIncome"].mean()

col1,col2,col3,col4 = st.columns(4)

col1.markdown(f'<div class="kpi-card"><h4>Total Employees</h4><h2>{total_employees}</h2></div>',unsafe_allow_html=True)
col2.markdown(f'<div class="kpi-card"><h4>Employees Left</h4><h2>{employees_left}</h2></div>',unsafe_allow_html=True)
col3.markdown(f'<div class="kpi-card"><h4>Attrition Rate</h4><h2>{attrition_rate:.2f}%</h2></div>',unsafe_allow_html=True)
col4.markdown(f'<div class="kpi-card"><h4>Avg Monthly Income</h4><h2>${avg_income:,.0f}</h2></div>',unsafe_allow_html=True)

# ==============================
# ATTRITION GAUGE
# ==============================

st.markdown("""
<h3 style='background:linear-gradient(90deg,#38bdf8,#22c55e);
padding:10px;border-radius:10px;color:white;'>
📉 Overall Attrition Risk
</h3>
""", unsafe_allow_html=True)

fig_gauge = go.Figure(go.Indicator(
mode="gauge+number",
value=attrition_rate,
title={'text':"Attrition Risk (%)"},
gauge={
'axis':{'range':[0,50]},
'bar':{'color':"red"},
'steps':[
{'range':[0,10],'color':"#2ecc71"},
{'range':[10,20],'color':"#f1c40f"},
{'range':[20,50],'color':"#e74c3c"}
]
}
))

st.plotly_chart(fig_gauge,use_container_width=True)

# ==============================
# DATA PREVIEW
# ==============================

st.dataframe(df.head())

st.divider()

# ==============================
# DEPARTMENT ATTRITION
# ==============================

dept_attrition = df.groupby("Department")["Attrition"].mean().reset_index()
dept_attrition["Attrition"] *= 100

fig_dept = px.bar(
dept_attrition,
x="Department",
y="Attrition",
color="Department",
template="plotly_dark",
title="Attrition Rate by Department"
)

# ==============================
# JOB ROLE ATTRITION
# ==============================

role_attrition = df.groupby("JobRole")["Attrition"].mean().reset_index()
role_attrition["Attrition"] *= 100

fig_role = px.bar(
role_attrition,
x="Attrition",
y="JobRole",
orientation="h",
color="Attrition",
template="plotly_dark",
title="Attrition Rate by Job Role"
)

col1,col2 = st.columns(2)

col1.plotly_chart(fig_dept,use_container_width=True)
col2.plotly_chart(fig_role,use_container_width=True)

st.divider()

# ==============================
# TENURE ANALYSIS
# ==============================

bins=[0,3,7,12,20,40]
labels=["0-3 Years","4-7 Years","8-12 Years","13-20 Years","20+ Years"]

df["TenureGroup"]=pd.cut(df["YearsAtCompany"],bins=bins,labels=labels)

tenure_attrition = df.groupby("TenureGroup")["Attrition"].mean().reset_index()
tenure_attrition["Attrition"] *= 100

fig_tenure = px.bar(
tenure_attrition,
x="TenureGroup",
y="Attrition",
color="TenureGroup",
template="plotly_dark",
title="Attrition by Tenure"
)

# ==============================
# OVERTIME ANALYSIS
# ==============================

overtime_attrition = df.groupby("OverTime")["Attrition"].mean().reset_index()
overtime_attrition["Attrition"] *= 100

fig_overtime = px.bar(
overtime_attrition,
x="OverTime",
y="Attrition",
color="OverTime",
template="plotly_dark",
title="Attrition by Overtime"
)

col3,col4 = st.columns(2)

col3.plotly_chart(fig_tenure,use_container_width=True)
col4.plotly_chart(fig_overtime,use_container_width=True)

st.divider()

# ==============================
# BUSINESS TRAVEL
# ==============================

travel_attrition = df.groupby("BusinessTravel")["Attrition"].mean().reset_index()
travel_attrition["Attrition"] *= 100

fig_travel = px.bar(
travel_attrition,
x="BusinessTravel",
y="Attrition",
color="BusinessTravel",
template="plotly_dark",
title="Attrition by Business Travel"
)

st.plotly_chart(fig_travel,use_container_width=True)

# ==============================
# CORRELATION HEATMAP
# ==============================

st.markdown("""
<h3 style='background:linear-gradient(90deg,#38bdf8,#22c55e);
padding:10px;border-radius:10px;color:white;'>
📊 Employee Factors Correlation
</h3>
""", unsafe_allow_html=True)

numeric_df = df.select_dtypes(include=["int64","float64"])

fig_heatmap = px.imshow(
numeric_df.corr(),
text_auto=True,
aspect="auto",
color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig_heatmap,use_container_width=True)

# ==============================
# AGE DISTRIBUTION
# ==============================

st.markdown("""
<h3 style='background:linear-gradient(90deg,#38bdf8,#22c55e);
padding:10px;border-radius:10px;color:white;'>
👥 Employee Age Distribution
</h3>
""", unsafe_allow_html=True)

fig_age = px.histogram(
df,
x="Age",
nbins=20,
color="Attrition",
template="plotly_dark"
)

st.plotly_chart(fig_age,use_container_width=True)

# ==============================
# FEATURE IMPORTANCE
# ==============================

st.markdown("""
<h3 style='background:linear-gradient(90deg,#38bdf8,#22c55e);
padding:10px;border-radius:10px;color:white;'>
⭐ Top Factors Driving Attrition
</h3>
""", unsafe_allow_html=True)

importances = model.feature_importances_

feature_importance = pd.DataFrame({
"Feature":X.columns,
"Importance":importances
}).sort_values("Importance",ascending=False).head(10)

fig_imp = px.bar(
feature_importance,
x="Importance",
y="Feature",
orientation="h",
template="plotly_dark"
)

st.plotly_chart(fig_imp,use_container_width=True)

# ==============================
# ATTRITION PREDICTOR
# ==============================

st.markdown("""
<h3 style='background:linear-gradient(90deg,#38bdf8,#22c55e);
padding:10px;border-radius:10px;color:white;'>
🤖 Predict Employee Attrition Risk
</h3>
""", unsafe_allow_html=True)

age = st.slider("Age",18,60,30)
years = st.slider("Years At Company",0,40,5)
income = st.slider("Monthly Income",1000,20000,5000)

input_df = pd.DataFrame({
"Age":[age],
"YearsAtCompany":[years],
"MonthlyIncome":[income]
})

input_df = input_df.reindex(columns=X.columns,fill_value=0)

prediction = model.predict(input_df)[0]

if st.button("Predict Attrition Risk"):

    if prediction==1:
        st.error("⚠️ High Risk of Attrition")
    else:
        st.success("✅ Low Risk of Attrition")

# ==============================
# FOOTER
# ==============================

st.divider()

st.markdown("""
<div style="text-align:center;padding:25px;background:rgba(255,255,255,0.05);
border-radius:12px;margin-top:40px;">

### 📊 Workforce Attrition Analytics Dashboard

Built with  
⚡ Python | 📊 Streamlit | 📈 Plotly | 🤖 Machine Learning | 🎨 Matplotlib

💡 Designed to help HR teams identify high-risk employees and improve retention strategies.

© 2026 HR Analytics Project

</div>
""",unsafe_allow_html=True)