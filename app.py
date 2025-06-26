import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Page configuration and styling
st.set_page_config(page_title="Customer Segmentation and Marketing App", layout="wide")
sns.set_style('whitegrid')

# Custom button styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45A049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv("clustered_customers.csv")
    return df

data = load_data()

@st.cache_resource
def generate_figures(data):
    figs = {}
    palette = sns.color_palette("deep")
    sizes = data["Cluster"].value_counts().to_dict()

    # 1) Cluster Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=data["Cluster"], palette=palette, ax=ax)
    ax.set_title("Cluster Distribution")
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Number of Customers")
    figs["Cluster Distribution"] = fig

    # 2) Income vs Spending
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = data["Cluster"].map(lambda x: palette[x])
    ax.scatter(data["Income"], data["Money_Spent"], c=colors, alpha=0.7)
    ax.set_title("Income vs Total Spending")
    ax.set_xlabel("Annual Income (€)")
    ax.set_ylabel("Total Spend (€)")
    figs["Income vs Spending"] = fig

    # 3) Promotions Accepted
    df_promo = data.copy()
    df_promo["Total_Promos"] = df_promo[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)
    prop_df = df_promo.groupby(["Cluster","Total_Promos"]).size().reset_index(name="Count")
    prop_df["Proportion"] = prop_df.apply(lambda r: r["Count"]/sizes[r["Cluster"]], axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=prop_df, x="Total_Promos", y="Proportion", hue="Cluster", palette=palette, ax=ax)
    ax.set_title("Promotions Accepted by Cluster")
    ax.set_xlabel("Number of Promotions")
    ax.set_ylabel("Proportion of Cluster")
    figs["Campaigns Accepted"] = fig

    # 4) Deals Purchased
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxenplot(x=data["Cluster"], y=data["NumDealsPurchases"], palette=palette, ax=ax)
    ax.set_title("Deals Purchased by Cluster")
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Number of Deals")
    figs["Deals Purchased"] = fig

    # Customer Profiling: continuous variables
    for var in ["Days","Age","Family_Size"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=data["Cluster"], y=data[var], palette=palette, ax=ax)
        ax.set_title(f"{var} by Cluster")
        ax.set_xlabel("Cluster Label")
        ax.set_ylabel(var.replace('_',' '))
        figs[f"{var} by Cluster"] = fig

    # Customer Profiling: discrete variables
    for var in ["Kidhome","Teenhome"]:
        df_var = data.groupby(["Cluster",var]).size().reset_index(name="Count")
        df_var["Proportion"] = df_var.apply(lambda r: r["Count"]/sizes[r["Cluster"]], axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_var, x=var, y="Proportion", hue="Cluster", palette=palette, ax=ax)
        ax.set_title(f"{var} by Cluster (Proportional)")
        ax.set_xlabel(var)
        ax.set_ylabel("Proportion")
        figs[f"{var} by Cluster"] = fig

    # Customer Profiling: education levels
    edu_cols = ["Education_Basic","Education_Graduation","Education_Master","Education_PhD"]
    edu_long = data[["Cluster"]+edu_cols].melt(id_vars="Cluster", var_name="Education", value_name="Flag")
    edu_long = edu_long[edu_long.Flag==1]
    edu_prop = edu_long.groupby(["Cluster","Education"]).size().reset_index(name="Count")
    edu_prop["Proportion"] = edu_prop.apply(lambda r: r["Count"]/sizes[r["Cluster"]], axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=edu_prop, x="Education", y="Proportion", hue="Cluster", palette=palette, ax=ax)
    ax.set_title("Education Level by Cluster")
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Proportion")
    fig.autofmt_xdate(rotation=15)
    figs["Education by Cluster"] = fig

    return figs

figs = generate_figures(data)

# Sidebar inputs (outside form for individual controls)
st.sidebar.header("Hypothetical Customer Inputs")
income = st.sidebar.number_input("Income (€)", 0, 150000, 30000, 1000)
kidhome = st.sidebar.slider("Number of Kids", 0, 5, 0)
teenh = st.sidebar.slider("Number of Teenagers", 0, 5, 0)
recency = st.sidebar.slider("Days Since Last Purchase", 0, 100, 20)
deals = st.sidebar.slider("Deals Purchased", 0, 20, 1)
days = st.sidebar.slider("Days as Customer", 0, 3000, 1000)
age = st.sidebar.slider("Age", 18, 100, 35)
family = st.sidebar.slider("Family Size", 1, 10, 2)
spent = st.sidebar.number_input("Total Spend (€)", 0, 20000, 1000)

# Initialize and store KNN model once
if 'knn_model' not in st.session_state:
    features = ["Income","Kidhome","Teenhome","Recency","NumDealsPurchases","Days","Age","Family_Size","Money_Spent"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_scaled, data["Cluster"])
    st.session_state.knn_scaler = scaler
    st.session_state.knn_model = knn_model

# App title and description
st.title("Customer Segmentation and Marketing App")
st.markdown(
    "Use the tabs above to explore cluster insights, demographic profiles, or predict a hypothetical customer’s cluster and marketing strategy."
)

# Top-level tabs
tab1, tab2, tab3 = st.tabs(["Clustering Overview","Customer Profiling","Marketing Strategy"])

# 1) Clustering Overview
a = ["Cluster Distribution","Income vs Spending","Campaigns Accepted","Deals Purchased"]
with tab1:
    st.subheader("Clustering Overview")
    overview_tabs = st.tabs(a)
    descriptions = {
        "Cluster Distribution": "Shows the size of each customer segment, helping you see which clusters are most prevalent. \n Clusters 0 and 2 are larger than cluster 1",
        "Income vs Spending": "Scatterplot of income vs total spend, color-coded by cluster to reveal spending patterns relative to earnings. \n Here we can see that cluster 1 are those we really wish to target, with higher income and spending than the other two clusters.",
        "Campaigns Accepted": "Proportion of customers in each cluster accepting various numbers of promotions; identifies promo receptiveness. \n Here we can see that cluster 1 seems to accept more promotions than the other clusters.",
        "Deals Purchased": "Distribution of deals purchased by cluster, highlighting deal sensitivity across segments. \n Despite accepting more promotions, cluster 1 does not purchase as many deals as the other two clusters."
    }
    for tab, key in zip(overview_tabs, a):
        with tab:
            st.pyplot(figs[key])
            st.markdown(descriptions[key])

# 2) Customer Profiling
b = ["Days by Cluster","Age by Cluster","Family_Size by Cluster","Kidhome by Cluster","Teenhome by Cluster","Education by Cluster"]
with tab2:
    st.subheader("Customer Profiling")
    profiling_tabs = st.tabs(b)
    descriptions = {
        "Days by Cluster": "Tenure distribution (days since first purchase) by cluster, showing how long each cluster has been customers for on average. \n Here we can see that there is not a great difference between clusters.",
        "Age by Cluster": "Age distribution within each cluster, revealing demographic age profiles. \n Here we can see that clusters 0 and 1 are slightly older on average than cluster 2.",
        "Family_Size by Cluster": "Household size distribution, indicating whether clusters are singles or larger families. \n Here we can see that cluster 1 typically does not have children, while clusters 0 and 2 do. Cluster 0 represents older parents, while cluster 2 are those with younger families.",
        "Kidhome by Cluster": "Proportion of customers with kids per cluster, highlighting family stages.",
        "Teenhome by Cluster": "Proportion of customers with teenagers per cluster, showing older child demographics.",
        "Education by Cluster": "Breakdown of education levels by cluster, indicating academic attainment of segments. \n Clusters 0 and 1 are slightly more likely to have completed higher education, such as a degree, masters or PhD than cluster 2."
    }
    for tab, key in zip(profiling_tabs, b):
        with tab:
            st.pyplot(figs[key])
            st.markdown(descriptions[key])

# 3) Marketing Strategy with KNN
with tab3:
    st.subheader("Marketing Strategy")
    st.markdown("Adjust the inputs in the sidebar, then click the button below to predict cluster and view a tailored strategy.")

    if st.button("Predict Cluster"):
        with st.spinner("Calculating cluster..."):
            scaler = st.session_state.knn_scaler
            model = st.session_state.knn_model
            user_vec = np.array([income, kidhome, teenh, recency, deals, days, age, family, spent]).reshape(1, -1)
            pred = model.predict(scaler.transform(user_vec))[0]
        st.success(f"Predicted Cluster: {pred}")

        # Tailored marketing recommendations
        strategies = {
            0: "High-value spenders: Make deals for high value products such as jewellery and wine",
            1: "High-income and High-value spenders, but do not buy luxury items: Focus on promotions for medium-priced products, loyalty rewards, and exclusive events.",
            2: "Younger parents with kids: Target with family-oriented promotions, kids' products, and educational content.",
            }
        text = strategies.get(pred, "General strategy: Use a mix of promotional offers and personalized outreach.")
        st.markdown(f"**Recommended Marketing Strategy:** {text}")
    else:
        st.info("Click 'Predict Cluster' above once you’ve set the inputs.")

# Footer spacing
st.markdown("---")
st.caption("App built with Streamlit | Louis Chislett")
