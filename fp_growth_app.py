import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px

# Sample dataset
dataset = [
    ["Rice", "Cooking Oil", "Salt", "Sugar", "Pulses"],
    ["Rice", "Salt", "Flour", "Cooking Oil"],
    ["Tea", "Sugar", "Milk", "Bread"],
    ["Coffee", "Sugar", "Milk"],
    ["Rice", "Spices", "Pulses", "Cooking Oil"],
    ["Flour", "Sugar", "Salt", "Yeast"],
    ["Bread", "Butter", "Eggs", "Milk"],
    ["Rice", "Cooking Oil", "Salt"],
    ["Tea", "Milk", "Sugar"],
    ["Coffee", "Sugar", "Cookies"],
    ["Pulses", "Spices", "Salt"],
    ["Flour", "Yeast", "Salt"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Sugar", "Bread"],
]


# --- Sidebar ---
st.sidebar.title("FP-Growth Settings")
min_support = st.sidebar.slider('Minimum Support (fraction)', 0.2, 1.0, 0.6, 0.05)
st.sidebar.markdown("### Customize chart type below")
chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Pie", "Treemap"])

# --- FP-Growth Processing ---
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(dataset), columns=te.columns_)
result = fpgrowth(df, min_support=min_support, use_colnames=True)
result["Itemset"] = result["itemsets"].apply(lambda x: ', '.join(list(x)))

# --- Main Content ---
st.title("Advanced FP-Growth Visualization Dashboard")
st.markdown("""<style>
    .main {background-color: #1F2630; color: #F6F6F6;}
    .st-bb {background: #0E1117;}
    .css-ffhzg2 {color: #F6F6F6;}
    </style>""", unsafe_allow_html=True)
st.dataframe(result.style.highlight_max(axis=0, color='#E74C3C'))

# --- Chart Visualization ---
if not result.empty:
    if chart_type == "Bar":
        fig = px.bar(result, x="Itemset", y="support", color="support",
                     labels={"support": "Support"}, title="Frequent Itemsets - Bar Chart")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Pie":
        fig = px.pie(result, names="Itemset", values="support",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     title="Frequent Itemsets - Pie Chart")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Treemap":
        fig = px.treemap(result, path=["Itemset"], values="support",
                         color="support", title="Frequent Itemsets - Treemap")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No frequent itemsets found for this support. Try lowering the support threshold.")

# --- Footer ---
st.caption("Created with Streamlit & Plotly for interactive FP-Growth data mining.")

