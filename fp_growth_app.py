import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import plotly.express as px
import time

st.title("FP-Growth vs Apriori with User Dataset Upload")

# Sidebar parameters
algo = st.sidebar.selectbox("Select Algorithm", ["FP-Growth", "Apriori"])
min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.3)
chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Treemap"])

# File uploader to accept CSV files
uploaded_file = st.file_uploader("Upload your dataset CSV file (each row is a transaction)")

def preprocess_data(data):
    # Clean transactions by removing empty or NaN items and converting to strings
    clean_data = []
    for transaction in data:
        clean_transaction = [str(item) for item in transaction if pd.notna(item) and str(item).strip() != '']
        clean_data.append(clean_transaction)

    all_items = sorted(set(i for transaction in clean_data for i in transaction))
    df = pd.DataFrame(0, index=range(len(clean_data)), columns=all_items)
    for idx, transaction in enumerate(clean_data):
        df.loc[idx, transaction] = 1
    return df

if uploaded_file:
    # Read user dataset
    user_df = pd.read_csv(uploaded_file, header=None)
    # Convert DataFrame of transactions to list of lists
    transactions = user_df.values.tolist()
    # Preprocess user_df to one-hot encoded DataFrame
    df = preprocess_data(transactions)
    st.write("Preview of one-hot encoded user data")
    st.dataframe(df.head())
else:
    # Default dataset (modify or extend as needed)
    default_data = [
        ["Rice", "Cooking Oil", "Salt", "Sugar", "Pulses"],
        ["Rice", "Salt", "Flour", "Cooking Oil"],
        ["Tea", "Sugar", "Milk", "Bread"],
        # add more sample transactions as needed
    ]
    df = preprocess_data(default_data)
    st.write("Using default sample dataset")

start_time = time.time()
if algo == "FP-Growth":
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
else:
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
end_time = time.time()

st.write(f"Execution time for {algo}: {end_time - start_time:.4f} seconds")

if frequent_itemsets.empty:
    st.write("No frequent itemsets found with the given support.")
else:
    frequent_itemsets['support_rounded'] = frequent_itemsets['support'].round(3)
    frequent_itemsets['itemset_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

    st.write(f"## Frequent Itemsets - {algo}")
    st.dataframe(frequent_itemsets[['itemset_str', 'support_rounded']].sort_values(by='support_rounded', ascending=False))

    if chart_type == "Bar Chart":
        fig = px.bar(frequent_itemsets.sort_values(by='support_rounded', ascending=False).head(10),
                     x='itemset_str', y='support_rounded',
                     labels={'itemset_str': 'Itemset', 'support_rounded': 'Support'},
                     title='Top 10 Frequent Itemsets')
    elif chart_type == "Pie Chart":
        fig = px.pie(frequent_itemsets.sort_values(by='support_rounded', ascending=False).head(10),
                     values='support_rounded', names='itemset_str',
                     title='Support Distribution of Top 10 Itemsets')
    else:  # Treemap
        fig = px.treemap(frequent_itemsets,
                         path=[frequent_itemsets['itemset_str']],
                         values='support_rounded',
                         color='support_rounded',
                         color_continuous_scale='blues',
                         title='Treemap of Frequent Itemsets')
    st.plotly_chart(fig)

    st.write(f"## Association Rules - {algo}")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    if rules.empty:
        st.write("No association rules found with the given threshold.")
    else:
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].sort_values(by='confidence', ascending=False))
