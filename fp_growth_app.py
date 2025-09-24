import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import plotly.express as px
import time

# Sample pantry staples dataset (transactions)
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

# Preprocess dataset to one-hot encoded dataframe
def preprocess(data):
    all_items = sorted(set(i for transaction in data for i in transaction))
    df = pd.DataFrame(0, index=range(len(data)), columns=all_items)
    for idx, transaction in enumerate(data):
        df.loc[idx, transaction] = 1
    return df

df = preprocess(dataset)

st.title("FP-Growth vs Apriori - Frequent Itemsets Visualization")

# Sidebar for parameters
algo = st.sidebar.selectbox("Select Algorithm", ["FP-Growth", "Apriori"])
min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.3)
chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Treemap"])

# Measure execution time of algorithm
start_time = time.time()

if algo == "FP-Growth":
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
else:
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

end_time = time.time()
exec_time = end_time - start_time
st.write(f"Execution time for {algo}: {exec_time:.4f} seconds")

st.write(f"### Frequent Itemsets - {algo}")
if frequent_itemsets.empty:
    st.write("No frequent itemsets found with the given support.")
else:
    frequent_itemsets['support_rounded'] = frequent_itemsets['support'].round(3)
    frequent_itemsets['itemset_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    
    st.dataframe(frequent_itemsets[['itemset_str', 'support_rounded']].sort_values(by='support_rounded', ascending=False))
    
    if chart_type == "Bar Chart":
        fig = px.bar(frequent_itemsets.sort_values(by='support_rounded', ascending=False).head(10),
                     x='itemset_str', y='support_rounded',
                     labels={'itemset_str': 'Itemset', 'support_rounded': 'Support'},
                     title='Top Frequent Itemsets')
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

    # Association Rules
    st.write(f"### Association Rules - {algo}")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    if rules.empty:
        st.write("No association rules found with the given threshold.")
    else:
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].sort_values(by='confidence', ascending=False))
