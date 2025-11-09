import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import io
import time
import random

def preprocess_data(data):
    # Deduplicate items in each transaction
    clean_transactions = []
    for t in data:
        clean_t = list(set([str(item).strip() for item in t if item and str(item).strip() != '']))
        clean_transactions.append(clean_t)
    all_items = sorted(set(i for t in clean_transactions for i in t))
    df = pd.DataFrame(0, index=range(len(clean_transactions)), columns=all_items)
    for idx, t in enumerate(clean_transactions):
        df.loc[idx, t] = 1
    return df

def plot_rules_network(rules):
    G = nx.DiGraph()
    def edge_style(lift):
        if lift < 1.2:
            return 1, 'lightgray', 'Low Lift (<1.2)'
        elif lift < 1.6:
            return 3, 'orange', 'Medium Lift (1.2-1.6)'
        else:
            return 6, 'red', 'High Lift (>1.6)'
    for _, row in rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                width, color, category = edge_style(row['lift'])
                G.add_edge(antecedent, consequent,
                           weight=width, color=color,
                           lift=row['lift'], support=row['support'], confidence=row['confidence'],
                           category=category)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    edge_traces = []
    categories_handled = set()
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=d['weight'], color=d['color']),
            hoverinfo='text', mode='lines',
            text=(f"Antecedent: {u}<br>Consequent: {v}<br>"
                  f"Lift: {d['lift']:.3f}<br>Confidence: {d['confidence']:.3f}<br>Support: {d['support']:.3f}"),
            name=d['category'] if d['category'] not in categories_handled else None,
            showlegend=d['category'] not in categories_handled
        )
        categories_handled.add(d['category'])
        edge_traces.append(edge_trace)
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition="top center", hoverinfo='text',
        marker=dict(color='#1f78b4', size=20, line=dict(width=2, color='DarkSlateGrey'))
    )
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=True,
                        legend=dict(title='Edge Lift Levels', x=0.85, y=1.0),
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig, use_container_width=True)

st.title("Market Basket Analysis with Summary Insights")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
sample_frac = st.sidebar.slider("Fraction of transactions to analyze", 0.01, 1.0, 0.3, 0.01)

if uploaded_file:
    content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    sample_size = max(1, int(len(lines) * sample_frac))
    sampled = random.sample(lines, sample_size)
    transactions = [line.split(",") for line in sampled]
    transactions = [[item.strip() for item in trans] for trans in transactions]
else:
    transactions = [
        ["milk", "bread"], ["milk", "butter"], ["bread", "jam"], ["milk", "bread", "butter"],
        ["bread", "eggs"], ["butter", "cheese"], ["milk", "cheese"], ["milk", "tea"],
        ["tea", "coffee"], ["bread", "tea"], ["coffee", "sugar"], ["tea", "sugar"], ["milk", "coffee"]
    ]
    st.info("Using default sample data")

if not transactions:
    st.warning("No transactions found to analyze")
else:
    df = preprocess_data(transactions)
    st.write(f"Data processed: {df.shape[0]} transactions, {df.shape[1]} unique items")
    with st.expander("Show a few processed transactions"):
        st.dataframe(df.head())

    min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.6, 0.01)
    min_lift = st.sidebar.slider("Minimum Lift", 0.01, 10.0, 1.2, 0.01)
    algo = st.sidebar.selectbox("Algorithm", ["FP-Growth", "Apriori", "MS-Apriori"])

    if algo in ["FP-Growth", "Apriori"]:
        min_support_global = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.1, 0.01)
    else:
        st.sidebar.markdown("### MS-Apriori per-item minimum supports")
        item_min_supports = {}
        for item in df.columns:
            item_min_supports[item] = st.sidebar.slider(f"{item} support", 0.01, 1.0, 0.1, 0.01)
        min_support_global = min(item_min_supports.values())

    start = time.time()
    if algo == "FP-Growth":
        frequent_itemsets = fpgrowth(df, min_support=min_support_global, use_colnames=True)
    elif algo == "Apriori":
        frequent_itemsets = apriori(df, min_support=min_support_global, use_colnames=True)
    else:
        frequent_itemsets = apriori(df, min_support=min_support_global, use_colnames=True)
        def filter_func(row):
            return all(row['support'] >= item_min_supports[item] for item in row['itemsets'])
        frequent_itemsets = frequent_itemsets[frequent_itemsets.apply(filter_func, axis=1)]
    duration = time.time() - start
    st.write(f"Mining finished in {duration:.2f}s, found {len(frequent_itemsets)} frequent itemsets.")

    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found. Adjust parameters.")
    else:
        frequent_itemsets['itemset_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(x)))
        top10 = frequent_itemsets.sort_values("support", ascending=False).head(10)

        chart_type = st.sidebar.selectbox("Itemsets Chart Type", ["Bar Chart", "Pie Chart"])
        if chart_type == "Bar Chart":
            fig = px.bar(top10, x='itemset_str', y='support', title='Top 10 Frequent Itemsets')
        else:
            fig = px.pie(top10, names='itemset_str', values='support', title='Top 10 Frequent Itemsets Support')
        st.plotly_chart(fig)

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]

        if not rules.empty:
            # Remove symmetric pairs duplicates in summary:
            shown_pairs = set()
            rules['pair_key'] = rules.apply(
                lambda r: tuple(sorted(list(r['antecedents']) + list(r['consequents']))), axis=1)
            rules = rules.drop_duplicates('pair_key')

            rules['rule_str'] = rules.apply(
                lambda r: f"{{{', '.join(sorted(r['antecedents']))}}} → {{{', '.join(sorted(r['consequents']))}}}", axis=1)

            st.subheader(f"Association Rules ({algo})")
            st.dataframe(rules[['rule_str', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))

            st.subheader("Association Rules Network Graph")
            plot_rules_network(rules)

            
            show_summary = st.sidebar.checkbox("Show Summary Insights", True)
            if show_summary:
                st.header("Summary Insights")

                st.subheader("Items to Place Together")
                for _, row in rules.head(20).iterrows():
                    pair = tuple(sorted(list(row['antecedents']) + list(row['consequents'])))
                    if pair in shown_pairs:
                        continue
                    shown_pairs.add(pair)
                    st.write(f"Place ({', '.join(sorted(row['antecedents']))}) together with ({', '.join(sorted(row['consequents']))}) - Lift: {row['lift']:.2f}")

                st.subheader("Cross-Selling Opportunities")
                cross_sell = rules[(rules['lift'] > 1.5) & (rules['confidence'] > 0.7)].sort_values(by='confidence', ascending=False)
                cross_shown = set()
                if not cross_sell.empty:
                    for _, row in cross_sell.iterrows():
                        pair = tuple(sorted(list(row['antecedents']) + list(row['consequents'])))
                        if pair in cross_shown or pair in shown_pairs:
                            continue
                        cross_shown.add(pair)
                        st.write(f"Bundle {', '.join(row['antecedents'])} and {', '.join(row['consequents'])} for higher sales (Confidence: {row['confidence']:.2f})")
                else:
                    st.write("No strong cross-selling opportunities found with current parameters.")

        else:
            st.warning("No association rules found with current parameters.")
