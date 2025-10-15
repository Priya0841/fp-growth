import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import io
import time

st.title("Frequent Itemsets and Association Rules Explorer")

# Sidebar parameters
algo = st.sidebar.selectbox("Select Algorithm", ["FP-Growth", "Apriori"])
min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.1, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.6, 0.01)
min_lift = st.sidebar.slider("Minimum Lift", 0.01, 10.0, 1.0, 0.01)

# Removed Treemap option, so only Bar and Pie charts remain
chart_type = st.sidebar.selectbox("Select Frequent Itemsets Chart", ["Bar Chart", "Pie Chart"])

uploaded_file = st.file_uploader("Upload your dataset CSV file\n(Each row: transaction items comma-separated, e.g. 'A, C')")

def preprocess_data(data):
    clean_data = []
    for transaction in data:
        clean_transaction = [str(item).strip() for item in transaction if item and str(item).strip() != '']
        clean_data.append(clean_transaction)
    all_items = sorted(set(i for transaction in clean_data for i in transaction))
    df = pd.DataFrame(0, index=range(len(clean_data)), columns=all_items)
    for idx, transaction in enumerate(clean_data):
        df.loc[idx, transaction] = 1
    return df

def plot_rules_network_enhanced(rules):
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
                           weight=width,
                           color=color,
                           lift=row['lift'],
                           support=row['support'],
                           confidence=row['confidence'],
                           category=category)

    pos = nx.spring_layout(G, k=0.5, seed=42)

    edge_traces = []
    categories_handled = set()
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=d['weight'], color=d['color']),
            hoverinfo='text',
            mode='lines',
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

if uploaded_file:
    try:
        file_text = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        transactions = [line.strip().split(',') for line in file_text.splitlines() if line.strip()]
        transactions = [[item.strip() for item in trans] for trans in transactions]
    except Exception as e:
        st.error(f"Error reading file: {e}")
        transactions = []
else:
    transactions = [
        ["A"],
        ["B"],
        ["A"],
        ["B"],
        ["C"],
        ["A"],
        ["B"],
        ["A", "C"],
        ["B", "C"],
        ["D"]
    ]
    st.info("Using default sample data")

if not transactions:
    st.warning("No transactions available to analyze")
else:
    df = preprocess_data(transactions)
    st.write(f"Processed one-hot encoded data: {df.shape[0]} transactions, {df.shape[1]} unique items")
    with st.expander("Show one-hot encoded data preview"):
        st.dataframe(df.head())

    start = time.time()
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True) if algo == "FP-Growth" else apriori(df, min_support=min_support, use_colnames=True)
    end = time.time()
    st.write(f"Execution time for {algo}: {end - start:.4f} seconds")

    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found for the chosen minimum support.")
    else:
        frequent_itemsets['support_rounded'] = frequent_itemsets['support'].round(3)
        frequent_itemsets['itemset_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))

        st.subheader(f"Frequent Itemsets ({algo})")
        st.dataframe(frequent_itemsets[['itemset_str', 'support_rounded']].sort_values(by='support_rounded', ascending=False))

        top10 = frequent_itemsets.sort_values(by='support_rounded', ascending=False).head(10)
        if chart_type == "Bar Chart":
            fig = px.bar(top10, x='itemset_str', y='support_rounded',
                         labels={'itemset_str': 'Itemset', 'support_rounded': 'Support'},
                         title='Top 10 Frequent Itemsets')
        else:  # Pie Chart
            fig = px.pie(top10, values='support_rounded', names='itemset_str',
                         title='Support Distribution of Top 10 Itemsets')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Association Rules ({algo})")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]

        if rules.empty:
            st.warning("No association rules found for the chosen confidence and lift thresholds.")
        else:
            # Create readable rule strings like {Bread, Butter} → {Milk}
            rules['rule_str'] = rules.apply(
                lambda row: f"{{{', '.join(sorted(row['antecedents']))}}} → {{{', '.join(sorted(row['consequents']))}}}",
                axis=1
            )
            st.dataframe(rules[['rule_str', 'support', 'confidence', 'lift']].sort_values(by='confidence', ascending=False))

            st.subheader("Association Rules Network Graph")
            plot_rules_network_enhanced(rules)
