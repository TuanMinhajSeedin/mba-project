import pandas as pd
import warnings
import pandas as pd
from collections import defaultdict
from pyvis.network import Network
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import community as community_louvain

st.set_page_config(page_title='Recomendation',page_icon='',layout='wide')
custom_css = """
<style>
body {
    background-color: #0E1117; 
    secondary-background {
    background-color: #262730; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
st.title('Items Relationships')

# Read CSV file into DataFrame
relationship_df = pd.read_csv("Outputs/relationships_greater_50.csv",nrows=1000)

# # # Create a graph from a pandas dataframe
G = nx.from_pandas_edgelist(relationship_df, 
                            source = "source", 
                            target = "target", 
                            edge_attr = "value", 
                            create_using = nx.Graph())

communities = community_louvain.best_partition(G)

net = Network(notebook = True, width="1000px", height="700px", bgcolor='#222222', font_color='white')

node_degree = dict(G.degree)

# #Setting up node size attribute
nx.set_node_attributes(G, node_degree, 'size')
nx.set_node_attributes(G, communities, 'group')

net.from_nx(G)
# net.repulsion(node_distance=9000, spring_length=20000)
# net.set_options("""
#     var options = {
#     "physics": {
#         "enabled": false
#     }
#     }
# """)
# net.show('relationship_new.html')

# col1,col2=st.columns((0.70,0.3))
# with col1:
st.markdown('Overall Items Relationships')
st.components.v1.html(open("relationship_new.html").read(), height=700, width=1000)


# st.markdown(html_content, unsafe_allow_html=True)

items=pd.read_csv('Outputs/items.csv',usecols=['Itemname'])
items.Itemname=items.Itemname.str.upper()
items=st.multiselect('Select Items',communities.keys(),default=['GREEN REGENCY TEACUP AND SAUCER'])


selected_communities = []  # Example: Selected communities

for i in items:
    selected_communities.append(communities[i])
community_items = [key for key, value in communities.items() if value in selected_communities]
# filtered_df = relationship_df.head(1000)[relationship_df.head(1000)['source'].isin(community_items) | relationship_df.head(1000)['target'].isin(community_items)]
filtered_df1 = relationship_df[relationship_df['source'].isin(items) & relationship_df['target'].isin(items)]
relationship_df = pd.concat([filtered_df1, relationship_df]).drop_duplicates(keep=False)
filtered_df2=relationship_df[relationship_df['source'].isin(items) | relationship_df['target'].isin(items)].sort_values(by=['value'],ascending=False)
filtered_df3=relationship_df[~relationship_df['source'].isin(items) & ~relationship_df['target'].isin(items)].sort_values(by=['value'],ascending=False)

filtered_df=pd.concat([filtered_df1,filtered_df2, filtered_df3], axis=0)
# Iterate over DataFrame
for idx, row in filtered_df.iterrows():
    if row['source'] not in items:
        filtered_df.at[idx, 'source'], filtered_df.at[idx, 'target'] = filtered_df.at[idx, 'target'], filtered_df.at[idx, 'source']


selected_communities=list(set(selected_communities))
filtered_nodes = [node for node, community in communities.items() if community in selected_communities]
filtered_graph = G.subgraph(filtered_nodes)

# Step 2: Convert the filtered graph to a PyVis network object
filtered_net = Network(notebook=True, width="1000px", height="700px", bgcolor='#222222', font_color='white')
filtered_net.from_nx(filtered_graph)

# Step 3: Show the filtered network
filtered_net.show("filtered_network.html")
st.markdown('Custom Items Relationships')

st.components.v1.html(open("filtered_network.html").read(), height=700, width=1000)

st.markdown('Item Recomendations')

st.write(filtered_df.drop(columns=['Unnamed: 0']))
