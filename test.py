import pandas as pd
import streamlit as st
from pyvis.network import Network

# Load the dataset
path = 'dataset/Assignment-1_Data.csv'
df = pd.read_csv(path, delimiter=';', nrows=1000)

# Create a Pyvis Network
net = Network(height='750px', width='100%', notebook=True)

# Add nodes for BillNo, CustomerID, and ItemName
for col in ['BillNo', 'CustomerID', 'Itemname']:
    nodes = df[col].astype(str).unique()
    net.add_nodes(nodes)

# Add edges representing 'buys' relationship
for _, row in df.iterrows():
    source = str(row['BillNo'])
    target_customer = str(row['CustomerID'])
    target_item = str(row['Itemname'])
    
    # Check if the source and target nodes exist
    if source in net.get_nodes() and target_customer in net.get_nodes():
        net.add_edge(source, target_customer, title='buys')
    if source in net.get_nodes() and target_item in net.get_nodes():
        net.add_edge(source, target_item, title='buys')

# Display the network graph in Streamlit
st_pyvis_chart = st.pydeck_chart(net)

# Display the network graph
st.write(st_pyvis_chart)
