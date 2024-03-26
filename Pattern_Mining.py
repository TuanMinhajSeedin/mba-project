import streamlit as st
import pandas as pd
import plotly.express as px

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

st.title('Pattern Mining with Association Rules')
items=pd.read_csv('Outputs/items.csv',usecols=['Itemname'])
items.Itemname=items.Itemname.str.lower()
frequent_items=pd.read_csv('Outputs/frequent items.csv',nrows=10)
association_rules=pd.read_csv('Outputs/association rules.csv')

col1,col2=st.columns((2))
with col1:
    with st.expander('View Top 10 Frequent Buying Products',expanded=True):
        st.write(frequent_items['items'])
        # Plotly Express bar chart
        fig = px.bar(frequent_items, x='items', y='freq', title='Frequency of Items', labels={'items': 'Items', 'freq': 'Frequency'})

        # Display the chart in Streamlit
        st.plotly_chart(fig)
with col2:
    with st.expander('View recomendations for Selected Products',expanded=True):
        items=st.multiselect('Select Items',items.Itemname.unique(),default=items.Itemname.unique()[:2])
        if len(items) !=0 :
            # Filter DataFrame based on selected items
            filtered_df = association_rules[association_rules['antecedent'].isin(items)]

            # Display recommendations
            if not filtered_df.empty:
                col3,col4=st.columns((0.3,0.70))
                with col4:
                    st.write("Recommendations:")
                    filtered_df = filtered_df.rename(columns={'consequent': 'Recomendation Item'})
                    st.dataframe(filtered_df[['Recomendation Item', 'confidence','lift']].sort_values(by=['confidence'],ascending=False))
                with col3:
                    st.write("Selected Items:")
                    items_df=pd.DataFrame(items)
                    items_df.columns = ['Selected Item']
                    st.dataframe(items_df)
            else:
                st.write("No recommendations available for the selected items.")

        else:
            st.warning('Enter items for recomendations')
