import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data=pd.read_csv("DataFrame_For_Pattern_Mining.csv")

data.dropna(subset=['ItemName'], inplace=True)

# Keep only unique ItemName
data = data[['ItemName']].drop_duplicates().reset_index(drop=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the item names into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(data['ItemName'])

# Step 4: Calculate Cosine Similarity
# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 5: Recommendation Function
def get_product_recommendations(selected_products, cosine_sim=cosine_sim, data=data):
    # Combine all selected products into one string
    combined_products = ' '.join(selected_products)

    # Transform the combined products into TF-IDF vector
    combined_product_tfidf = tfidf_vectorizer.transform([combined_products])

    # Compute cosine similarity between the combined products and all products in the dataset
    cosine_sim_combined = linear_kernel(combined_product_tfidf, tfidf_matrix).flatten()

    # Sort the products based on the similarity scores
    sim_scores = sorted(list(enumerate(cosine_sim_combined)), key=lambda x: x[1], reverse=True)

    # Get recommended products based on the similarity scores
    recommended_products = []
    for idx, score in sim_scores:
        recommended_product = data.loc[idx, 'ItemName']
        if recommended_product not in selected_products:
            recommended_products.append((recommended_product, score))
    
    return recommended_products

# Step 6: Get Product Recommendations
# Example: Get recommendations for multiple selected products as one basket

selected_products = st.multiselect("Select products", data['ItemName'].unique())

# Get recommendations for the selected products as one basket
product_recommendations = get_product_recommendations(selected_products)

# Convert recommendations to DataFrame
df_recommendations = pd.DataFrame(product_recommendations, columns=['Recommended Product', 'Cosine Similarity'])

# Filter out low similarity scores
df_recommendations = df_recommendations[df_recommendations['Cosine Similarity'] > 0.2].drop_duplicates().sort_values(by='Cosine Similarity',ascending=False)

# Display the DataFrame using st.write()
st.write(df_recommendations)