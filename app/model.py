import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from bson import ObjectId  # Needed for MongoDB ObjectId

class RecommendationModel:
    def __init__(self):
        self.user_product_matrix = None
        self.user_similarity_df = None
        
        
  

    def train(self, df):
        if df.empty:
            print(" No data available for training!")
            return

        # ✅ Convert IDs to strings for consistency
        df["userId"] = df["userId"].astype(str)
        df["productId"] = df["productId"].astype(str)

        # ✅ Assign a default interaction score (1) and aggregate interactions
        df["interaction"] = 1  
        df = df.groupby(["userId", "productId"], as_index=False).sum()

        # ✅ Create user-product matrix (pivot table)
        self.user_product_matrix = df.pivot(index="userId", columns="productId", values="interaction").fillna(0)

        # ✅ Normalize the matrix to prevent bias
        from sklearn.preprocessing import normalize
        normalized_matrix = pd.DataFrame(
            normalize(self.user_product_matrix, axis=1),
            index=self.user_product_matrix.index,
            columns=self.user_product_matrix.columns
        )

        # ✅ Compute cosine similarity
        self.user_similarity_df = pd.DataFrame(
            cosine_similarity(normalized_matrix),
            index=self.user_product_matrix.index,
            columns=self.user_product_matrix.index
        )

        print("Training completed successfully!")


   
    def recommend(self, user_id, top_n=3):
        user_id = str(user_id)  # Convert to string
        
        print("abc",self.user_product_matrix.index)

        # Ensure both indices are strings
        self.user_product_matrix.index = self.user_product_matrix.index.astype(str)
        self.user_similarity_df.index = self.user_similarity_df.index.astype(str)
        self.user_similarity_df.columns = self.user_similarity_df.columns.astype(str)

        print("User ID in request:", user_id)
        print("Available User IDs in matrix:", self.user_product_matrix.index.tolist())
        print("Available User IDs in similarity dataframe:", self.user_similarity_df.columns.tolist())

        # 🔹 Check if user exists in BOTH matrices
        if user_id not in self.user_product_matrix.index:
            print(" User ID not found in product matrix!")
            return []
        
        if user_id not in self.user_similarity_df.columns:
            print(" User ID not found in similarity dataframe!")
            return []

        # Fetch similar users
        similar_users = self.user_similarity_df[user_id].drop(user_id, errors="ignore").sort_values(ascending=False).index
        print(" Similar Users:", similar_users)

        recommended_products = set()

        for similar_user in similar_users:
            user_purchases = self.user_product_matrix.loc[similar_user]
            recommended_products.update(user_purchases[user_purchases > 0].index.tolist())

            if len(recommended_products) >= top_n:
                break

        return list(recommended_products)



    def save_model(self, filename="model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump((self.user_product_matrix, self.user_similarity_df), f)

    def load_model(self, filename="model.pkl"):
        with open(filename, "rb") as f:
            self.user_product_matrix, self.user_similarity_df = pickle.load(f)

    