from flask import Flask, request, jsonify
import pandas as pd
import os
from app.model import RecommendationModel  
from app.utils import fetch_all_user_interactions  

app = Flask(__name__)

# Load or initialize the model
model = RecommendationModel()
try:
    model.load_model("models/recommendation_model.pkl")
    print("Model loaded successfully!")
except:
    print("No pre-trained model found, starting fresh.")

# 🔹 Train the model with MongoDB data
@app.route("/", methods=["GET"])
def home():
    return "API is running!"

@app.route("/train", methods=["POST"])
def train_model():
    try:
        df = fetch_all_user_interactions()  # Fetch all users' interactions
        if df.empty:
            return jsonify({"message": "No interactions found, skipping training."}), 400

        model.train(df)  # Train on complete dataset
        model.save_model("models/recommendation_model.pkl")
        print("Model trained on all users and saved successfully!")

        return jsonify({"message": "Model trained successfully!"})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/recommend/<user_id>", methods=["GET"])
def get_recommendations(user_id):
    recommended_products = model.recommend(user_id)
    return jsonify({"recommended_products": recommended_products})

print("Registered routes:", app.url_map)  


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Render will set PORT automatically; default to 5001 locally
    app.run(host="0.0.0.0", port=port)  # 0.0.0.0 makes it accessible publicly (Render needs this)
