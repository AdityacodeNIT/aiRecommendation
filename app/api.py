from flask import Flask, request, jsonify
from app.model import RecommendationModel
import os

app = Flask(__name__)

model = RecommendationModel()
model_path = "models/recommendation_model.pkl"

# 🔹 Only load the model if it exists
if os.path.exists(model_path):
    model.load_model(model_path)
    print(" Model loaded successfully!")
else:
    print(" Model file not found. Train the model first.")

#  Fix: Home route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI Recommendation API is running!"})

# fix: Corrected /recommend/<user_id> route
@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    """API endpoint to get recommendations for a user."""
    recommendations = model.recommend(user_id)
    return jsonify({"user_id": user_id, "recommended_products": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # ✅ Changed port to 5000 for consistency
