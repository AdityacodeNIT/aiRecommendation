import pymongo
import pandas as pd
from bson import ObjectId  # Needed for MongoDB ObjectId
from dotenv import load_dotenv
import os

load_dotenv()
# Connect to MongoDB
key = os.getenv("MONGODB_URI")

dbName=os.getenv("DB_NAME")
client = pymongo.MongoClient(key)
db = client[dbName]
collection = db["userinteractions"]

# Fetch interactions from MongoDB

def fetch_all_user_interactions():
    print("Fetching all user interactions...")
    
    # Fetch all interactions from the database
    interactions = list(collection.find({}, {"_id": 0, "userId": 1, "productId": 1, "action": 1}))
    
    # Convert to DataFrame
    return pd.DataFrame(interactions)







