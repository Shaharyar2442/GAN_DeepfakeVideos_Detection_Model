import os
from pymongo import MongoClient
import pymongo
from dotenv import load_dotenv

# Ensure the correct path to .env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("[VisionSnare DB] WARNING: MONGO_URI not found in environment!")

# Initialize MongoDB client
try:
    client = MongoClient(MONGO_URI)
    db = client['visionsnare']
    users_collection = db['users']
    history_collection = db['history']

    # Ensure unique username index
    users_collection.create_index("username", unique=True)
    print("[VisionSnare DB] Connected to MongoDB Atlas successfully")
except Exception as e:
    print(f"[VisionSnare DB] ERROR: Failed to connect to MongoDB. Details: {e}")

def get_user_by_username(username: str):
    return users_collection.find_one({"username": username})

def create_user(user_data: dict):
    users_collection.insert_one(user_data)
    return user_data

def get_history_by_username(username: str):
    # Retrieve sorted by date/time (newest first)
    cursor = history_collection.find({"username": username}, {"_id": 0}).sort("id", pymongo.DESCENDING)
    return list(cursor)

def add_history_entry(entry: dict):
    history_collection.insert_one(entry)
    return entry
