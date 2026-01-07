import firebase_admin
from firebase_admin import credentials, firestore

# Step 1: Load the credentials file
cred = credentials.Certificate("firebase_credentials.json")  

# Step 2: Initialize the Firebase app
firebase_admin.initialize_app(cred)

# Step 3: Get Firestore client
db = firestore.client()

# Step 4: Test write - create a document
doc_ref = db.collection("analyses").document("test_analysis")
doc_ref.set({
    "user_id": "test_user",
    "video_url": "https://youtube.com/test123",
    "sentiment": "Positive",
    "created_at": firestore.SERVER_TIMESTAMP
})

print("✅ Test document written to Firestore successfully.")
