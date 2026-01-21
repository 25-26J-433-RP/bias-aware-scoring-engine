"""
Script to delete test data from Firebase.
Deletes documents from 'userImages' where source == 'automated_script'
"""

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def delete_collection(coll_ref, batch_size):
    docs = coll_ref.where("source", "==", "automated_script").limit(batch_size).stream()
    deleted = 0

    for doc in docs:
        print(f"Deleting doc {doc.id} => {doc.to_dict().get('source')}")
        doc.reference.delete()
        deleted += 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)
    
    return deleted

print("⚠️ DELETING test data from 'userImages'...")
deleted_count = delete_collection(db.collection("userImages"), 50)
print(f"✅ Deleted {deleted_count} documents.")
