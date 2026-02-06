import firebase_admin
from firebase_admin import credentials, firestore

def inspect_data():
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except:
        pass
        
    db = firestore.client()
    docs = list(db.collection("userImages").limit(5).stream())
    
    print(f"Inspecting {len(docs)} documents...")
    for doc in docs:
        d = doc.to_dict()
        print(f"\n--- Doc: {doc.id} ---")
        print(f"Top-level keys: {list(d.keys())}")
        print(f"studentGrade: {d.get('studentGrade')}")
        details = d.get('details', {})
        print(f"Details keys: {list(details.keys())}")
        print(f"Essay text length: {len(details.get('essay_text', ''))}")
        print(f"Dyslexic flag: {details.get('dyslexic_flag')}")

if __name__ == "__main__":
    inspect_data()
