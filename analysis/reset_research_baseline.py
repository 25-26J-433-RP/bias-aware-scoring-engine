import firebase_admin
from firebase_admin import credentials, firestore

def reset_baseline():
    print("=" * 60)
    print("🔥 FIREBASE RESEARCH BASELINE RESET")
    print("=" * 60)
    
    # Initialize Firestore
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        # If already initialized
        pass
        
    db = firestore.client()
    
    collections_to_clear = ["userImages", "fairnessReports"]
    
    for coll_name in collections_to_clear:
        print(f"\n📁 Clearing collection: {coll_name}...")
        docs = db.collection(coll_name).list_documents()
        deleted_count = 0
        
        for doc in docs:
            doc.delete()
            deleted_count += 1
            if deleted_count % 10 == 0:
                print(f"   Deleted {deleted_count} documents...")
        
        print(f"✅ Finished! Total deleted in {coll_name}: {deleted_count}")

    print("\n" + "=" * 60)
    print("🎉 BASELINE RESET COMPLETE")
    print("Your environment is now clean for fresh research data collection.")
    print("=" * 60)

if __name__ == "__main__":
    confirm = input("⚠️ This will PERMANENTLY delete all essay scores. Proceed? (y/n): ")
    if confirm.lower() == 'y':
        reset_baseline()
    else:
        print("❌ Reset cancelled.")
