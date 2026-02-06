import firebase_admin
from firebase_admin import credentials, firestore

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
except:
    pass

db = firestore.client()
docs = list(db.collection("fairnessReports").stream())
print(f"Found {len(docs)} reports in fairnessReports collection")
print("-" * 60)

for d in docs:
    data = d.to_dict()
    print(f"ID: {d.id}")
    print(f"   Grade: {data.get('grade')}, DIR: {data.get('dir')}, SPD: {data.get('spd')}")
    print(f"   Timestamp: {data.get('timestamp')}")
    print()
