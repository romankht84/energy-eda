import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("energy-eda-firebase-adminsdk-5v4ee-8b271f1a29.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://energy-eda-default-rtdb.firebaseio.com/'
        })
