import firebase_admin
from firebase_admin import credentials, firestore
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KEY_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "deteksi-fokus-firebase-adminsdk-fbsvc-d2b97392a4.json"
    )
)

db = None


# =========================
# INIT FIREBASE
# =========================
def init_firebase():

    global db

    if not os.path.exists(KEY_PATH):
        print("Firebase key tidak ditemukan:", KEY_PATH)
        return

    try:

        if not firebase_admin._apps:

            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)

        db = firestore.client()

        print("Firebase connected")

    except Exception as e:
        print("Firebase init error:", e)


# =========================
# SEND SESSION DATA
# =========================
def send_session_data(session_data, max_retry=3):

    if db is None:
        print("Firebase belum siap")
        return

    session_id = session_data.get("session_id")

    if not session_id:
        print("Session ID tidak ada")
        return

    for i in range(max_retry):

        try:

            db.collection("sessions_timeline") \
              .document(str(session_id)) \
              .set(session_data)

            print("Firebase: timeline sesi berhasil dikirim")

            return

        except Exception as e:

            print(f"Firebase error retry {i+1}:", e)

            time.sleep(1)

    print("Gagal kirim ke Firebase")


# INIT
init_firebase()