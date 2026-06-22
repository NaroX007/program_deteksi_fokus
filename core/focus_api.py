import requests
import time

BASE_URL = "http://10.44.9.40:5000"


# =========================
# GET CURRENT USER
# =========================
def get_current_user():

    for _ in range(3):

        try:
            response = requests.get(
                f"{BASE_URL}/api/get_current_user",
                timeout=5
            )

            data = response.json()

            if data.get("status") == "success":
                return data.get("user_id")

        except Exception as e:
            print("Error current user:", e)

        time.sleep(1)

    return None


# =========================
# GET ACTIVE ANAK
# =========================
def get_active_anak(user_id):

    for _ in range(3):

        try:
            response = requests.get(
                f"{BASE_URL}/api/get_active_anak",
                params={"user_id": user_id},
                timeout=5
            )

            data = response.json()

            if data.get("status") == "success":
                return data.get("anak")

        except Exception as e:
            print("Error active anak:", e)

        time.sleep(1)

    return None


# =========================
# GET ACTIVE SESSION
# =========================
def get_active_session(anak_id):

    for _ in range(3):

        try:
            response = requests.get(
                f"{BASE_URL}/api/get_active_session",
                params={"anak_id": anak_id},
                timeout=5
            )

            data = response.json()

            if data.get("status") == "success":
                return data.get("session")

        except Exception as e:
            print("Error active session:", e)

        time.sleep(1)

    return None

# =========================
# SAVE FINAL FOCUS RESULT
# =========================
def save_focus_result(anak_id, session_id, status):

    # Jangan kirim jika session belum ada
    if not anak_id or not session_id:
        print("[API] Session belum aktif")
        return

    try:
        response = requests.post(
            f"{BASE_URL}/api/save_focus_result",
            json={
                "anak_id": anak_id,
                "session_id": session_id,
                "status": status
            },
            timeout=5
        )

        data = response.json()

        if data.get("status") == "success":
            print(f"[API] Hasil akhir '{status}' berhasil dikirim!")
        else:
            print("[API] Gagal kirim:", data)

    except Exception as e:
        print("[API] Error:", e)

# =========================
# GET LATEST ACTIVE SESSION
# =========================
def get_latest_active_session():

    for _ in range(3):

        try:
            response = requests.get(
                f"{BASE_URL}/api/get_latest_active_session",
                timeout=5
            )

            data = response.json()

            if data.get("status") == "success":
                return data.get("session")

        except Exception as e:
            print("Error latest session:", e)

        time.sleep(1)

    return None

# =========================
# GET FULL ACTIVE CONTEXT
# =========================
def get_active_context():

    """
    Mengambil:
    - user aktif
    - anak aktif
    - session aktif

    Return:
    {
        "user_id": ...,
        "anak_id": ...,
        "session_id": ...
    }
    """

    user_id = get_current_user()

    if not user_id:
        print("Tidak ada user aktif")
        return None

    anak = get_active_anak(user_id)

    if not anak:
        print("Tidak ada anak aktif")
        return None

    anak_id = anak["id"]

    session_data = get_active_session(anak_id)

    if not session_data:
        print("Tidak ada session aktif")
        return None

    session_id = session_data["id"]

    return {
        "user_id": user_id,
        "anak_id": anak_id,
        "session_id": session_id
    }
