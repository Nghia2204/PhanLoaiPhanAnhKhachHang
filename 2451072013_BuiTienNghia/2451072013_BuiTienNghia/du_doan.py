import re
import joblib


def tien_xu_ly_van_ban(van_ban):
    van_ban = van_ban.lower()
    van_ban = re.sub(r"http\S+|www\S+", " ", van_ban)
    van_ban = re.sub(r"[^\w\sÀ-ỹ]", " ", van_ban)
    van_ban = re.sub(r"\s+", " ", van_ban).strip()
    return van_ban


mo_hinh = joblib.load("model/mo_hinh.pkl")
tfidf = joblib.load("model/tfidf.pkl")

ten_nhan = {
    "khan_cap": "Khẩn cấp",
    "binh_thuong": "Bình thường",
    "gop_y": "Góp ý"
}

noi_dung = input("Nhập phản ánh của khách hàng: ")

noi_dung_sach = tien_xu_ly_van_ban(noi_dung)
vector = tfidf.transform([noi_dung_sach])

nhan_du_doan = mo_hinh.predict(vector)[0]
xac_suat = mo_hinh.predict_proba(vector)[0]

print("\nKết quả:", ten_nhan.get(nhan_du_doan, nhan_du_doan))

print("\nXác suất từng nhãn:")
for nhan, gia_tri in zip(mo_hinh.classes_, xac_suat):
    print(f"- {ten_nhan.get(nhan, nhan)}: {gia_tri * 100:.2f}%")