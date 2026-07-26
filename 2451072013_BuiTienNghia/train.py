import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


def tien_xu_ly_van_ban(van_ban):
    """Chuẩn hóa nội dung phản ánh."""
    if not isinstance(van_ban, str):
        return ""

    van_ban = van_ban.lower()
    van_ban = re.sub(r"http\S+|www\S+", " ", van_ban)
    van_ban = re.sub(r"[^\w\sÀ-ỹ]", " ", van_ban)
    van_ban = re.sub(r"\s+", " ", van_ban).strip()

    return van_ban


def main():
    duong_dan_du_lieu = "data/phan_anh_khach_hang.csv"

    if not os.path.exists(duong_dan_du_lieu):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {duong_dan_du_lieu}"
        )

    du_lieu = pd.read_csv(duong_dan_du_lieu)

    cot_bat_buoc = {"noi_dung", "nhan"}
    if not cot_bat_buoc.issubset(du_lieu.columns):
        raise ValueError(
            "File CSV phải có hai cột: noi_dung và nhan"
        )

    du_lieu = du_lieu.dropna(subset=["noi_dung", "nhan"])
    du_lieu = du_lieu.drop_duplicates(subset=["noi_dung"])

    du_lieu["noi_dung_sach"] = du_lieu["noi_dung"].apply(
        tien_xu_ly_van_ban
    )

    x = du_lieu["noi_dung_sach"]
    y = du_lieu["nhan"]

    print("Số lượng dữ liệu theo nhãn:")
    print(y.value_counts())

    # 4. Chia tập huấn luyện và kiểm tra
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1
    )

    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    mo_hinh = MultinomialNB(alpha=1.0)
    mo_hinh.fit(x_train_tfidf, y_train)

    y_du_doan = mo_hinh.predict(x_test_tfidf)

    do_chinh_xac = accuracy_score(y_test, y_du_doan)

    print(f"\nAccuracy: {do_chinh_xac:.4f}")
    print("\nBáo cáo đánh giá:")
    print(
        classification_report(
            y_test,
            y_du_doan,
            zero_division=0
        )
    )

    # 9. Vẽ ma trận nhầm lẫn
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_du_doan,
        cmap="Blues",
        values_format="d"
    )

    plt.title("Ma trận nhầm lẫn")
    plt.tight_layout()
    plt.savefig("ma_tran_nham_lan.png", dpi=200)
    plt.show()

    # 10. Lưu mô hình
    os.makedirs("model", exist_ok=True)

    joblib.dump(mo_hinh, "model/mo_hinh.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

    print("\nĐã lưu mô hình vào thư mục model/")


if __name__ == "__main__":
    main()