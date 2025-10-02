import os
import joblib
import numpy as np
from PIL import Image
import piexif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 중요 EXIF 필드
CRITICAL_FIELDS = ["DateTime", "Make", "Model", "Software", "GPSInfo"]

# ------------------ EXIF 추출 ------------------ #
def get_exif(path):
    try:
        img = Image.open(path)
        exif_data = img.info.get("exif")
        if not exif_data:
            return {}
        exif_dict = piexif.load(exif_data)
        labeled = {}
        for ifd in exif_dict:
            for tag, value in exif_dict[ifd].items():
                tag_name = piexif.TAGS[ifd][tag]["name"] if tag in piexif.TAGS[ifd] else tag
                if isinstance(value, bytes):
                    try:
                        value = value.decode(errors="ignore")
                    except:
                        value = str(value)
                labeled[tag_name] = value
        return labeled
    except Exception:
        return {}

# ------------------ 벡터화 ------------------ #
def exif_to_vector(img_path):
    exif = get_exif(img_path)
    vector = []
    for field in CRITICAL_FIELDS:
        vector.append(1 if field in exif else 0)
    return vector

# ------------------ 데이터셋 생성 ------------------ #
def load_dataset(dataset_dir):
    X, y = [], []
    for label_name, label in [("normal", 0), ("tampered", 1)]:
        folder = os.path.join(dataset_dir, label_name)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)  # ✅ 자동 폴더 생성
            print(f"⚠️ '{folder}' 폴더가 없어 새로 생성했습니다.")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(folder, fname)
                vec = exif_to_vector(path)
                X.append(vec)
                y.append(label)
    return np.array(X), np.array(y)

# ------------------ 학습 ------------------ #
def train_model(dataset_dir="dataset"):
    # ✅ dataset 기본 구조 확인 및 생성
    for sub in ["normal", "tampered"]:
        subdir = os.path.join(dataset_dir, sub)
        os.makedirs(subdir, exist_ok=True)

    X, y = load_dataset(dataset_dir)
    if len(X) == 0:
        print("❌ 데이터셋이 비어 있습니다.")
        print("👉 dataset/normal, dataset/tampered 폴더에 이미지를 넣으세요.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 분류 리포트:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "exif_model.pkl")
    print("✅ 모델 저장 완료: exif_model.pkl")

if __name__ == "__main__":
    train_model()
