import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sqlite3
from PIL import Image
import piexif
import os
import random
import joblib
import numpy as np
import shutil

DB_NAME = "database.db"
MODEL_PATH = "exif_model.pkl"

# 중요 EXIF 필드
CRITICAL_FIELDS = ["DateTime", "Make", "Model", "Software", "GPSInfo"]

# ------------------ DB ------------------ #
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    row = cursor.fetchone()
    conn.close()
    return row is not None

# ------------------ EXIF 유틸 ------------------ #
def embed_exif_watermark(input_path, output_path, user_id):
    img = Image.open(input_path)

    # 원래 EXIF 불러오기
    try:
        exif_bytes = img.info.get("exif")
        exif_dict = piexif.load(exif_bytes) if exif_bytes else {
            "0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None
        }
    except:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Artist 필드 추가/수정
    exif_dict["0th"][piexif.ImageIFD.Artist] = user_id.encode("utf-8")

    # 1차 저장 (EXIF 유지)
    img.save(output_path)

    # 2차 저장 (원래 EXIF + Artist 삽입)
    exif_bytes_new = piexif.dump(exif_dict)
    piexif.insert(exif_bytes_new, output_path)

    # dataset/normal 자동 저장
    dataset_normal_dir = os.path.join("dataset", "normal")
    os.makedirs(dataset_normal_dir, exist_ok=True)
    save_copy = os.path.join(dataset_normal_dir, os.path.basename(output_path))
    shutil.copy(output_path, save_copy)

def get_exif(path):
    try:
        img = Image.open(path)
        exif_data = img.getexif()
        if not exif_data:
            return {}
        labeled = {}
        for tag_id, value in exif_data.items():
            tag_name = piexif.TAGS["0th"].get(tag_id, {"name": str(tag_id)})["name"]
            if isinstance(value, bytes):
                try:
                    value = value.decode(errors="ignore")
                except:
                    value = str(value)
            labeled[tag_name] = value
        return labeled
    except Exception as e:
        print("EXIF 읽기 실패:", e)
        return {}

def verify_exif_watermark(img_path, expected_user):
    exif_data = get_exif(img_path)
    artist = exif_data.get("Artist", None)

    if artist is None:
        wm_result = "❌ 워터마크 없음 (EXIF Artist 필드 없음)\n"
    elif artist == expected_user:
        wm_result = f"✅ 워터마크 확인됨: {artist}\n"
    else:
        wm_result = f"⚠️ 다른 워터마크 발견: {artist}\n"

    missing_fields = [f for f in CRITICAL_FIELDS if f not in exif_data]
    if missing_fields:
        wm_result += "\n⚠️ 중요 EXIF 필드 누락: " + ", ".join(missing_fields) + "\n"
    else:
        wm_result += "\n✅ 중요 EXIF 필드 정상적으로 존재\n"

    exif_lines = [f"{k}: {v}" for k, v in exif_data.items()]
    full_exif = "\n".join(exif_lines) if exif_lines else "EXIF 메타데이터 없음"
    return wm_result + "\n📌 EXIF 메타데이터:\n" + full_exif

# ------------------ 규칙 기반 비교 ------------------ #
def compare_exif(original_path, suspect_path):
    exif1 = get_exif(original_path)
    exif2 = get_exif(suspect_path)
    diffs = []
    keys = set(exif1.keys()).union(set(exif2.keys()))

    for k in keys:
        # ✅ Artist(워터마크) 필드는 무시
        if k == "Artist":
            continue

        v1 = exif1.get(k, "없음")
        v2 = exif2.get(k, "없음")
        if v1 != v2:
            explanation = ""
            if k.lower().startswith("date"):
                explanation = "📅 촬영 시간이 다릅니다 → 촬영 시각 변조 가능성"
            elif k in ["Make", "Model"]:
                explanation = "📷 카메라 정보가 다릅니다 → 다른 기기 or EXIF 수정"
            elif k == "Software":
                explanation = "💻 소프트웨어 기록이 다릅니다 → 편집/저장 프로그램 사용 흔적"
            elif "GPS" in k:
                explanation = "📍 GPS 정보가 다릅니다 → 위치 정보 변조 가능성"
            else:
                explanation = "⚠️ 일반 메타데이터 차이 발견"
            diffs.append(f"{k}: 원본={v1}, 의심={v2} → {explanation}")

    return ["✅ EXIF 차이 없음 (조작 흔적 없음으로 보임)"] if not diffs else ["⚠️ EXIF 차이 발견! 조작 가능성 있음"] + diffs

# ------------------ ML 판정 ------------------ #
def exif_to_vector(img_path):
    exif = get_exif(img_path)
    return np.array([1 if field in exif else 0 for field in CRITICAL_FIELDS]).reshape(1, -1)

def ml_predict(img_path):
    if not os.path.exists(MODEL_PATH):
        return "❌ 학습된 모델(exif_model.pkl)이 없습니다.\n먼저 train_exif_model.py를 실행하세요."
    try:
        model = joblib.load(MODEL_PATH)
        vec = exif_to_vector(img_path)
        prob = model.predict_proba(vec)[0][1] * 100
        final_label = "⚠️ 조작 의심" if prob >= 50 else "✅ 정상"
        return f"🤖 머신러닝 판정 결과: {final_label}\n조작일 확률 {prob:.2f}%"
    except Exception as e:
        return f"❌ 머신러닝 판정 오류: {e}"

# ------------------ 조작 이미지 생성 ------------------ #
def strip_exif(img_path, out_path):
    img = Image.open(img_path)
    img.save(out_path)  # EXIF 없음
    return True

def modify_fields(img_path, out_path):
    img = Image.open(img_path)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["0th"][piexif.ImageIFD.Artist] = b"tampered_user"
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = b"2000:01:01 00:00:00"
    exif_bytes = piexif.dump(exif_dict)
    img.save(out_path, exif=exif_bytes)
    return True

def random_tamper(img_path, out_path):
    return strip_exif(img_path, out_path) if random.choice(["strip", "modify"]) == "strip" else modify_fields(img_path, out_path)

# ------------------ GUI ------------------ #
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VeriMark - EXIF 기반 워터마킹 시스템")
        self.username = None
        self.tamper_mode = None
        self.build_login()

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # 로그인 화면
    def build_login(self):
        self.clear_root()
        tk.Label(self.root, text="아이디").grid(row=0, column=0)
        tk.Label(self.root, text="비밀번호").grid(row=1, column=0)
        self.entry_user = tk.Entry(self.root)
        self.entry_pass = tk.Entry(self.root, show="*")
        self.entry_user.grid(row=0, column=1)
        self.entry_pass.grid(row=1, column=1)
        tk.Button(self.root, text="로그인", command=self.login).grid(row=2, column=0)
        tk.Button(self.root, text="회원가입", command=self.register).grid(row=2, column=1)

    def login(self):
        user, pw = self.entry_user.get(), self.entry_pass.get()
        if login_user(user, pw):
            self.username = user
            messagebox.showinfo("성공", "로그인 성공!")
            self.build_main()
        else:
            messagebox.showerror("실패", "로그인 실패!")

    def register(self):
        if register_user(self.entry_user.get(), self.entry_pass.get()):
            messagebox.showinfo("성공", "회원가입 성공!")
        else:
            messagebox.showerror("실패", "이미 존재하는 아이디입니다.")

    # 로그인 이후 첫 화면
    def build_main(self):
        self.clear_root()
        tk.Button(self.root, text="워터마크 삽입", command=self.do_embed).pack(pady=10)
        tk.Button(self.root, text="이미지 삽입", command=self.build_image_menu).pack(pady=10)

    # 이미지 삽입 후 메뉴
    def build_image_menu(self):
        self.clear_root()
        tk.Button(self.root, text="원본이 있는 경우", command=self.build_with_original).pack(pady=10)
        tk.Button(self.root, text="원본이 없는 경우", command=self.build_without_original).pack(pady=10)

    # 원본이 있는 경우
    def build_with_original(self):
        self.clear_root()
        tk.Button(self.root, text="EXIF 워터마크 검증", command=self.do_verify).pack(pady=10)
        tk.Button(self.root, text="EXIF 차이 비교 (규칙 기반만)", command=self.do_compare).pack(pady=10)
        tk.Button(self.root, text="EXIF 차이 비교 (규칙 + ML)", command=self.do_compare_ml).pack(pady=10)

        self.tamper_mode = tk.StringVar(value="random")
        mode_label = tk.Label(self.root, text="조작 모드 선택:")
        mode_label.pack(pady=5)
        mode_combo = ttk.Combobox(self.root, textvariable=self.tamper_mode,
                                  values=["random", "strip", "modify"], state="readonly")
        mode_combo.pack(pady=5)

        tk.Button(self.root, text="조작 이미지 자동 생성", command=self.do_generate).pack(pady=10)
        tk.Button(self.root, text="⬅ 뒤로가기", command=self.build_image_menu).pack(pady=10)

    def build_without_original(self):
        self.clear_root()
        tk.Label(self.root, text="(여기에 원본 없는 경우 기능들을 넣으시면 됩니다)").pack(pady=20)
        tk.Button(self.root, text="⬅ 뒤로가기", command=self.build_image_menu).pack(pady=10)

    # ------------------ 기능 구현 ------------------ #
    def do_embed(self):
        file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if not file_path: return
        folder, filename = os.path.split(file_path)
        save_path = os.path.join(folder, "wm_" + filename)
        embed_exif_watermark(file_path, save_path, self.username)
        messagebox.showinfo("완료", f"워터마크 삽입 완료!\n저장 경로: {save_path}\n"
                                   f"dataset/normal 에도 자동 저장됨")

    def do_verify(self):
        file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if not file_path: return
        result = verify_exif_watermark(file_path, self.username)
        self._show_text_window("EXIF 검증 결과", result)

    def do_compare(self):
        original_path = filedialog.askopenfilename(title="원본 이미지 선택", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        suspect_path = filedialog.askopenfilename(title="의심 이미지 선택", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if not (original_path and suspect_path): return
        diffs = compare_exif(original_path, suspect_path)
        self._show_text_window("EXIF 차이 비교 결과", "\n".join(diffs))

    def do_compare_ml(self):
        original_path = filedialog.askopenfilename(title="원본 이미지 선택", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        suspect_path = filedialog.askopenfilename(title="의심 이미지 선택", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if not (original_path and suspect_path): return
        rule_results, ml_result = compare_exif(original_path, suspect_path), ml_predict(suspect_path)
        self._show_text_window("규칙 기반 판정", "\n".join(rule_results))
        self._show_text_window("ML 판정", ml_result)

    def do_generate(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if not file_paths: return
        mode, count, tampered_dir = self.tamper_mode.get(), 0, None
        for file_path in file_paths:
            folder, fname = os.path.split(file_path)
            tampered_dir = os.path.join(folder, "tampered")
            os.makedirs(tampered_dir, exist_ok=True)
            for i in range(2):
                out_name = f"tampered_{mode}_{i}_{fname}"
                out_path = os.path.join(tampered_dir, out_name)
                if mode == "strip": strip_exif(file_path, out_path)
                elif mode == "modify": modify_fields(file_path, out_path)
                else: random_tamper(file_path, out_path)
                count += 1
        messagebox.showinfo("완료", f"{count}개의 조작 이미지 생성 완료\n저장 위치: {tampered_dir}")

    def _show_text_window(self, title, content):
        win = tk.Toplevel(self.root)
        win.title(title)
        text_box = tk.Text(win, wrap="word", width=100, height=35)
        text_box.pack(expand=True, fill="both")
        text_box.tag_config("ok", foreground="green")
        text_box.tag_config("warn", foreground="orange")
        text_box.tag_config("error", foreground="red")

        for line in content.split("\n"):
            if "✅" in line: text_box.insert("end", line + "\n", "ok")
            elif "⚠️" in line: text_box.insert("end", line + "\n", "warn")
            elif "❌" in line: text_box.insert("end", line + "\n", "error")
            else: text_box.insert("end", line + "\n")
        text_box.config(state="disabled")

# ------------------ 실행 ------------------ #
if __name__ == "__main__":
    init_db()
    root = tk.Tk()
    app = App(root)
    root.mainloop()
