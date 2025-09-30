import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
from PIL import Image, ImageDraw, ImageFont, ExifTags
import imagehash
import os

DB_NAME = "database.db"

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
        return False  # 이미 존재하는 아이디
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    row = cursor.fetchone()
    conn.close()
    return row is not None

# ------------------ 워터마크 ------------------ #
def embed_watermark(input_path, output_path, text):
    img = Image.open(input_path).convert("RGBA")
    watermark_layer = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(watermark_layer)
    font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255,0,0,100), font=font)

    watermarked = Image.alpha_composite(img, watermark_layer)
    watermarked = watermarked.convert("RGB")  # JPEG 저장 가능하도록 변환
    watermarked.save(output_path)

def compare_images(img1_path, img2_path):
    h1 = imagehash.phash(Image.open(img1_path))
    h2 = imagehash.phash(Image.open(img2_path))
    return h1 - h2  # 값이 작을수록 유사

def get_exif(path):
    try:
        img = Image.open(path)
        exif_data = img._getexif()
        if not exif_data:
            return {}
        labeled = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            labeled[decoded] = value
        return labeled
    except:
        return {}

def compare_exif(exif1, exif2):
    diffs = []
    keys = set(exif1.keys()).union(set(exif2.keys()))
    for k in keys:
        v1 = exif1.get(k, "없음")
        v2 = exif2.get(k, "없음")
        if v1 != v2:
            diffs.append(f"{k}: 원본={v1}, 의심={v2}")
    return diffs

# ------------------ GUI ------------------ #
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VeriMark - 워터마크 인증 시스템")
        self.username = None
        self.build_login()

    def build_login(self):
        tk.Label(self.root, text="아이디").grid(row=0, column=0)
        tk.Label(self.root, text="비밀번호").grid(row=1, column=0)
        self.entry_user = tk.Entry(self.root)
        self.entry_pass = tk.Entry(self.root, show="*")
        self.entry_user.grid(row=0, column=1)
        self.entry_pass.grid(row=1, column=1)

        tk.Button(self.root, text="로그인", command=self.login).grid(row=2, column=0)
        tk.Button(self.root, text="회원가입", command=self.register).grid(row=2, column=1)

    def login(self):
        user = self.entry_user.get()
        pw = self.entry_pass.get()
        if login_user(user, pw):
            self.username = user
            messagebox.showinfo("성공", "로그인 성공!")
            self.build_main()
        else:
            messagebox.showerror("실패", "로그인 실패!")

    def register(self):
        user = self.entry_user.get()
        pw = self.entry_pass.get()
        if register_user(user, pw):
            messagebox.showinfo("성공", "회원가입 성공!")
        else:
            messagebox.showerror("실패", "이미 존재하는 아이디입니다.")

    def build_main(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Button(self.root, text="워터마크 삽입", command=self.do_watermark).pack(pady=10)
        tk.Button(self.root, text="이미지 검증", command=self.do_verify).pack(pady=10)

    def do_watermark(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        folder, filename = os.path.split(file_path)
        save_path = os.path.join(folder, "wm_" + filename)

        embed_watermark(file_path, save_path, self.username)
        messagebox.showinfo("완료", f"워터마크 삽입 완료!\n저장 경로: {save_path}")

    def do_verify(self):
        img1 = filedialog.askopenfilename(title="원본 이미지 선택", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        img2 = filedialog.askopenfilename(title="의심 이미지 선택", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if img1 and img2:
            # 이미지 유사도 비교
            diff = compare_images(img1, img2)

            # EXIF 비교
            exif1 = get_exif(img1)
            exif2 = get_exif(img2)
            diffs = compare_exif(exif1, exif2)

            result_msg = f"유사도 차이값: {diff}\n"
            if diff < 5:
                result_msg += "→ 거의 동일한 이미지\n\n"
            else:
                result_msg += "→ 차이가 큼 (조작 의심)\n\n"

            if diffs:
                result_msg += "📌 EXIF 차이점:\n" + "\n".join(diffs[:10])  # 최대 10개만 표시
            else:
                result_msg += "EXIF 데이터 차이 없음"

            messagebox.showinfo("검증 결과", result_msg)

# ------------------ 실행 ------------------ #
if __name__ == "__main__":
    init_db()
    root = tk.Tk()
    app = App(root)
    root.mainloop()
