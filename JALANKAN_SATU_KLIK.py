import os
import subprocess
import time
import re
from google.colab import drive

print("=" * 60)
print("🚀 ACSR DASHBOARD — SATU KLIK JALANKAN")
print("=" * 60)

# Selalu mulai dari /content, apapun posisi direktori kerja sebelumnya
os.chdir("/content")

ZIP_NAME = "/content/acsr_dashboard_v81.zip"
PROJECT_DIR = "/content/acsr_v81"

# 1. Ekstrak zip -- SELALU dijalankan (timpa apapun yang sudah ada), TIDAK
# pernah di-skip. Sebelumnya kalau PROJECT_DIR sudah ada dari percobaan
# lama, ekstraksi zip baru dilewati -- akibatnya app.py LAMA yang tetap
# jalan meski kamu upload zip baru dengan nama sama. Sekarang dijamin
# selalu pakai isi zip yang paling baru diupload.
print("\n⏳ [1/5] Mengekstrak project (selalu timpa versi lama)...")
if not os.path.exists(ZIP_NAME):
    raise FileNotFoundError(
        f"'{ZIP_NAME}' tidak ditemukan. "
        f"Upload dulu file zip-nya ke Colab (ikon folder di sidebar kiri -> upload), "
        f"lalu jalankan cell ini lagi."
    )
os.system(f"unzip -oq {ZIP_NAME} -d {PROJECT_DIR}")
os.chdir(PROJECT_DIR)
print(f"   -> Project siap di {PROJECT_DIR} (fresh dari zip terbaru)")

# 2. Mount Google Drive
print("\n⏳ [2/5] Menghubungkan Google Drive...")

# Cek dulu apakah SUDAH ter-mount dengan baik -- kalau sudah dan sehat,
# JANGAN force_remount (itu bisa memicu proses mount ulang yang diam-diam
# gagal / tidak menunjukkan popup otorisasi seperti biasanya kalau sesi
# sebelumnya bermasalah). Cuma mount kalau BENAR-BENAR belum ter-mount.
already_ok = os.path.exists('/content/drive/MyDrive') and len(os.listdir('/content/drive/MyDrive')) > 0

if already_ok:
    print("   -> Drive sudah ter-mount & terisi sebelumnya, skip mount ulang.")
else:
    print("   -> Drive belum ter-mount / kosong, menjalankan drive.mount()...")
    print("   -> PERHATIKAN: seharusnya muncul link/popup otorisasi Google di sini.")
    print("      Kalau TIDAK ada popup/link sama sekali dalam beberapa detik,")
    print("      itu tandanya ada yang salah -- coba Runtime > Restart session dulu.")
    try:
        drive.mount('/content/drive', force_remount=True)
    except Exception as e:
        print(f"   ❌ drive.mount() melempar error: {e}")

# Verifikasi Drive benar-benar siap (kadang butuh beberapa detik lagi
# setelah mount() return sebelum semua file benar-benar terlihat) --
# tunggu sampai folder Drive terisi, maksimal 15 detik.
import glob as _glob
for _ in range(15):
    if _glob.glob('/content/drive/MyDrive/*'):
        break
    time.sleep(1)

if os.path.exists('/content/drive/MyDrive') and len(os.listdir('/content/drive/MyDrive')) > 0:
    print(f"   ✅ Drive siap, {len(os.listdir('/content/drive/MyDrive'))} item terlihat di MyDrive.")
else:
    print("   ❌ Drive TIDAK siap / MyDrive kosong setelah menunggu. Model TIDAK akan ketemu.")
    print("      -> Jalankan Runtime > Restart session, lalu coba dari awal lagi.")

# 3. Install dependency
print("\n⏳ [3/5] Menginstall dependency (torch/streamlit/dll)...")
os.system("pip install -q -r requirements.txt")

# 4. Matikan proses lama & jalankan Streamlit
print("\n⏳ [4/5] Menjalankan Streamlit...")
os.system("pkill -9 -f streamlit")
time.sleep(2)
os.system("pkill -f cloudflared")

# Log Streamlit ke file (bukan cuma subprocess.PIPE yang tidak dibaca) --
# supaya kalau Streamlit CRASH saat start (misal gagal load model), kita
# bisa langsung tunjukkan alasan aslinya ke user, bukan cuma "Bad Gateway"
# generik dari Cloudflare yang sama sekali tidak menjelaskan akar masalah.
#
# PYTHONUNBUFFERED=1 WAJIB di sini -- tanpa ini, proses Python anak (Streamlit)
# akan menumpuk outputnya di buffer internal dan TIDAK menulis apa pun ke file
# log sampai buffer penuh atau prosesnya keluar. Itu sebabnya percobaan
# sebelumnya log-nya tampak KOSONG TOTAL walau Streamlit sebenarnya masih
# hidup/masih proses import -- bukan berarti tidak ada output sama sekali.
STREAMLIT_LOG = "/content/streamlit_log.txt"
with open(STREAMLIT_LOG, "w") as _f:
    pass  # kosongkan log lama
_streamlit_env = os.environ.copy()
_streamlit_env["PYTHONUNBUFFERED"] = "1"
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=open(STREAMLIT_LOG, "a"), stderr=subprocess.STDOUT,
    env=_streamlit_env,
)

# --- CEK KESEHATAN: pastikan Streamlit BENAR-BENAR merespons di port 8501
# sebelum membuka tunnel. Ini yang selama ini TIDAK dicek -- akibatnya kalau
# Streamlit crash duluan (paling sering karena model gagal di-load), tunnel
# tetap dibuka dan menghasilkan link yang keliatan "berhasil" padahal di
# baliknya kosong -> browser selalu dapat "502 Bad Gateway", persis seperti
# yang dialami sekarang, dan errornya BERULANG di setiap percobaan karena
# akar masalahnya bukan soal jaringan/tunnel sama sekali.
#
# Timeout dinaikkan jadi 100 detik (dari 30 detik) -- import torch + cv2 +
# reportlab lalu load model dari Google Drive di Colab bisa makan waktu
# cukup lama saat cold-start, jadi 30 detik kemarin kemungkinan besar
# membuat kita menyerah SEBELUM Streamlit sempat siap, bukan berarti dia
# benar-benar gagal.
print("   -> Menunggu Streamlit siap merespons di http://localhost:8501 (bisa sampai ~100 detik saat cold-start)...")
import urllib.request as _urlreq
streamlit_ready = False
died_early = False
for i in range(100):
    if streamlit_proc.poll() is not None:
        died_early = True
        break  # proses sudah mati duluan, tidak perlu tunggu lagi
    try:
        _urlreq.urlopen("http://localhost:8501", timeout=2)
        streamlit_ready = True
        break
    except Exception:
        if i > 0 and i % 15 == 0:
            print(f"      ...masih menunggu ({i} detik berlalu, proses masih hidup, kemungkinan masih loading)")
        time.sleep(1)

if not streamlit_ready:
    print("\n" + "=" * 60)
    if died_early:
        print("❌ STREAMLIT CRASH (proses berhenti sendiri) -- inilah akar masalah 'Bad Gateway' selama ini.")
    else:
        print("❌ STREAMLIT TIDAK KUNJUNG SIAP dalam 100 detik (proses masih hidup, tapi tidak merespons).")
    print("   Cloudflare Tunnel TIDAK akan dibuka karena tidak ada yang bisa disambungkan.")
    print("   Isi log Streamlit (alasan aslinya biasanya ada di baris-baris terakhir ini):")
    print("-" * 60)
    os.system(f"tail -n 60 {STREAMLIT_LOG}")
    print("-" * 60)
    print("   -> Salin/screenshot bagian log di atas dan kirimkan, supaya bisa didiagnosis pasti.")
    raise SystemExit(
        "Streamlit tidak merespons di port 8501 -- lihat log di atas untuk penyebab aslinya."
    )

print("   ✅ Streamlit sudah jalan & merespons.")

# 5. Siapkan & jalankan Cloudflare Tunnel
print("\n⏳ [5/5] Menyiapkan Cloudflare Tunnel...")
if not os.path.exists("./cloudflared"):
    os.system(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared"
    )
    os.system("chmod +x cloudflared")

tunnel = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(5)

print("\n⏳ Menunggu link tunnel muncul...")
link_found = False
for _ in range(40):
    line = tunnel.stderr.readline().decode("utf-8")
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        print("\n" + "=" * 60)
        print("🎉 DASHBOARD SIAP DIBUKA — KLIK LINK BERIKUT:")
        print(f"👉 {match.group(0)}")
        print("=" * 60 + "\n")
        print("   PENTING: link ini BARU dan HANYA berlaku untuk sesi ini.")
        print("   Kalau nanti muncul 'Bad Gateway' lagi, itu tandanya sesi Colab")
        print("   sudah terputus/berubah -- jangan buka link LAMA, jalankan cell")
        print("   ini lagi dari awal untuk dapat link BARU.")
        link_found = True
        break

if not link_found:
    print("\n⚠️ Link belum muncul dalam waktu tunggu. Streamlit sudah jalan (dicek di atas),")
    print("   jadi kemungkinan besar cloudflared-nya yang gagal konek keluar. Cek dengan:")
    print("   !cat /content/streamlit_log.txt  # log Streamlit")
    print("   atau jalankan ulang cell ini dari awal.")
