# Dashboard Evaluasi Kelayakan Konduktor ACSR Ex-Reconductoring

Dashboard Streamlit dwibahasa (ID/EN) untuk mengevaluasi kelayakan pakai
konduktor ACSR (Aluminium Conductor Steel Reinforced) pasca-reconductoring —
riset Tugas Akhir kolaborasi PT PLN (Persero) & Universitas Diponegoro.
Menggabungkan NDT X-Ray (klasifikasi cacat via ConvNeXt-Tiny), Mikrografi
korosi (ResNet18 + ASTM Grain Size), dan uji mekanis (Tarik/Torsi/Lilit).

## Struktur project
```
acsr_work/
├── app.py                   <- logika & tampilan utama
├── requirements.txt
├── .gitattributes           <- menandai file .pth supaya lewat Git LFS di GitHub
├── JALANKAN_SATU_KLIK.py    <- HANYA dipakai kalau run via Google Colab (opsional)
├── utils/
│   ├── ui.py                <- tema, CSS, komponen visual
│   └── scoring.py           <- skor 0-100 (VISUAL saja, tidak menentukan keputusan)
├── models/                  <- taruh 2 file .pth model AI di sini (lihat models/README.md)
└── assets/                  <- logo dll (opsional)
```

## Cara akses permanen (GitHub + Streamlit Community Cloud) — direkomendasikan

Dashboard ini didesain supaya bisa diakses semua orang lewat 1 link tetap,
tanpa perlu buka Colab & klik jalankan setiap kali. Gratis & TIDAK perlu
menulis ulang kode apapun (dashboard ini sudah Streamlit native — Streamlit
Community Cloud dibuat khusus oleh pembuat Streamlit sendiri utk hosting app
seperti ini). Panduan lengkap langkah-demi-langkah ada di file docx terpisah
yang dikirim bersama zip ini — ringkasannya:

1. Download 2 file model dari Google Drive kamu, taruh di folder `models/`
   (nama file harus persis `cnn_kabel_model.pth` & `cnn_mikro_korosi_model.pth`
   — lihat `models/README.md`).
2. Install **GitHub Desktop** (aplikasi gratis, tidak perlu jago command
   line) + jalankan SATU perintah `git lfs install` di terminal (cuma sekali
   seumur hidup di komputer itu — file .pth ~106MB & ~45MB butuh Git LFS
   krn GitHub menolak file tunggal di atas 100MB tanpa LFS).
3. Buat repository baru di github.com (Public), clone lewat GitHub Desktop,
   salin SEMUA isi folder `acsr_work/` ini ke dalamnya (termasuk `models/`
   yang sudah berisi 2 file model), lalu commit & push lewat GitHub Desktop.
4. Buka share.streamlit.io, login pakai akun GitHub, klik "New app", pilih
   repo ini, isi "Main file path" dengan `app.py`, klik Deploy.
5. Tunggu build (~beberapa menit) — link permanennya otomatis dibuat, mis.
   `https://nama-app-kamu.streamlit.app`.

Catatan: kuota gratis Git LFS di GitHub adalah 10GB penyimpanan + 10GB
bandwidth/bulan — jauh lebih dari cukup utk 2 file model (~150MB total).
App gratis Streamlit Cloud dibatasi RAM ±2.7GB (lebih kecil dari Colab) —
cukup utk inferensi 1 citra per klik seperti dashboard ini, tapi kalau nanti
terasa lambat/crash karena memori, itu batasan tier gratisnya, bukan bug.

## Cara jalankan sementara via Google Colab (opsional, untuk pengembangan)

1. Upload zip project ini ke Colab (jangan diekstrak dulu).
2. Buka `JALANKAN_SATU_KLIK.py`, copy semua isinya ke satu cell kosong Colab.
3. Sesuaikan `ZIP_NAME`/`PROJECT_DIR` di bagian atas file sesuai nama zip yang
   diupload, lalu jalankan cell (Shift+Enter) — otomatis mengekstrak project,
   mount Google Drive (mencari 2 file model di sana), install dependency,
   menjalankan Streamlit, dan membuka Cloudflare Tunnel.
4. Klik link `https://xxxxx.trycloudflare.com` yang muncul di output.
   **Link ini SEMENTARA** (mati begitu sesi Colab berakhir) — untuk akses
   permanen semua orang, pakai cara GitHub + Streamlit Community Cloud di atas.

## Cara mengubah tampilan
Semua styling ada di `utils/ui.py`. Setelah edit, commit & push ulang lewat
GitHub Desktop (Streamlit Cloud akan otomatis build ulang & update live) —
atau jalankan ulang `JALANKAN_SATU_KLIK.py` kalau masih pakai Colab.

## Aturan permanen (jangan diubah)
Keputusan LAYAK PAKAI / AFKIR selalu dari logika boolean
`final_pass = ndt_lolos OR (al_ok AND steel_ok AND tor_ok)` di `app.py` —
skor 0-100 (donut/bar chart Beranda) murni tampilan visual, TIDAK PERNAH
menentukan keputusan akhir.
