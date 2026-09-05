# Folder `models/`

Taruh 2 file bobot model AI di sini SEBELUM di-push ke GitHub. Nama file
HARUS PERSIS seperti ini (app.py mencarinya dengan nama pasti ini):

- `cnn_kabel_model.pth` — model NDT X-Ray (ConvNeXt-Tiny, ~106 MB)
- `cnn_mikro_korosi_model.pth` — model Mikrografi Korosi (ResNet18, ~45 MB)

File ini SEKARANG masih ada di Google Drive kamu (dipakai selama ini lewat
Colab). Download dulu kedua file itu dari Drive ke komputer kamu, taruh di
folder ini (`models/`), baru push folder `acsr_work` ini (semuanya, termasuk
folder `models/` yang sudah berisi 2 file di atas) ke repository GitHub kamu
lewat GitHub Desktop.

PENTING soal ukuran file: kedua file di atas melebihi 100MB (batas GitHub
tanpa Git LFS), jadi WAJIB lewat Git LFS -- `.gitattributes` di root project
ini SUDAH menandai semua file `*.pth` supaya otomatis lewat Git LFS begitu
kamu commit & push (asal `git lfs install` sudah pernah dijalankan sekali di
komputer kamu -- lihat panduan deployment). Kalau lupa jalankan `git lfs
install` dulu, push kemungkinan akan ditolak GitHub krn ukuran file terlalu
besar.

Kalau folder ini kosong (tanpa 2 file di atas), app.py otomatis akan mencoba
mencari di Google Drive seperti biasa (cara lama, cuma jalan kalau di-run
lewat Colab) -- jadi meng-upload project ini TANPA isi folder `models/` tidak
akan error saat push, tapi dashboard akan menampilkan status model
"Not Found" begitu dijalankan di Streamlit Community Cloud (karena di sana
tidak ada Google Drive sama sekali).
