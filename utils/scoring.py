"""Perhitungan skor kelayakan ACSR: sub-skor 0-100 per parameter dan skor
agregat berbobot, untuk PRESENTASI/VISUALISASI di halaman Beranda saja.

PENTING -- pemisahan tanggung jawab yang disengaja:
Skor 0-100 di modul ini TIDAK dipakai untuk menentukan keputusan akhir
LAYAK/TIDAK LAYAK -- itu tetap murni logika boolean di app.py
(`final_pass = ndt_lolos or (al_ok and steel_ok and tor_ok)`), sesuai draft
Tugas Akhir yang secara sengaja menghapus kritik soal "pembobotan parameter
mesin keputusan" dari Bab IV penutup & Bab V. Skor agregat di sini murni
alat bantu visual (dashboard summary, donut, bar chart) supaya pengguna
dashboard bisa melihat seberapa jauh/dekat tiap parameter dari ambang
standarnya -- bukan pengganti mesin keputusan.
"""

AMBANG_DEFAULT = 70.0

# Bobot tiap kriteria (persen) -- HANYA dipakai untuk skor agregat visual,
# tidak memengaruhi keputusan LAYAK/TIDAK LAYAK yang tetap dari final_pass.
BOBOT_DEFAULT = {
    "ndt": 25,
    "mikro": 15,
    "tarik_al": 20,
    "tarik_steel": 20,
    "torsi": 10,
    "lilit": 10,
}

LABEL_BOBOT = {
    "ndt": "NDT X-Ray",
    "mikro": "Mikrografi",
    "tarik_al": "Tarik alumunium",
    "tarik_steel": "Tarik inti baja",
    "torsi": "Torsi puntir",
    "lilit": "Uji lilitan",
}


def clamp(nilai, bawah=0.0, atas=100.0):
    return max(bawah, min(atas, nilai))


def ratio_score(actual, minimum, ambang=AMBANG_DEFAULT):
    """Ubah nilai uji jadi skor 0-100 relatif terhadap ambang standarnya.

    Tepat di ambang minimum -> skor = `ambang` (default 70). Margin +10% ->
    100, kekurangan 10% -> 40. Skala ini dipilih supaya satu garis ambang
    (70) berlaku untuk semua parameter walau satuannya berbeda-beda.
    """
    if minimum <= 0:
        return 0.0
    return clamp(ambang + (actual / minimum - 1.0) * 300.0)


def ndt_score(kelas, confidence):
    """Sub-skor NDT: confidence dipakai apa adanya kalau kelas NORMAL,
    dan dibalik kalau model justru yakin ada cacat. `confidence` boleh
    pecahan 0-1 atau sudah dalam persen (0-100), dua-duanya diterima."""
    conf_pct = confidence * 100.0 if confidence <= 1.0 else float(confidence)
    if "NORMAL" in str(kelas).upper():
        return clamp(conf_pct)
    return clamp(100.0 - conf_pct)


def mikro_score(is_mulus, astm_g, ambang=AMBANG_DEFAULT):
    """Sub-skor mikrografi dari klasifikasi permukaan + ukuran butir ASTM."""
    if not is_mulus:
        return 22.0
    if astm_g is None:
        return 78.0  # permukaan baik, grain size belum dihitung
    return clamp(ratio_score(astm_g, 6.0, ambang) * 0.6 + 40.0)


def aggregate(subscores, bobot=None):
    """Skor agregat berbobot dari dict sub-skor {kunci: skor 0-100}."""
    bobot = bobot or BOBOT_DEFAULT
    total_bobot = sum(bobot.get(k, 0) for k in subscores) or 1
    total = sum(subscores[k] * bobot.get(k, 0) for k in subscores)
    return clamp(total / total_bobot)


def status_color(skor, ambang=AMBANG_DEFAULT, palette=None):
    """palette: dict opsional {"pass_kuat","pass","warn","fail"} -- kalau
    tidak diisi pakai warna tema Navy/PLN/Gold proyek ini (lihat utils/ui.py)."""
    palette = palette or {"pass_kuat": "#0093DD", "pass": "#1E7F4E", "warn": "#C77E12", "fail": "#C8443C"}
    if skor >= ambang + 15:
        return palette["pass_kuat"]
    if skor >= ambang:
        return palette["pass"]
    if skor >= ambang - 20:
        return palette["warn"]
    return palette["fail"]


def status_label(skor, ambang=AMBANG_DEFAULT):
    if skor >= ambang + 15:
        return "Layak"
    if skor >= ambang:
        return "Layak, pantau"
    if skor >= ambang - 20:
        return "Perlu perbaikan"
    return "Tidak layak"
