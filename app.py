import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from datetime import datetime, date

from utils.ui import (
    apply_custom_css, render_sidebar, render_header, header_tabs, gauge_chart, ring_chart, xray_gauge_card,
    class_chip_row, donut_score, bar_parameter_scores,
    step_card, info_tile, result_box, result_box_detailed,
    framed_image, upload_header, sidebar_collapse_toggle, language_toggle, nav_button, mini_score_bar, plain_verdict,
    PASS as PASS_COLOR, FAIL as FAIL_COLOR, WARN as WARN_COLOR,
    GOLD as GOLD_COLOR, PLN_BLUE as PLN_BLUE_COLOR,
)
from utils.scoring import (
    AMBANG_DEFAULT, BOBOT_DEFAULT, LABEL_BOBOT, ratio_score, ndt_score, mikro_score, aggregate,
    status_color as skor_status_color,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

st.set_page_config(page_title="ACSR Evaluation Dashboard", page_icon="⚡", layout="wide")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_in_drive(pattern, fallback="", local_pattern=None):
    """Cari file model/aset dgn urutan prioritas (v80 -- deployment permanen):
    1) folder LOKAL relatif ke app.py (glob `local_pattern`, kalau diisi) --
       dipakai saat file diupload LANGSUNG ke repo hosting permanen (misal
       folder 'models/'/'assets/' di hosting permanen spt Streamlit Community
       Cloud). Ditambahkan supaya
       app.py yang SAMA PERSIS bisa jalan baik di Colab (via Drive, method
       lama) MAUPUN di hosting permanen tanpa Google Drive sama sekali --
       tidak perlu 2 versi app.py terpisah.
    2) Google Drive (glob `pattern`, method LAMA -- TIDAK diubah sama sekali)
       -- kalau local_pattern tidak diisi/tidak ketemu, tetap jalan seperti
       sebelumnya. Di lingkungan non-Colab, '/content/drive' tidak pernah ada
       jadi glob ini otomatis kosong (bukan error) -- aman dibiarkan.
    Return path pertama yang match & terbukti ada di disk, atau fallback."""
    if local_pattern:
        local_matches = glob.glob(os.path.join(BASE_DIR, local_pattern), recursive=True)
        if local_matches:
            return local_matches[0]
    drive_matches = glob.glob(pattern, recursive=True)
    return drive_matches[0] if drive_matches else fallback


# >>> SESUAIKAN pola pencarian ini kalau nama file model/logo kamu berbeda <<<
# local_pattern dicek LEBIH DULU (folder models/ & assets/ di sebelah app.py)
# -- lihat docstring find_in_drive() di atas.
MODEL_FILE = find_in_drive('/content/drive/**/cnn_kabel_model.pth', "model_not_found.pth",
                            local_pattern="models/cnn_kabel_model.pth")
LOGO_FILE = find_in_drive('/content/drive/**/logo_greener*', "",
                           local_pattern="assets/logo_greener*")

# ---------------------------------------------------------------------------
# BAHASA (ID/EN) -- kamus terjemahan untuk teks UTAMA (nav, header, label
# kunci). Kalimat alasan otomatis (hasil evaluasi dinamis) TETAP bahasa
# Indonesia untuk saat ini -- itu dirangkai dari banyak variabel & butuh
# usaha terpisah untuk diterjemahkan penuh.
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    # v75: dikembalikan lagi ke "Beranda"/"Home" (v74 sempat diganti
    # "Ringkasan"/"Summary" mengikuti referensi, tapi user minta balik ke
    # "Beranda" saja) -- key internal "nav_beranda" & routing value
    # menu=="beranda" TIDAK diubah.
    "nav_beranda": {"id": "Beranda", "en": "Home"},
    "nav_ndt": {"id": "NDT (X-Ray)", "en": "NDT (X-Ray)"},
    "nav_dt": {"id": "DT (Mekanis & Mikro)", "en": "DT (Mechanical & Micro)"},
    "nav_keputusan": {"id": "Keputusan & PDF", "en": "Decision & PDF"},
    "nav_riwayat": {"id": "Riwayat Pengujian", "en": "Test History"},
    "sidebar_nav_label": {"id": "NAVIGASI", "en": "NAVIGATION"},
    "sidebar_spec_label": {"id": "⚙️ Parameter Spesifikasi ACSR", "en": "⚙️ ACSR Specification Parameters"},
    "sidebar_spec_select": {"id": "Pilih Tipe Konduktor ACSR:", "en": "Select ACSR Conductor Type:"},
    # Emoji "⚡" SENGAJA dihapus (v74) -- judul ini sekarang jadi judul KONSTAN
    # di header utama untuk SEMUA halaman (dulu cuma judul Beranda), dan
    # permintaan user sebelumnya: tanpa ikon/emoji biar tampak profesional.
    "header_beranda_title": {"id": "EVALUASI KELAYAKAN KONDUKTOR ACSR EX-RECONDUCTORING",
                              "en": "ACSR EX-RECONDUCTORING CONDUCTOR FEASIBILITY EVALUATION DASHBOARD"},
    "header_ndt_title": {"id": "🔍 Inspeksi NDT (X-Ray)", "en": "🔍 NDT Inspection (X-Ray)"},
    "header_ndt_sub": {"id": "Klasifikasi citra radiografi konduktor menggunakan model ConvNeXt-Tiny",
                        "en": "Conductor radiograph image classification using ConvNeXt-Tiny model"},
    "header_dt_title": {"id": "🧪 Destructive Testing (DT)", "en": "🧪 Destructive Testing (DT)"},
    "header_dt_sub": {"id": "Mikrografi (ASTM E112) · Uji Tarik (ASTM E8/B498) · Torsi & Lilit (IEC 60888)",
                       "en": "Micrography (ASTM E112) · Tensile Test (ASTM E8/B498) · Torsion & Wrap (IEC 60888)"},
    "header_keputusan_title": {"id": "📄 Keputusan & Sertifikat PDF", "en": "📄 Decision & PDF Certificate"},
    "header_keputusan_sub": {"id": "Matriks keputusan gabungan NDT + DT dan penerbitan sertifikat validasi resmi",
                              "en": "Combined NDT + DT decision matrix and official validation certificate issuance"},
    "header_riwayat_title": {"id": "🕘 Riwayat Pengujian", "en": "🕘 Test History"},
    "header_riwayat_sub": {"id": "Daftar hasil pengujian yang telah disimpan selama sesi ini",
                            "en": "List of test results saved during this session"},
    "page_title_beranda": {"id": "EVALUASI KELAYAKAN KABEL ACSR EX-RECONDUCTORING",
                            "en": "ACSR EX-RECONDUCTORING CABLE FEASIBILITY EVALUATION"},
    "page_subtitle_beranda": {"id": "Kolaborasi Riset PT PLN (Persero) x Universitas Diponegoro",
                               "en": "Research Collaboration between PT PLN (Persero) x Universitas Diponegoro"},

    # --- Tab & label modul DT ---
    # Label tab DT sengaja TANPA ikon & TANPA kata "Uji"/"Test" -- permintaan
    # user supaya tampak lebih ringkas & profesional (dulu "🔬 Mikrografi",
    # "📐 Uji Tarik (Al vs Baja)", dst.)
    "tab_mikrografi": {"id": "Mikrografi", "en": "Micrography"},
    # v76: disederhanakan lagi jadi "Tensile" saja (dulu "Tarik (Al vs Baja)"/
    # "Tensile (Al vs Steel)") -- permintaan user, ID & EN sekarang SAMA.
    "tab_tarik": {"id": "Tensile", "en": "Tensile"},
    "tab_torsi": {"id": "Torsi Puntir", "en": "Torsion"},
    "tab_lilit": {"id": "Lilit", "en": "Wrap"},
    "tab_cek_satu": {"id": "🖼️ Cek Satu Citra", "en": "🖼️ Check Single Image"},
    "tab_cek_batch": {"id": "📦 Cek Batch (Banyak Citra Sekaligus)", "en": "📦 Batch Check (Multiple Images)"},

    "upload_xray_title": {"id": "Upload Citra X-Ray", "en": "Upload X-Ray Image"},
    "upload_xray_sub": {"id": "Format JPG, PNG, TIF — maksimal 200MB per file",
                         "en": "JPG, PNG, TIF format — max 200MB per file"},
    "upload_mikro_title": {"id": "Upload Citra Mikrografi", "en": "Upload Micrograph Image"},
    "upload_mikro_sub": {"id": "Format JPG, PNG — citra hasil pengamatan mikroskop",
                          "en": "JPG, PNG format — microscope observation image"},
    "upload_batch_title": {"id": "Upload Banyak Citra Sekaligus", "en": "Upload Multiple Images at Once"},
    "upload_batch_sub": {"id": "Pilih beberapa file JPG/PNG/JPEG untuk diperiksa satu per satu",
                          "en": "Select several JPG/PNG/JPEG files to be checked one by one"},

    "info_upload_xray_prompt": {"id": "Unggah citra X-Ray di sisi kiri untuk memulai inspeksi.",
                                 "en": "Upload an X-Ray image on the left to start the inspection."},
    "info_upload_mikro_prompt": {"id": "Unggah citra mikrografi untuk analisis otomatis kondisi permukaan & grain size.",
                                  "en": "Upload a micrograph image for automatic surface condition & grain size analysis."},
    "mag_label": {"id": "Perbesaran mikroskop yang dipakai saat pengambilan citra (x):",
                  "en": "Microscope magnification used when capturing the image (x):"},
    "mag_help": {"id": "Gunakan 50x khusus untuk evaluasi kondisi permukaan (butir belum terlihat jelas di perbesaran ini, "
                        "jadi grain size TIDAK dihitung otomatis). Perbesaran 200x-1000x untuk perhitungan grain size.",
                 "en": "Use 50x specifically for surface condition evaluation (grains aren't clearly visible at this "
                       "magnification, so grain size is NOT auto-calculated). Use 200x-1000x for grain size calculation."},

    "section_al": {"id": "A. Kawat Alumunium (Outer Strands)", "en": "A. Aluminium Wire (Outer Strands)"},
    "section_steel": {"id": "B. Kawat Inti Baja (Steel Core)", "en": "B. Steel Core Wire"},
    "label_tensile_al": {"id": "Tensile Al (MPa):", "en": "Tensile Al (MPa):"},
    "label_elong_al": {"id": "Elongasi Al (%):", "en": "Elongation Al (%):"},
    "label_tensile_steel": {"id": "Tensile Steel (MPa):", "en": "Tensile Steel (MPa):"},
    "label_elong_steel": {"id": "Elongasi Steel (%):", "en": "Elongation Steel (%):"},
    "label_torsi": {"id": "Momen Torsi (N.m):", "en": "Torque Moment (N.m):"},
    "label_lilit": {"id": "Jumlah Putaran (Turns):", "en": "Number of Turns:"},
    "btn_grafik_tarik": {"id": "📊 Tampilkan Grafik Uji Tarik", "en": "📊 Show Tensile Test Chart"},
    "btn_grafik_torsi": {"id": "📊 Tampilkan Grafik Torsi", "en": "📊 Show Torsion Chart"},
    "btn_grafik_lilit": {"id": "📊 Tampilkan Grafik Lilitan", "en": "📊 Show Wrap Chart"},
    "eval_al_title": {"id": "Evaluasi Kawat Alumunium", "en": "Aluminium Wire Evaluation"},
    "eval_steel_title": {"id": "Evaluasi Kawat Inti Baja", "en": "Steel Core Wire Evaluation"},
    "eval_torsi_title": {"id": "Evaluasi Torsi Puntir", "en": "Torsion Evaluation"},
    "eval_lilit_title": {"id": "Evaluasi Uji Lilitan", "en": "Wrap Test Evaluation"},

    "koreksi_manual": {"id": "Koreksi manual (jika hasil auto perlu disesuaikan)",
                        "en": "Manual correction (if the automatic result needs adjusting)"},
    "status_korosi_label": {"id": "Status Korosi Permukaan:", "en": "Surface Corrosion Status:"},
    "label_G_manual": {"id": "Ukuran Butir ASTM (G):", "en": "ASTM Grain Size (G):"},
    "glossary_ndt_title": {"id": "📖 Kamus Istilah — Kondisi Konduktor Hasil Inspeksi NDT",
                            "en": "📖 Glossary — Conductor Condition Terms from NDT Inspection"},
    "glossary_mech_title": {"id": "🔩 Kamus Istilah — Pengujian Mekanis (DT)",
                             "en": "🔩 Glossary — Mechanical Testing Terms (DT)"},
    "glossary_caption": {"id": "Klik tiap istilah untuk melihat penjelasannya.",
                          "en": "Click each term to see its explanation."},
    "desc_unavailable": {"id": "Deskripsi kelas belum tersedia — tambahkan di kamus NDT_CLASS_DESC pada app.py.",
                          "en": "Class description not available yet — add it to the NDT_CLASS_DESC dictionary in app.py."},
    "keterangan_kondisi": {"id": "Keterangan Kondisi", "en": "Condition Details"},
    "model_not_found_title": {"id": "Model tidak ditemukan", "en": "Model not found"},
    "model_not_found_body": {"id": "Pastikan file `cnn_kabel_model.pth` ada di Google Drive dan path pencarian glob sudah benar.",
                              "en": "Make sure the `cnn_kabel_model.pth` file exists in Google Drive and the glob search path is correct."},
    "status_model_label": {"id": "Status Model", "en": "Model Status"},
    "akurasi_model_label": {"id": "Akurasi Validasi Model", "en": "Model Validation Accuracy"},
    "batch_size_label": {"id": "Batch Size Training", "en": "Training Batch Size"},
}

# Terjemahan satuan/kata kunci yang dipakai berulang di kalimat alasan dinamis
UNIT_WORDS = {
    "memenuhi": {"id": "memenuhi", "en": "meets"},
    "tidak_memenuhi": {"id": "TIDAK memenuhi", "en": "does NOT meet"},
    "margin": {"id": "margin", "en": "margin"},
    "kekurangan": {"id": "kekurangan", "en": "shortfall"},
    "tercatat": {"id": "tercatat", "en": "recorded at"},
    "ambang_minimum": {"id": "ambang minimum", "en": "minimum threshold"},
}


def t(key):
    """Ambil teks sesuai bahasa aktif di session_state['lang'] ('id'/'en')."""
    lang = st.session_state.get('lang', 'id')
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get('id', key))


def tt(id_text, en_text):
    """Terjemahan sekali-pakai untuk teks yang sudah dirakit dengan variabel
    (f-string dinamis: nama file, angka hasil, dsb) -- alternatif ringkas dari
    t() untuk kasus yang tidak praktis didaftarkan sebagai key tetap di
    TRANSLATIONS (karena isinya beda setiap render). Dipakai di SELURUH
    halaman supaya toggle ID/EN benar-benar konsisten, bukan cuma sebagian
    label seperti versi sebelumnya."""
    return en_text if st.session_state.get('lang', 'id') == 'en' else id_text


if 'lang' not in st.session_state:
    st.session_state['lang'] = 'id'

if 'sidebar_compact' not in st.session_state:
    st.session_state['sidebar_compact'] = False

# Riwayat Pengujian -- SENGAJA hanya di st.session_state (tidak ditulis ke
# file/CSV/Drive), sesuai keputusan eksplisit user: riwayat ini sementara
# per sesi saja, otomatis hilang saat halaman di-reload atau runtime Colab
# di-restart. Tiap entri adalah snapshot dict yang dibuat lewat tombol
# "Simpan Pengujian" di header utama (v74, dulu di halaman Keputusan --
# lihat blok header dekat sidebar_nav_list, dan aksinya dekat MARGIN_TERKECIL).
if 'riwayat_list' not in st.session_state:
    st.session_state['riwayat_list'] = []

# Data spesimen -- BARU di v74 (dulu tidak ada field ini sama sekali).
# Diedit lewat panel "Data Pengujian" di header (lihat blok header di bawah),
# ditampilkan sebagai badge/subtitle di header, dan ikut disertakan ke
# snapshot Riwayat Pengujian. Sama seperti riwayat_list, ini SESSION-STATE
# SAJA (tidak ditulis ke file/Drive).
if 'spesimen_kode' not in st.session_state:
    st.session_state['spesimen_kode'] = "ACSR-2026-001"
if 'spesimen_haspel' not in st.session_state:
    st.session_state['spesimen_haspel'] = "H-001"
if 'spesimen_panjang' not in st.session_state:
    st.session_state['spesimen_panjang'] = 250.0
if 'spesimen_tanggal' not in st.session_state:
    st.session_state['spesimen_tanggal'] = date.today()
if 'spesimen_tim' not in st.session_state:
    st.session_state['spesimen_tim'] = "Tim Lab UNDIP"
if 'show_data_pengujian' not in st.session_state:
    st.session_state['show_data_pengujian'] = False

# Navigasi halaman (v74): dipindah dari baris tab header ke daftar tombol
# vertikal di sidebar (permintaan user, "lebih rapi") -- di-init & dibaca DI
# SINI (lebih awal dari sebelumnya) supaya sidebar_nav_list di bawah tahu
# tombol mana yang aktif SEBELUM dirender.
if 'active_menu' not in st.session_state:
    st.session_state['active_menu'] = "beranda"
menu = st.session_state['active_menu']

apply_custom_css(compact=st.session_state['sidebar_compact'])
if sidebar_collapse_toggle():
    st.session_state['sidebar_compact'] = not st.session_state['sidebar_compact']
    st.rerun()
render_sidebar(LOGO_FILE, compact=st.session_state['sidebar_compact'])

# Daftar nav sidebar (v74) -- menggantikan header_tabs(). Dibungkus
# st.container(key="sidebar_nav_list") supaya CSS-nya TERISOLASI (lihat
# apply_custom_css), tidak ikut menghajar tombol sidebar lain seperti
# tombol ciutkan. Label pakai t() yang SAMA dengan yang dulu dipakai
# header_tabs() (nav_beranda/nav_ndt/dst.), TANPA ikon saat sidebar normal
# (permintaan user v72: tampilan profesional). v75: tiap item juga dikasih
# `icon` (emoji, dipakai SATU set yang sama dgn ikon yang sudah ada di
# tempat lain aplikasi ini -- 🩻 NDT, 🧪 DT, 📄 Keputusan, 🕘 Riwayat, 🏠
# Beranda) -- HANYA ditampilkan sebagai isi tombol saat sidebar_compact=True
# (lihat nav_button()), supaya nav diciutkan jadi kotak ikon rapi, BUKAN
# teks kecil yang wrap jadi tumpukan huruf per baris (bug v74, laporan user).
_NAV_ITEMS = [
    ("beranda", t("nav_beranda"), "🏠"), ("ndt", t("nav_ndt"), "🩻"), ("dt", t("nav_dt"), "🧪"),
    ("keputusan", t("nav_keputusan"), "📄"), ("riwayat", t("nav_riwayat"), "🕘"),
]
with st.sidebar:
    with st.container(key="sidebar_nav_list"):
        for _nav_key, _nav_label, _nav_icon in _NAV_ITEMS:
            if nav_button(_nav_label, active=(menu == _nav_key), key=f"navbtn_{_nav_key}",
                          compact=st.session_state['sidebar_compact'], icon=_nav_icon):
                st.session_state['active_menu'] = _nav_key
                st.rerun()

ACSR_DATABASE = {
    "ACSR 70/12 mm² (Hawk Spec)": {
        "min_tensile_al": 165.0, "min_elong_al": 1.5,
        "min_tensile_steel": 1250.0, "min_elong_steel": 3.0,
        "min_torsi": 12.0, "min_lilit": 16,
        # >>> VERIFIKASI nilai diameter & standar ini terhadap datasheet resmi kamu <<<
        "diameter_mm": 11.4, "jumlah_kawat": "6 Al / 1 Baja",
        "standar_acuan": "ASTM B232 (ACSR), ASTM E8/B498 (Tarik), ASTM E112 (Mikrografi), IEC 60888 (Torsi/Lilit)",
    },
    "ACSR 150/25 mm² (Penguin Spec)": {
        "min_tensile_al": 170.0, "min_elong_al": 1.6,
        "min_tensile_steel": 1300.0, "min_elong_steel": 3.5,
        "min_torsi": 14.0, "min_lilit": 18,
        "diameter_mm": 15.6, "jumlah_kawat": "26 Al / 7 Baja",
        "standar_acuan": "ASTM B232 (ACSR), ASTM E8/B498 (Tarik), ASTM E112 (Mikrografi), IEC 60888 (Torsi/Lilit)",
    },
    "ACSR 240/40 mm² (Zebra Spec)": {
        "min_tensile_al": 175.0, "min_elong_al": 1.8,
        "min_tensile_steel": 1350.0, "min_elong_steel": 4.0,
        "min_torsi": 16.0, "min_lilit": 20,
        "diameter_mm": 21.0, "jumlah_kawat": "54 Al / 7 Baja",
        "standar_acuan": "ASTM B232 (ACSR), ASTM E8/B498 (Tarik), ASTM E112 (Mikrografi), IEC 60888 (Torsi/Lilit)",
    }
}

# >>> Penjelasan tiap kelas prediksi NDT — SESUAIKAN kalau nama kelas model kamu berbeda <<<
NDT_CLASS_DESC = {
    "KONDUKTOR NORMAL": {
        "id": "Kondisi konduktor baik, tidak ditemukan indikasi cacat struktural pada citra radiografi. Konduktor tergolong layak dipertimbangkan untuk dioperasikan kembali.",
        "en": "Conductor condition is good, no structural defect indication found in the radiograph image. The conductor is considered suitable for re-operation."},
    "KONDUKTOR BENDING": {
        "id": "Konduktor terdeteksi mengalami pembengkokan (bending) pada bagian tertentu. Kondisi ini berpotensi menimbulkan konsentrasi tegangan pada titik tekuk dan melemahkan struktur mekanis secara lokal.",
        "en": "The conductor is detected to have bending in certain sections. This condition may cause stress concentration at the bend point and locally weaken the mechanical structure."},
    "KONDUKTOR MEKAR": {
        "id": "Kawat-kawat penyusun konduktor terdeteksi mekar/terurai dari lilitan utama (strand loosening). Umumnya diakibatkan oleh gaya tarik berlebih, abrasi, atau kegagalan proses reconductoring sebelumnya.",
        "en": "The conductor's constituent wires are detected to be loosened/unwound from the main strand (strand loosening). Usually caused by excessive tensile force, abrasion, or a previous reconductoring process failure."},
    "KONDUKTOR MULUR": {
        "id": "Konduktor terdeteksi mengalami mulur (elongasi permanen di luar batas elastis material). Kondisi ini mengindikasikan material sempat mengalami beban yang melampaui batas luluhnya (yield point).",
        "en": "The conductor is detected to have crept (permanent elongation beyond the material's elastic limit). This condition indicates the material has been subjected to a load exceeding its yield point."},
    "KONDUKTOR PUTUS": {
        "id": "Konduktor terdeteksi mengalami putus pada satu atau lebih kawat penyusun. Ini adalah kondisi kegagalan struktural paling kritis dan konduktor harus dinyatakan tidak layak pakai.",
        "en": "The conductor is detected to have broken in one or more of its constituent wires. This is the most critical structural failure condition and the conductor must be declared unfit for use."},
}

# Ikon berbeda tiap kondisi -- supaya bisa dikenali sekilas tanpa baca teks dulu
NDT_CLASS_ICON = {
    "KONDUKTOR NORMAL": "✅",
    "KONDUKTOR BENDING": "🔀",
    "KONDUKTOR MEKAR": "🎗️",
    "KONDUKTOR MULUR": "📏",
    "KONDUKTOR PUTUS": "💥",
}

# >>> Kamus istilah pengujian mekanis (DT) — SESUAIKAN/lengkapi kalau perlu <<<
# Sekarang dwibahasa (id/en) supaya kamus istilah ikut berubah saat toggle EN
# -- sebelumnya cuma Indonesia, jadi permanen Indonesia walau bahasa=EN.
MECH_TERMS_DESC = {
    "Uji Tarik (Tensile Test)": {
        "id": "Pengujian untuk mengetahui kekuatan maksimum kawat menahan beban tarik sebelum putus (Ultimate Tensile Strength/UTS), mengacu ASTM E8 (umum) dan ASTM B498 (khusus kawat baja galvanis inti ACSR).",
        "en": "A test to determine the maximum strength of a wire in withstanding tensile load before breaking (Ultimate Tensile Strength/UTS), referring to ASTM E8 (general) and ASTM B498 (specific to galvanized steel core wire of ACSR)."},
    "Elongasi (Elongation)": {
        "id": "Persentase pertambahan panjang spesimen saat putus dibanding panjang awalnya (gauge length). Mengindikasikan keuletan (ductility) material — makin tinggi elongasi, makin ulet & tidak getas.",
        "en": "The percentage increase in specimen length at break compared to its original length (gauge length). Indicates the material's ductility — the higher the elongation, the more ductile and less brittle it is."},
    "Uji Torsi Puntir (Torsion Test)": {
        "id": "Pengujian jumlah puntiran yang mampu ditahan kawat sebelum patah, mengacu IEC 60888. Mengindikasikan ketahanan kawat terhadap beban puntir saat instalasi/operasi.",
        "en": "A test of the number of twists a wire can withstand before breaking, referring to IEC 60888. Indicates the wire's resistance to torsional load during installation/operation."},
    "Uji Lilit (Wrap Test)": {
        "id": "Pengujian jumlah lilitan kawat mengelilingi mandrel/kawat lain tanpa retak, mengacu IEC 60888. Mengindikasikan fleksibilitas & ketahanan permukaan galvanis terhadap retak saat dibentuk.",
        "en": "A test of the number of wraps a wire can make around a mandrel/other wire without cracking, referring to IEC 60888. Indicates the flexibility & crack resistance of the galvanized surface when formed."},
    "Uji Mikrografi (Micrography)": {
        "id": "Pengamatan struktur mikro permukaan/penampang kawat menggunakan mikroskop, mengacu ASTM E112 untuk penentuan ukuran butir (grain size).",
        "en": "Observation of the wire surface/cross-section microstructure using a microscope, referring to ASTM E112 for grain size determination."},
    "ASTM Grain Size Number (G)": {
        "id": "Angka standar ASTM E112 yang menyatakan ukuran rata-rata butir kristal logam. Semakin besar nilai G, semakin halus (kecil) ukuran butirnya — umumnya berkorelasi dengan sifat mekanis yang lebih baik.",
        "en": "The ASTM E112 standard number expressing the average size of metal crystal grains. The higher the G value, the finer (smaller) the grain size — generally correlated with better mechanical properties."},
    "UTS (Ultimate Tensile Strength)": {
        "id": "Tegangan tarik maksimum yang mampu ditahan material sebelum mengalami kegagalan/putus, dinyatakan dalam satuan MPa (MegaPascal).",
        "en": "The maximum tensile stress a material can withstand before failure/breaking, expressed in MPa (MegaPascal) units."},
}

# Label glosarium sengaja TANPA kata "Uji"/"Test" & TANPA ikon (permintaan
# user supaya tampak lebih ringkas & profesional) -- dict MECH_TERMS_DESC di
# atas TIDAK diubah kuncinya (masih "Uji Tarik (Tensile Test)" dst., cuma
# dipakai sebagai key lookup internal), label yang ditampilkan ke pengguna
# sekarang selalu lewat MECH_TERMS_LABEL_ID/EN di bawah ini.
MECH_TERMS_LABEL_ID = {
    "Uji Tarik (Tensile Test)": "Tarik (Tensile Test)",
    "Elongasi (Elongation)": "Elongasi (Elongation)",
    "Uji Torsi Puntir (Torsion Test)": "Torsi Puntir (Torsion Test)",
    "Uji Lilit (Wrap Test)": "Lilit (Wrap Test)",
    "Uji Mikrografi (Micrography)": "Mikrografi (Micrography)",
    "ASTM Grain Size Number (G)": "ASTM Grain Size Number (G)",
    "UTS (Ultimate Tensile Strength)": "UTS (Ultimate Tensile Strength)",
}
MECH_TERMS_LABEL_EN = {
    "Uji Tarik (Tensile Test)": "Tensile",
    "Elongasi (Elongation)": "Elongation",
    "Uji Torsi Puntir (Torsion Test)": "Torsion",
    "Uji Lilit (Wrap Test)": "Wrap",
    "Uji Mikrografi (Micrography)": "Micrography",
    "ASTM Grain Size Number (G)": "ASTM Grain Size Number (G)",
    "UTS (Ultimate Tensile Strength)": "UTS (Ultimate Tensile Strength)",
}

MECH_TERMS_ICON = {
    "Uji Tarik (Tensile Test)": "📐",
    "Elongasi (Elongation)": "📏",
    "Uji Torsi Puntir (Torsion Test)": "🔄",
    "Uji Lilit (Wrap Test)": "🌀",
    "Uji Mikrografi (Micrography)": "🔬",
    "ASTM Grain Size Number (G)": "🧫",
    "UTS (Ultimate Tensile Strength)": "💪",
}

# ===========================================================================
# HEADER UTAMA (v74) -- SATU kartu navy: brand row (logo chip + "UNDIP x
# PLN" + toggle bahasa, sekarang DINAMIS lewat st.columns bukan posisi
# hardcode, ditaruh berdampingan/"pas" dengan elemen brand lain di baris
# yang sama -- permintaan user), badge kode spesimen + subtitle info
# (Haspel/panjang/tanggal/tim), judul dashboard KONSTAN untuk semua halaman
# (dulu berbeda-beda per halaman lewat PAGES dict -- dihapus), lalu baris
# interaktif selectbox tipe ACSR + tombol "Data Pengujian"/"Simpan
# Pengujian". Navigasi halaman (dulu baris tab header_tabs di sini) SUDAH
# PINDAH ke sidebar (lihat blok sidebar_nav_list di atas, permintaan user
# "lebih rapi"). Header ini SENGAJA ditaruh di sini (sebelum blok
# "PERHITUNGAN TERPUSAT" di bawah) karena selectbox di dalamnya harus
# dieksekusi duluan untuk menghasilkan `acsr_spec` yang dipakai blok itu --
# tombol "Simpan Pengujian" cuma DIRENDER di sini, aksi simpannya (butuh
# final_pass/SKOR_TOTAL yang baru tersedia setelah blok terpusat) baru
# dijalankan di bawah (cari "if _clicked_simpan_pengujian:").
# ===========================================================================
_BULAN_ID = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
             7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}


def _format_tanggal_spesimen(d):
    """Format tanggal pendek dwibahasa -- ID pakai singkatan bulan Indonesia
    (mis. "30 Agu 2026"), EN pakai singkatan bawaan Python (mis. "30 Aug
    2026"). Python strftime('%b') defaultnya selalu Inggris apapun lang UI,
    jadi versi ID di-map manual di sini."""
    if st.session_state['lang'] == 'id':
        return f"{d.day} {_BULAN_ID.get(d.month, d.strftime('%b'))} {d.year}"
    return d.strftime("%d %b %Y")


with st.container(key="main_header_card"):
    col_brand, col_lang = st.columns([3, 1], vertical_alignment="center")
    with col_brand:
        st.markdown(f"""
        <div class="header-brand-row">
            <span class="header-logo-chip" style="background:{GOLD_COLOR};"></span>
            <span class="header-logo-chip" style="background:{PLN_BLUE_COLOR};"></span>
            <span class="header-brand-text">UNDIP &times; PLN</span>
        </div>
        """, unsafe_allow_html=True)
    with col_lang:
        # Toggle bahasa dipindah dari sidebar ke SINI (kanan-atas kartu
        # header) -- selalu terjangkau dari halaman manapun (sama seperti
        # sebelumnya), tapi sekarang jadi satu baris dgn brand row lewat
        # st.columns (dinamis/responsif), bukan blok terpisah di sidebar.
        new_lang = language_toggle(st.session_state['lang'])
        if new_lang != st.session_state['lang']:
            st.session_state['lang'] = new_lang
            st.rerun()

    st.markdown(f"""
    <div class="header-spec-row">
        <span class="header-spec-badge">{st.session_state['spesimen_kode']}</span>
        <span class="header-spec-sub">{tt('Haspel', 'Reel')} {st.session_state['spesimen_haspel']} ·
        {st.session_state['spesimen_panjang']:.0f} m · {_format_tanggal_spesimen(st.session_state['spesimen_tanggal'])} ·
        {st.session_state['spesimen_tim']}</span>
    </div>
    <div class="header-main-title">{t('header_beranda_title')}</div>
    """, unsafe_allow_html=True)

    col_sel, col_data_btn, col_simpan_btn = st.columns([2.2, 1, 1])
    with col_sel:
        selected_acsr_type = st.selectbox(
            tt("Tipe Konduktor ACSR", "ACSR Conductor Type"), list(ACSR_DATABASE.keys()),
            label_visibility="collapsed", key="acsr_type_select_header",
        )
    with col_data_btn:
        if st.button(tt("Data Pengujian", "Test Data"), key="btn_toggle_data_pengujian", use_container_width=True):
            st.session_state['show_data_pengujian'] = not st.session_state['show_data_pengujian']
    with col_simpan_btn:
        # Tombol yang sama dengan "Simpan ke Riwayat Pengujian" versi v73
        # (cuma dipindah ke header supaya bisa diklik dari halaman manapun,
        # bukan cuma dari halaman Keputusan) -- KLIK dideteksi di sini,
        # AKSI simpannya dijalankan di bawah setelah final_pass/SKOR_TOTAL
        # terpusat selesai dihitung.
        _clicked_simpan_pengujian = st.button(
            tt("Simpan Pengujian", "Save Test"), key="btn_simpan_pengujian_header",
            use_container_width=True, type="primary")

acsr_spec = ACSR_DATABASE[selected_acsr_type]

if st.session_state['show_data_pengujian']:
    with st.container(border=True):
        st.markdown(f"##### {tt('Data Pengujian / Spesimen', 'Test / Specimen Data')}")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.session_state['spesimen_kode'] = st.text_input(
                tt("Kode Spesimen", "Specimen Code"), value=st.session_state['spesimen_kode'])
            st.session_state['spesimen_haspel'] = st.text_input(
                tt("ID Haspel", "Reel ID"), value=st.session_state['spesimen_haspel'])
        with dcol2:
            st.session_state['spesimen_panjang'] = st.number_input(
                tt("Panjang Kabel (m)", "Cable Length (m)"), value=float(st.session_state['spesimen_panjang']),
                min_value=0.0, step=10.0)
            st.session_state['spesimen_tanggal'] = st.date_input(
                tt("Tanggal Pengujian", "Test Date"), value=st.session_state['spesimen_tanggal'])
        with dcol3:
            st.session_state['spesimen_tim'] = st.text_input(
                tt("Tim Laboratorium", "Lab Team"), value=st.session_state['spesimen_tim'])

# ---------------------------------------------------------------------------
# SESSION STATES
# ---------------------------------------------------------------------------
defaults = {
    'ndt_status': "Belum Diuji", 'ndt_defek': "-", 'ndt_conf': 0.0,
    'korosi_status': "Belum Dianalisis", 'korosi_reason': "",
    'astm_G': 7.9, 'tensile_al': 170.0, 'elong_al': 1.8,
    'tensile_steel': 1280.0, 'elong_steel': 3.8, 'torsion': 14.5, 'turns': 18,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# HELPER: evaluasi dengan alasan tertulis
# ---------------------------------------------------------------------------
# Label parameter (dipakai di puluhan pemanggilan evaluate_reason() di seluruh
# app.py) -- diterjemahkan otomatis DI SINI (satu tempat) supaya kalimat alasan
# EN tidak lagi bercampur label Indonesia seperti sebelumnya (mis. "Torsi
# Puntir recorded at..." -> sekarang "Torsion recorded at...").
_EVAL_LABEL_EN = {
    "Tensile Al": "Aluminium Tensile", "Elongasi Al": "Aluminium Elongation",
    "Tensile Steel": "Steel Tensile", "Elongasi Steel": "Steel Elongation",
    "Torsi Puntir": "Torsion", "Jumlah Lilitan": "Number of Wraps",
}


def evaluate_reason(label, actual, minimum, unit):
    passed = actual >= minimum
    lang = st.session_state.get('lang', 'id')
    if lang == 'en':
        label_disp = _EVAL_LABEL_EN.get(label, label)
        if passed:
            margin = actual - minimum
            reason = f"{label_disp} recorded at {actual:.2f} {unit}, meets the minimum threshold of {minimum:.2f} {unit} (margin +{margin:.2f} {unit})."
        else:
            deficit = minimum - actual
            reason = f"{label_disp} recorded at {actual:.2f} {unit}, does NOT meet the minimum threshold of {minimum:.2f} {unit} (shortfall {deficit:.2f} {unit})."
    else:
        if passed:
            margin = actual - minimum
            reason = f"{label} tercatat {actual:.2f} {unit}, memenuhi ambang minimum {minimum:.2f} {unit} (margin +{margin:.2f} {unit})."
        else:
            deficit = minimum - actual
            reason = f"{label} tercatat {actual:.2f} {unit}, TIDAK memenuhi ambang minimum {minimum:.2f} {unit} (kekurangan {deficit:.2f} {unit})."
    return passed, reason


# ---------------------------------------------------------------------------
# PERHITUNGAN TERPUSAT -- dihitung SEKALI di sini (bukan diulang-ulang per
# halaman seperti sebelumnya) supaya Beranda & Keputusan bisa menampilkan
# ringkasan hasil pengujian yang konsisten, walau pengguna belum membuka
# semua halaman. Nilai diambil dari st.session_state yang sama dipakai/
# diperbarui oleh widget input di halaman NDT & DT -- Streamlit menjalankan
# ulang seluruh script tiap kali ada interaksi, jadi blok ini otomatis ikut
# ter-update begitu pengguna kembali ke Beranda/Keputusan.
# ---------------------------------------------------------------------------
# Prefiks "ev_" sengaja dipakai (BUKAN r_al/r_st/dst polos) supaya tidak
# tertimpa/rancu dengan variabel lokal bernama sama di dalam blok halaman DT
# & Keputusan di bawah (di sana r_al dkk. sudah dipakai sebagai string alasan
# saja, hasil unpack tuple -- lihat `p_al, r_al = evaluate_reason(...)`).
ev_al = evaluate_reason("Tensile Al", st.session_state['tensile_al'], acsr_spec['min_tensile_al'], "MPa")
ev_eal = evaluate_reason("Elongasi Al", st.session_state['elong_al'], acsr_spec['min_elong_al'], "%")
ev_st = evaluate_reason("Tensile Steel", st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], "MPa")
ev_est = evaluate_reason("Elongasi Steel", st.session_state['elong_steel'], acsr_spec['min_elong_steel'], "%")
ev_tor = evaluate_reason("Torsi Puntir", st.session_state['torsion'], acsr_spec['min_torsi'], "N.m")
ev_lil = evaluate_reason("Jumlah Lilitan", st.session_state['turns'], acsr_spec['min_lilit'], "Putaran")

al_ok = ev_al[0] and ev_eal[0]
steel_ok = ev_st[0] and ev_est[0]
tor_ok = ev_tor[0] and ev_lil[0]
ndt_lolos = st.session_state['ndt_status'] == "LOLOS"
mikro_ok = ("Mulus" in st.session_state['korosi_status']) and (st.session_state['astm_G'] >= 6.0)
final_pass = ndt_lolos or (al_ok and steel_ok and tor_ok)

CEK_PARAM = [ndt_lolos, mikro_ok, ev_al[0], ev_eal[0], ev_st[0], ev_est[0], ev_tor[0], ev_lil[0]]
lolos_count = sum(1 for c in CEK_PARAM if c)

PARAM_ROWS_RINGKAS = [
    (tt("NDT X-Ray", "NDT X-Ray"), st.session_state['ndt_defek'], tt("Class NORMAL", "Class NORMAL"), ndt_lolos),
    (tt("Mikrografi", "Micrography"), f"{st.session_state['korosi_status']} (G={st.session_state['astm_G']:.1f})",
     tt("Mulus, G ≥ 6.0", "Smooth, G ≥ 6.0"), mikro_ok),
    ("Tensile Al", f"{st.session_state['tensile_al']:.2f} MPa", f"{acsr_spec['min_tensile_al']:.2f} MPa", ev_al[0]),
    (tt("Elongasi Al", "Al Elongation"), f"{st.session_state['elong_al']:.2f} %", f"{acsr_spec['min_elong_al']:.2f} %", ev_eal[0]),
    ("Tensile Steel", f"{st.session_state['tensile_steel']:.2f} MPa", f"{acsr_spec['min_tensile_steel']:.2f} MPa", ev_st[0]),
    (tt("Elongasi Steel", "Steel Elongation"), f"{st.session_state['elong_steel']:.2f} %", f"{acsr_spec['min_elong_steel']:.2f} %", ev_est[0]),
    (tt("Torsi Puntir", "Torsion"), f"{st.session_state['torsion']:.2f} N.m", f"{acsr_spec['min_torsi']:.2f} N.m", ev_tor[0]),
    (tt("Jumlah Lilitan", "Number of Wraps"), tt(f"{st.session_state['turns']:.0f} Putaran", f"{st.session_state['turns']:.0f} Turns"),
     tt(f"{acsr_spec['min_lilit']:.0f} Putaran", f"{acsr_spec['min_lilit']:.0f} Turns"), ev_lil[0]),
]

# ---------------------------------------------------------------------------
# SKOR AGREGAT 0-100 (utils/scoring.py) -- KHUSUS untuk visualisasi ringkasan
# Beranda (donut, bar chart per parameter, kartu "Temuan yang perlu
# ditindak"). SENGAJA TERPISAH dari mesin keputusan: final_pass di atas
# TETAP satu-satunya penentu LAYAK/TIDAK LAYAK (logika boolean, sesuai draft
# TA), skor 0-100 di sini murni alat bantu visual seberapa dekat/jauh tiap
# parameter dari ambang standarnya -- tidak menggantikan/mengubah keputusan.
# ---------------------------------------------------------------------------
AMBANG_SKOR = AMBANG_DEFAULT  # 70.0 -- garis ambang "kelayakan" di grafik & kartu skor

# Ambang confidence RAGU khusus tab NDT X-Ray (v78) -- BUKAN ambang lolos/gagal
# klasifikasi (itu tetap murni dari kelas NORMAL vs lainnya, lihat blok NDT di
# bawah), tapi sinyal terpisah: kalau confidence model DI BAWAH ambang ini,
# kemungkinan besar citra yang diunggah bukan radiograf X-Ray yang valid sama
# sekali (mis. keliru unggah citra mikrografi/foto lain) -- model tetap
# "memaksa" memberi 1 label dari 5 kelasnya (softmax selalu menjumlah ke 100%
# apapun inputnya) walau confidence-nya rendah, jadi user perlu diberi
# peringatan eksplisit, bukan cuma angka confidence kecil yang gampang
# terlewat. TIDAK memengaruhi ndt_lolos/final_pass/SUB_SKOR -- murni
# peringatan UI tambahan di tab NDT X-Ray.
AMBANG_CONF_RAGU_NDT = 0.60

# NDT belum diuji ("-" adalah nilai default sebelum citra diunggah) sengaja
# diberi skor 0 (bukan dihitung ndt_score("-", 0%) yang keliru bisa jadi 100)
# -- konsisten dengan ndt_lolos yang juga False di kondisi "Belum Diuji".
_ndt_sub = 0.0 if st.session_state['ndt_defek'] == "-" else ndt_score(st.session_state['ndt_defek'], st.session_state['ndt_conf'])
_mulus_flag = "Mulus" in st.session_state['korosi_status']

SUB_SKOR = {
    "ndt": _ndt_sub,
    "mikro": mikro_score(_mulus_flag, st.session_state['astm_G'] if _mulus_flag else None, AMBANG_SKOR),
    "tarik_al": (ratio_score(st.session_state['tensile_al'], acsr_spec['min_tensile_al'], AMBANG_SKOR)
                 + ratio_score(st.session_state['elong_al'], acsr_spec['min_elong_al'], AMBANG_SKOR)) / 2,
    "tarik_steel": (ratio_score(st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], AMBANG_SKOR)
                    + ratio_score(st.session_state['elong_steel'], acsr_spec['min_elong_steel'], AMBANG_SKOR)) / 2,
    "torsi": ratio_score(st.session_state['torsion'], acsr_spec['min_torsi'], AMBANG_SKOR),
    "lilit": ratio_score(float(st.session_state['turns']), float(acsr_spec['min_lilit']), AMBANG_SKOR),
}
SKOR_TOTAL = aggregate(SUB_SKOR, BOBOT_DEFAULT)

_ndt_reason_ringkas = (
    tt("Citra X-Ray belum diunggah/diuji di halaman NDT.", "X-Ray image not yet uploaded/tested on the NDT page.")
    if st.session_state['ndt_defek'] == "-" else
    (tt(f"Model mengklasifikasikan kondisi sebagai '{st.session_state['ndt_defek']}' dengan confidence {st.session_state['ndt_conf']*100:.1f}%, sesuai kelas NORMAL yang dipersyaratkan.",
        f"The model classified the condition as '{st.session_state['ndt_defek']}' with {st.session_state['ndt_conf']*100:.1f}% confidence, meeting the required NORMAL class.")
     if ndt_lolos else
     tt(f"Model mendeteksi indikasi cacat '{st.session_state['ndt_defek']}' dengan confidence {st.session_state['ndt_conf']*100:.1f}%, bukan kelas NORMAL sehingga perlu diverifikasi lebih lanjut.",
        f"The model detected a defect indication '{st.session_state['ndt_defek']}' with {st.session_state['ndt_conf']*100:.1f}% confidence, not the NORMAL class, so further verification is needed."))
)
_mikro_reason_ringkas = (
    tt("Data mikrografi belum dianalisis di halaman DT.", "Micrography data not yet analyzed on the DT page.")
    if st.session_state['korosi_status'] == "Belum Dianalisis" else
    (tt(f"Permukaan terklasifikasi mulus dan ukuran butir G={st.session_state['astm_G']:.1f} memenuhi ambang minimum G≥6.0.",
        f"Surface classified as smooth and grain size G={st.session_state['astm_G']:.1f} meets the minimum threshold G≥6.0.")
     if mikro_ok else
     tt(f"Kondisi '{st.session_state['korosi_status']}' (G={st.session_state['astm_G']:.1f}) tidak memenuhi kriteria mulus & G≥6.0.",
        f"Condition '{st.session_state['korosi_status']}' (G={st.session_state['astm_G']:.1f}) does not meet the smooth & G≥6.0 criteria."))
)

# Baris (nama, skor, alasan) sejajar urutan PARAM_ROWS_RINGKAS -- dipakai
# untuk bar chart "Skor per parameter" & kartu "Temuan yang perlu ditindak".
PARAM_SKOR_RINGKAS = [
    (tt("NDT X-Ray", "NDT X-Ray"), SUB_SKOR["ndt"], _ndt_reason_ringkas),
    (tt("Mikrografi", "Micrography"), SUB_SKOR["mikro"], _mikro_reason_ringkas),
    ("Tensile Al", ratio_score(st.session_state['tensile_al'], acsr_spec['min_tensile_al'], AMBANG_SKOR), ev_al[1]),
    (tt("Elongasi Al", "Al Elongation"), ratio_score(st.session_state['elong_al'], acsr_spec['min_elong_al'], AMBANG_SKOR), ev_eal[1]),
    ("Tensile Steel", ratio_score(st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], AMBANG_SKOR), ev_st[1]),
    (tt("Elongasi Steel", "Steel Elongation"), ratio_score(st.session_state['elong_steel'], acsr_spec['min_elong_steel'], AMBANG_SKOR), ev_est[1]),
    (tt("Torsi Puntir", "Torsion"), ratio_score(st.session_state['torsion'], acsr_spec['min_torsi'], AMBANG_SKOR), ev_tor[1]),
    (tt("Jumlah Lilitan", "Number of Wraps"), ratio_score(float(st.session_state['turns']), float(acsr_spec['min_lilit']), AMBANG_SKOR), ev_lil[1]),
]

# Margin (%) terhadap ambang standar -- HANYA parameter DT mekanis (NDT &
# mikrografi tidak berupa rasio nilai/ambang linear), dipakai kartu KPI
# "Margin Terkecil" di Beranda.
MARGIN_TERKECIL = min(
    st.session_state['tensile_al'] / acsr_spec['min_tensile_al'] - 1,
    st.session_state['elong_al'] / acsr_spec['min_elong_al'] - 1,
    st.session_state['tensile_steel'] / acsr_spec['min_tensile_steel'] - 1,
    st.session_state['elong_steel'] / acsr_spec['min_elong_steel'] - 1,
    st.session_state['torsion'] / acsr_spec['min_torsi'] - 1,
    st.session_state['turns'] / acsr_spec['min_lilit'] - 1,
) * 100

# Aksi tombol "Simpan Pengujian" di header (klik dideteksi lebih awal, lihat
# blok header di atas -- _clicked_simpan_pengujian) -- BARU dijalankan di
# sini karena butuh final_pass & SKOR_TOTAL dari blok terpusat di atas.
# Snapshot yang disimpan sama persis field-nya dengan versi v73 (sebelumnya
# di halaman Keputusan), ditambah data spesimen baru (kode/Haspel/panjang/
# tanggal/tim), dan "hasil" tetap dari final_pass (mesin keputusan asli),
# BUKAN re-derive dari SKOR_TOTAL -- pemisahan skor visual vs keputusan
# tetap dijaga (lihat utils/scoring.py).
if _clicked_simpan_pengujian:
    st.session_state['riwayat_list'].insert(0, {
        tt("Waktu", "Time"): datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        tt("Kode Spesimen", "Specimen Code"): st.session_state['spesimen_kode'],
        tt("Tipe ACSR", "ACSR Type"): selected_acsr_type,
        "NDT X-Ray": f"{st.session_state['ndt_defek']} ({st.session_state['ndt_status']})",
        tt("Mikrografi", "Micrography"): f"{st.session_state['korosi_status']} (G={st.session_state['astm_G']})",
        tt("Tarik Al/Baja (MPa)", "Al/Steel Tensile (MPa)"):
            f"{st.session_state['tensile_al']:.1f} / {st.session_state['tensile_steel']:.1f}",
        tt("Torsi (N.m) / Lilit", "Torsion (N.m) / Wrap"):
            f"{st.session_state['torsion']:.1f} / {st.session_state['turns']:.1f}",
        tt("Skor Agregat", "Aggregate Score"): f"{SKOR_TOTAL:.1f}",
        tt("Hasil", "Result"): tt("LAYAK PAKAI", "FIT FOR USE") if final_pass else tt("AFKIR", "REJECTED"),
    })
    st.toast(tt("Tersimpan ke Riwayat Pengujian.", "Saved to Test History."), icon="💾")


@st.cache_resource
def load_trained_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_FILE):
        return None, None, None, None, f"File tidak ditemukan di path: {MODEL_FILE}"
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=dev)
        class_names = checkpoint.get('class_names', ['KONDUKTOR BENDING', 'KONDUKTOR MEKAR', 'KONDUKTOR MULUR', 'KONDUKTOR NORMAL', 'KONDUKTOR PUTUS'])
        # Arsitektur asli training: ConvNeXt-Tiny (BUKAN ResNet50 -- itu
        # penyebab semua kegagalan "Not Found" sebelumnya, arsitekturnya
        # tidak cocok dengan state_dict yang tersimpan di checkpoint).
        # weights=None karena kita load bobot hasil training sendiri, bukan
        # bobot pretrained ImageNet lagi.
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, len(class_names))
        )
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model = model.to(dev)
        model.eval()
        val_acc = checkpoint.get('val_accuracy', None)
        batch_size = checkpoint.get('batch_size', None)
        return model, class_names, val_acc, batch_size, None
    except Exception as e:
        import traceback
        return None, None, None, None, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"


# device DIPISAH dari hasil cache function -- sebelumnya slot ke-5 dobel
# fungsi (torch.device saat sukses, STRING pesan error saat gagal), yang
# bisa bikin bug lain di tempat yang mengasumsikan `device` selalu berupa
# torch.device (misal .to(device) di run_inference/run_korosi_inference).
model, class_names, model_acc, model_batch, model_load_error = load_trained_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------------
# MODEL KOROSI MIKROGRAFI (ResNet18, 2 kelas: Mikro_Mulus / Mikro_Korosi)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_korosi_model():
    path = find_in_drive('/content/drive/**/cnn_mikro_korosi_model.pth', "",
                          local_pattern="models/cnn_mikro_korosi_model.pth")
    if not path or not os.path.exists(path):
        return None, None, None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=dev)
        cnames = checkpoint.get('class_names', ['Mikro_Mulus', 'Mikro_Korosi'])
        m = models.resnet18()
        # fc.1 di checkpoint (bukan fc langsung) -> fc aslinya nn.Sequential(Dropout, Linear)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, len(cnames)))
        m.load_state_dict(checkpoint['model_state_dict'])
        m = m.to(dev)
        m.eval()
        val_acc = checkpoint.get('val_accuracy', None)
        if hasattr(val_acc, "item"):
            val_acc = val_acc.item()
        return m, cnames, val_acc
    except Exception as e:
        st.warning(f"Gagal memuat model korosi mikrografi: {e}")
        return None, None, None


korosi_model, korosi_class_names, korosi_val_acc = load_korosi_model()


def run_korosi_inference(pil_img):
    """Return (label, confidence) dari model korosi ResNet18."""
    tensor = img_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = korosi_model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)
    return korosi_class_names[idx.item()], conf.item()


# ---------------------------------------------------------------------------
# GRAIN SIZE — dihitung pakai FORMULA ASTM E112 (metode Jeffries/Planimetric),
# BUKAN model AI. Jumlah butir dideteksi otomatis dari citra pakai OpenCV,
# lalu dimasukkan ke rumus standar.
#
# >>> PENTING: konstanta di bawah (diameter lingkaran uji 79.8mm / luas area
# acuan 5000 mm^2) adalah nilai standar yang UMUM DIPAKAI DI LITERATUR TEKNIK
# untuk ASTM E112 metode Jeffries. VERIFIKASI ulang terhadap dokumen resmi
# ASTM E112 dan SOP laboratorium kamu sebelum dipakai untuk hasil publikasi --
# saya susun ini dari referensi umum, bukan dari salinan resmi standarnya. <<<
# ---------------------------------------------------------------------------
def count_grains_cv2(pil_img):
    """Deteksi jumlah butir (blob) dalam citra mikrografi pakai OpenCV.
    Return jumlah kontur yang lolos filter luas (grain count kasar)."""
    img_arr = np.array(pil_img.convert("L"))  # grayscale
    img_arr = cv2.GaussianBlur(img_arr, (3, 3), 0)
    _, thresh = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_arr.shape[0] * img_arr.shape[1]
    min_area = img_area * 0.0002  # buang noise/kontur super kecil
    max_area = img_area * 0.15    # buang blob raksasa (background salah kesegmentasi)
    valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    return len(valid), thresh


def astm_grain_size(n_grains, magnification, test_circle_area_mm2=5000.0):
    """Hitung ASTM Grain Size Number (G) dari jumlah butir & perbesaran,
    metode Jeffries Planimetric (lihat catatan verifikasi di atas)."""
    if n_grains <= 0:
        return None
    na = n_grains * (magnification ** 2) / test_circle_area_mm2  # butir/mm^2 pada 1x
    g = 3.322 * np.log10(na) - 2.954
    return round(float(g), 1)


def run_inference(img):
    tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)
    return class_names[idx.item()], conf.item()


def create_benchmark_chart(title, actual, std, unit, chart_type="bar"):
    fig = go.Figure()
    if chart_type == "line":
        # Dua garis horizontal sejajar -- garis standar minimum vs garis hasil uji --
        # supaya perbandingannya jelas terbaca, bukan satu garis diagonal yang
        # menyambungkan 2 kategori (kurang bermakna secara teknis).
        x_range = [tt("Awal", "Start"), tt("Akhir", "End")]
        pass_ok = actual >= std
        fig.add_trace(go.Scatter(
            x=x_range, y=[std, std], mode="lines", name=f"{tt('Standar Minimum', 'Minimum Standard')} ({std} {unit})",
            line=dict(color="#94A3B8", width=2.5, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=[actual, actual], mode="lines+markers", name=f"{tt('Hasil Uji', 'Test Result')} ({actual} {unit})",
            line=dict(color="#1E7F4E" if pass_ok else "#C8443C", width=3.5),
            marker=dict(size=9),
        ))
        fig.update_layout(
            title=f"{tt('Grafik Evaluasi', 'Evaluation Chart')} {title}", height=260, margin=dict(l=10, r=10, t=35, b=10),
            yaxis_title=unit, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            template="plotly_white",
        )
    else:
        fig.add_trace(go.Bar(
            x=[tt("Nilai Uji Spesimen", "Specimen Test Value"), tt("Min. Standar", "Min. Standard")], y=[actual, std],
            marker_color=["#1E7F4E" if actual >= std else "#C8443C", "#C3CEDB"],
            text=[f"{actual} {unit}", f"{std} {unit}"], textposition="auto"
        ))
        fig.update_layout(title=f"{tt('Grafik Evaluasi', 'Evaluation Chart')} {title}", height=250, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def mpl_benchmark_png(title, actual, std, unit):
    """Buat chart batang sederhana pakai matplotlib -> PNG bytes, untuk dilampirkan ke PDF."""
    fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=150)
    color = "#1E7F4E" if actual >= std else "#C8443C"
    bars = ax.bar(["Spesimen", "Min. Standar"], [actual, std], color=[color, "#C3CEDB"])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=9, color="#002B5C")
    ax.set_ylabel(unit, fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(all_reasons):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=15, leading=18, alignment=1, textColor=colors.HexColor("#002B5C"))
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=9, leading=11, alignment=1, textColor=colors.gray)
    reason_style = ParagraphStyle('ReasonStyle', parent=styles['Normal'], fontSize=8.3, leading=11.5, textColor=colors.HexColor("#334155"))
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=11, leading=13, textColor=colors.HexColor("#002B5C"), spaceBefore=10, spaceAfter=4)

    elements.append(Paragraph("SERTIFIKAT VALIDASI KELAYAKAN KONDUKTOR ACSR", title_style))
    elements.append(Paragraph(f"Tipe Spesimen: {selected_acsr_type} | Konsorsium PT PLN x UNDIP", subtitle_style))
    elements.append(Paragraph(
        f"Diameter: {acsr_spec['diameter_mm']} mm ({acsr_spec['jumlah_kawat']}) | Standar Acuan: {acsr_spec['standar_acuan']}",
        subtitle_style))
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#F5B921")))
    elements.append(Spacer(1, 12))

    al_pass = (st.session_state['tensile_al'] >= acsr_spec['min_tensile_al']) and (st.session_state['elong_al'] >= acsr_spec['min_elong_al'])
    steel_pass = (st.session_state['tensile_steel'] >= acsr_spec['min_tensile_steel']) and (st.session_state['elong_steel'] >= acsr_spec['min_elong_steel'])
    tor_pass = (st.session_state['torsion'] >= acsr_spec['min_torsi']) and (st.session_state['turns'] >= acsr_spec['min_lilit'])
    final_pass = (st.session_state['ndt_status'] == "LOLOS") or (al_pass and steel_pass and tor_pass)
    status_text = "LAYAK PAKAI (SUITABLE)" if final_pass else "TIDAK LAYAK PAKAI (REJECTED)"
    status_color = colors.HexColor("#1E7F4E") if final_pass else colors.HexColor("#C8443C")

    data_summary = [
        ["Parameter Pengujian", "Hasil Evaluasi", "Ambang Batas Standar", "Status Validasi"],
        ["NDT Radiografi X-Ray", st.session_state['ndt_defek'], "Class NORMAL", st.session_state['ndt_status']],
        ["Kondisi Mikrografi", st.session_state['korosi_status'], f"ASTM G={st.session_state['astm_G']}", "WARNING" if "Korosi" in st.session_state['korosi_status'] else "PASSED"],
        ["Tarik Kawat Alumunium", f"{st.session_state['tensile_al']} MPa / {st.session_state['elong_al']}%", f"Min. {acsr_spec['min_tensile_al']} MPa", "PASSED" if al_pass else "FAILED"],
        ["Tarik Inti Baja", f"{st.session_state['tensile_steel']} MPa / {st.session_state['elong_steel']}%", f"Min. {acsr_spec['min_tensile_steel']} MPa", "PASSED" if steel_pass else "FAILED"],
        ["Uji Torsi Puntir", f"{st.session_state['torsion']} N.m", f"Min. {acsr_spec['min_torsi']} N.m", "PASSED" if st.session_state['torsion'] >= acsr_spec['min_torsi'] else "FAILED"],
        ["Uji Lilitan Kawat", f"{st.session_state['turns']} Putaran", f"Min. {acsr_spec['min_lilit']} Putaran", "PASSED" if st.session_state['turns'] >= acsr_spec['min_lilit'] else "FAILED"],
    ]

    t = Table(data_summary, colWidths=[135, 125, 125, 95])
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#002B5C")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]
    for row_i, row in enumerate(data_summary[1:], start=1):
        status_val = row[-1]
        color = colors.HexColor("#1E7F4E") if status_val in ("PASSED", "LOLOS") else (
            colors.HexColor("#C77E12") if status_val == "WARNING" else colors.HexColor("#C8443C"))
        style_cmds.append(('TEXTCOLOR', (3, row_i), (3, row_i), color))
        style_cmds.append(('FONTNAME', (3, row_i), (3, row_i), 'Helvetica-Bold'))
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    elements.append(Spacer(1, 14))

    # --- Rincian alasan evaluasi: tabel per parameter (Status + Kenapa) ---
    elements.append(Paragraph("Rincian Alasan Evaluasi", section_style))

    al_pass2 = st.session_state['elong_al'] >= acsr_spec['min_elong_al']
    steel_pass2 = st.session_state['elong_steel'] >= acsr_spec['min_elong_steel']
    tor_pass_flag = st.session_state['torsion'] >= acsr_spec['min_torsi']
    lil_pass_flag = st.session_state['turns'] >= acsr_spec['min_lilit']
    r_al_pdf = evaluate_reason("Tensile Al", st.session_state['tensile_al'], acsr_spec['min_tensile_al'], "MPa")[1]
    r_al2_pdf = evaluate_reason("Elongasi Al", st.session_state['elong_al'], acsr_spec['min_elong_al'], "%")[1]
    r_st_pdf = evaluate_reason("Tensile Steel", st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], "MPa")[1]
    r_st2_pdf = evaluate_reason("Elongasi Steel", st.session_state['elong_steel'], acsr_spec['min_elong_steel'], "%")[1]
    r_tor_pdf = evaluate_reason("Torsi Puntir", st.session_state['torsion'], acsr_spec['min_torsi'], "N.m")[1]
    r_lil_pdf = evaluate_reason("Jumlah Lilitan", st.session_state['turns'], acsr_spec['min_lilit'], "Putaran")[1]
    is_korosi = "Korosi" in st.session_state['korosi_status']

    reason_para_style = ParagraphStyle('ReasonPara', parent=styles['Normal'], fontSize=7.6, leading=10.2, textColor=colors.HexColor("#334155"))
    param_para_style = ParagraphStyle('ParamPara', parent=styles['Normal'], fontSize=8, leading=10, fontName='Helvetica-Bold', textColor=colors.HexColor("#0F172A"))

    detail_specs = [
        ("NDT X-Ray", st.session_state['ndt_status'] == "LOLOS", False,
         f"NDT X-Ray mendeteksi kondisi '{st.session_state['ndt_defek']}'. Status {st.session_state['ndt_status']} karena kelas ini "
         + ("sesuai kriteria NORMAL yang dipersyaratkan." if st.session_state['ndt_status'] == "LOLOS" else "menunjukkan indikasi cacat struktural (bukan kelas NORMAL).")),
        ("Mikrografi", not is_korosi, is_korosi,
         st.session_state['korosi_reason'] or "Mikrografi belum dianalisis — unggah citra terlebih dahulu di Modul DT."),
        ("Tensile Al", al_pass, False, r_al_pdf),
        ("Elongasi Al", al_pass2, False, r_al2_pdf),
        ("Tensile Steel", steel_pass, False, r_st_pdf),
        ("Elongasi Steel", steel_pass2, False, r_st2_pdf),
        ("Torsi Puntir", tor_pass_flag, False, r_tor_pdf),
        ("Jumlah Lilitan", lil_pass_flag, False, r_lil_pdf),
    ]

    table_rows = [["Parameter", "Status", "Alasan (Kenapa Passed / Failed / Warning)"]]
    row_colors = []
    for name, ok, is_warn, reason_text in detail_specs:
        status_label = "WARNING" if is_warn else ("PASSED" if ok else "FAILED")
        table_rows.append([Paragraph(name, param_para_style), status_label, Paragraph(reason_text, reason_para_style)])
        row_colors.append(colors.HexColor("#C77E12") if is_warn else (colors.HexColor("#1E7F4E") if ok else colors.HexColor("#C8443C")))

    detail_table = Table(table_rows, colWidths=[85, 55, 355])
    detail_style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#002B5C")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    for row_i, c in enumerate(row_colors, start=1):
        detail_style_cmds.append(('TEXTCOLOR', (1, row_i), (1, row_i), c))
        detail_style_cmds.append(('FONTNAME', (1, row_i), (1, row_i), 'Helvetica-Bold'))
    detail_table.setStyle(TableStyle(detail_style_cmds))
    elements.append(detail_table)
    elements.append(Spacer(1, 10))

    # --- Grafik pendukung ---
    elements.append(Paragraph("Grafik Pendukung Pengujian Mekanis", section_style))
    chart_specs = [
        ("Tensile Alumunium", st.session_state['tensile_al'], acsr_spec['min_tensile_al'], "MPa"),
        ("Tensile Inti Baja", st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], "MPa"),
        ("Torsi Puntir", st.session_state['torsion'], acsr_spec['min_torsi'], "N.m"),
        ("Jumlah Lilitan", st.session_state['turns'], acsr_spec['min_lilit'], "Turns"),
    ]
    chart_imgs = [RLImage(mpl_benchmark_png(t_, a, s, u), width=65 * mm, height=45 * mm) for t_, a, s, u in chart_specs]
    chart_table = Table([[chart_imgs[0], chart_imgs[1]], [chart_imgs[2], chart_imgs[3]]])
    chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    elements.append(chart_table)
    elements.append(Spacer(1, 16))

    res_style = ParagraphStyle('ResStyle', parent=styles['Heading2'], fontSize=11, leading=13, alignment=1, textColor=status_color)
    elements.append(Paragraph(f"<b>REKOMENDASI FINAL SPESIMEN ACSR: {status_text}</b>", res_style))
    elements.append(Spacer(1, 8))

    # --- Paragraf kesimpulan ringkas, dirangkai otomatis dari data hasil evaluasi ---
    summary_style = ParagraphStyle('SummaryStyle', parent=styles['Normal'], fontSize=9, leading=13.5,
                                    alignment=4, textColor=colors.HexColor("#1E293B"))  # alignment=4 -> justify

    failed_list_pdf = []
    if st.session_state['ndt_status'] != "LOLOS":
        failed_list_pdf.append(f"inspeksi NDT X-Ray (terdeteksi kondisi {st.session_state['ndt_defek']})")
    if is_korosi:
        failed_list_pdf.append("uji mikrografi (terindikasi korosi permukaan)")
    if not al_pass or not al_pass2:
        failed_list_pdf.append("uji tarik kawat alumunium")
    if not steel_pass or not steel_pass2:
        failed_list_pdf.append("uji tarik kawat inti baja")
    if not tor_pass_flag:
        failed_list_pdf.append("uji torsi puntir")
    if not lil_pass_flag:
        failed_list_pdf.append("uji lilitan")

    if final_pass:
        summary_text = (
            f"Berdasarkan hasil evaluasi menyeluruh terhadap spesimen konduktor "
            f"<b>{selected_acsr_type}</b> (diameter {acsr_spec['diameter_mm']} mm, {acsr_spec['jumlah_kawat']}) "
            f"pascaproses ex-reconductoring, seluruh parameter kritis yang diuji &mdash; meliputi inspeksi "
            f"Non-Destructive Testing (NDT) berbasis radiografi X-Ray, uji mikrografi, uji tarik kawat "
            f"alumunium dan inti baja, serta uji torsi puntir dan lilitan &mdash; telah memenuhi ambang batas "
            f"minimum yang dipersyaratkan berdasarkan standar acuan {acsr_spec['standar_acuan']}. "
            f"Dengan demikian, spesimen konduktor dinyatakan <b>LAYAK PAKAI</b> dan direkomendasikan "
            f"untuk dapat dioperasikan kembali pada jaringan transmisi maupun distribusi tenaga listrik."
        )
    else:
        failed_text = "; ".join(failed_list_pdf) if failed_list_pdf else "sejumlah parameter kritis"
        summary_text = (
            f"Berdasarkan hasil evaluasi menyeluruh terhadap spesimen konduktor "
            f"<b>{selected_acsr_type}</b> (diameter {acsr_spec['diameter_mm']} mm, {acsr_spec['jumlah_kawat']}) "
            f"pascaproses ex-reconductoring, ditemukan bahwa spesimen <b>TIDAK memenuhi</b> ambang batas "
            f"minimum pada parameter: {failed_text}. Kondisi ini mengindikasikan penurunan integritas "
            f"struktural dan/atau mekanis konduktor sehingga tidak memenuhi persyaratan standar acuan "
            f"{acsr_spec['standar_acuan']}. Dengan demikian, spesimen konduktor dinyatakan "
            f"<b>TIDAK LAYAK PAKAI (AFKIR)</b> dan tidak direkomendasikan untuk dioperasikan kembali "
            f"tanpa perbaikan atau penggantian lebih lanjut."
        )
    elements.append(Paragraph(summary_text, summary_style))
    elements.append(Spacer(1, 20))

    sign_data = [["Diperiksa oleh,", "Disahkan oleh,"], ["", ""], ["", ""],
                 ["(_________________________)", "(_________________________)"],
                 ["Peneliti / Engineer", "Kepala Laboratorium"]]
    sign_tbl = Table(sign_data, colWidths=[250, 250])
    sign_tbl.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 1), (-1, 2), 16),
    ]))
    elements.append(sign_tbl)

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===========================================================================
# (v74: header utama + toggle bahasa sudah dirender di atas, SEBELUM blok
# PERHITUNGAN TERPUSAT -- lihat komentar "HEADER UTAMA (v74)" dekat sidebar.
# Navigasi tab pindah ke sidebar (sidebar_nav_list). PAGES dict/render_header/
# header_tabs versi lama TIDAK dipakai lagi di sini -- fungsinya dibiarkan
# ada di utils/ui.py sebagai dead code, tidak dihapus, kalau-kalau perlu
# dikembalikan nanti.)
# ===========================================================================
# 0. BERANDA
# ===========================================================================
if menu == "beranda":
    st.markdown(f"""
    <div class="page-title-block">
        <h1>{t("page_title_beranda")}</h1>
        <p class="page-subtitle">{t("page_subtitle_beranda")}</p>
    </div>
    """, unsafe_allow_html=True)

    intro_paragraph = tt(
        """Penelitian ini bertujuan untuk mengevaluasi tingkat kelayakan operasional
        konduktor <b>Aluminium Conductor Steel Reinforced (ACSR)</b> pascaproses
        <i>ex-reconductoring</i>, guna menentukan apakah konduktor tersebut masih
        memenuhi persyaratan teknis untuk dioperasikan kembali pada jaringan
        transmisi maupun distribusi tenaga listrik, atau sebaliknya harus
        dinyatakan afkir (<i>rejected</i>). Evaluasi dilakukan melalui pendekatan
        gabungan <b>Non-Destructive Testing (NDT)</b> berbasis analisis citra
        radiografi X-Ray dan <b>Destructive Testing (DT)</b> yang meliputi
        pengujian mikrografi, uji tarik, uji torsi, serta uji lilitan, dengan
        mengacu pada standar internasional ASTM dan IEC. Pendekatan berbasis
        kecerdasan buatan diterapkan untuk mempercepat proses klasifikasi dan
        interpretasi data pengujian, sehingga evaluasi kelayakan konduktor dapat
        dilakukan secara lebih presisi, objektif, dan efisien dibandingkan metode
        konvensional.""",
        """This research aims to evaluate the operational feasibility level of
        <b>Aluminium Conductor Steel Reinforced (ACSR)</b> conductors after the
        <i>ex-reconductoring</i> process, to determine whether the conductor still
        meets the technical requirements to be re-operated on the electrical
        transmission or distribution network, or otherwise must be declared
        rejected. The evaluation is carried out through a combined
        <b>Non-Destructive Testing (NDT)</b> approach based on X-Ray radiograph
        image analysis and <b>Destructive Testing (DT)</b> which includes
        micrography testing, tensile testing, torsion testing, and wrap testing,
        referring to the ASTM and IEC international standards. An artificial
        intelligence-based approach is applied to accelerate the classification
        process and interpretation of test data, so that the conductor
        feasibility evaluation can be carried out more precisely, objectively,
        and efficiently compared to conventional methods.""",
    )
    st.markdown(f"""
    <div class="info-tile" style="margin-bottom:18px;">
    <p style="font-size:0.92rem; line-height:1.7; margin:0;">
    {intro_paragraph}
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 🏠 {tt('Ringkasan Sistem & Konsorsium Riset', 'System Overview & Research Consortium')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        step_card("1", "🔬", tt("Inspeksi NDT", "NDT Inspection"),
                   tt("Pemindaian cacat internal via Radiografi X-Ray berbasis AI (ConvNeXt-Tiny), termasuk mode pemeriksaan batch.",
                      "Internal defect scanning via AI-based X-Ray Radiography (ConvNeXt-Tiny), including batch inspection mode."))
    with c2:
        step_card("2", "🧪", tt("Uji Destruktif", "Destructive Test"),
                   tt("Uji Mikrografi (otomatis), Uji Tarik Al/Baja, Torsi Puntir, dan Uji Lilitan mengacu ASTM/IEC.",
                      "Micrography Test (automatic), Al/Steel Tensile Test, Torsion Test, and Wrap Test referring to ASTM/IEC."))
    with c3:
        step_card("3", "📜", tt("Laporan Validasi", "Validation Report"),
                   tt("Matriks keputusan otomatis dengan alasan tertulis & pencetakan Sertifikat Validasi PDF resmi.",
                      "Automatic decision matrix with written reasoning & official PDF Validation Certificate printing."))

    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        info_tile(tt("Tarik Min. Alumunium", "Min. Aluminium Tensile"), f"{acsr_spec['min_tensile_al']} MPa", accent="blue")
    with sc2:
        info_tile(tt("Tarik Min. Inti Baja", "Min. Steel Core Tensile"), f"{acsr_spec['min_tensile_steel']} MPa", accent="purple")
    with sc3:
        info_tile(tt("Torsi Puntir Min.", "Min. Torsion"), f"{acsr_spec['min_torsi']} N.m", accent="teal")
    with sc4:
        info_tile(tt("Jumlah Lilitan Min.", "Min. Wrap Turns"), f"{acsr_spec['min_lilit']} Turns", accent="orange")

    # -----------------------------------------------------------------------
    # RINGKASAN HASIL PENGUJIAN -- gaya dashboard skor agregat (donut + kartu
    # KPI + bar chart per parameter + "Temuan yang perlu ditindak"), dihitung
    # dari blok "PERHITUNGAN TERPUSAT" & blok skor di atas app.py. Kalau belum
    # ada citra/nilai yang diuji, ini otomatis menampilkan status default
    # (Belum Diuji/Belum Dianalisis) -- bukan seolah-olah sudah LOLOS semua.
    # CATATAN: skor 0-100 di sini murni visual (lihat utils/scoring.py) --
    # label LAYAK PAKAI/AFKIR tetap dari final_pass (logika boolean), BUKAN
    # dari ambang skor, supaya mesin keputusan tidak berubah.
    # -----------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"##### 📊 {tt('Ringkasan Hasil Pengujian', 'Test Result Summary')}")
    st.caption(tt("Diperbarui otomatis begitu citra/nilai diuji di halaman NDT & DT.",
                   "Updates automatically as images/values are tested on the NDT & DT pages."))

    # Label EN untuk 6 sub-skor -- LABEL_BOBOT di utils/scoring.py cuma versi
    # ID (dipakai juga di tabel rincian & tempat lain), jadi di sini disiapkan
    # padanan EN-nya khusus untuk daftar mini_score_bar() di kartu donut (v77).
    _SUB_SKOR_LABEL_EN = {
        "ndt": "NDT X-Ray", "mikro": "Micrography", "tarik_al": "Aluminium Tensile",
        "tarik_steel": "Steel Core Tensile", "torsi": "Torsion", "lilit": "Wrap Test",
    }
    _SUB_SKOR_URUTAN = ["ndt", "mikro", "tarik_al", "tarik_steel", "torsi", "lilit"]

    kol_kiri, kol_kanan = st.columns([1, 2.1])
    with kol_kiri:
        with st.container(border=True):
            st.plotly_chart(donut_score(SKOR_TOTAL, skor_status_color(SKOR_TOTAL, AMBANG_SKOR)),
                             use_container_width=True, config={"displayModeBar": False})
            plain_verdict(
                "pass" if final_pass else "fail",
                tt("LAYAK PAKAI", "FIT FOR USE") if final_pass else tt("AFKIR (TIDAK LAYAK)", "REJECTED (NOT FIT)"),
                tt("Parameter kritis NDT dan/atau DT memenuhi ambang standar acuan.",
                   "Critical NDT and/or DT parameters meet the reference standard threshold.")
                if final_pass else
                tt("Terdapat parameter kritis di bawah ambang standar acuan.",
                   "Critical parameters are below the reference standard threshold."))
            st.divider()
            for _sk in _SUB_SKOR_URUTAN:
                mini_score_bar(tt(LABEL_BOBOT[_sk], _SUB_SKOR_LABEL_EN[_sk]),
                                SUB_SKOR[_sk], skor_status_color(SUB_SKOR[_sk], AMBANG_SKOR))

    with kol_kanan:
        kk1, kk2, kk3, kk4 = st.columns(4)
        with kk1:
            info_tile(tt("Parameter Memenuhi", "Parameters Met"), f"{lolos_count}", unit="/ 8",
                       explanation=tt("gabungan NDT, mikrografi, dan uji mekanis", "combined NDT, micrography, and mechanical tests"),
                       accent="green" if lolos_count == 8 else "red",
                       value_color=PASS_COLOR if lolos_count == 8 else FAIL_COLOR)
        with kk2:
            info_tile(tt("Confidence NDT", "NDT Confidence"), f"{st.session_state['ndt_conf']*100:.1f}", unit="%",
                       explanation=tt(f"kelas {st.session_state['ndt_defek'].replace('KONDUKTOR ', '').lower()}",
                                      f"class {st.session_state['ndt_defek'].replace('KONDUKTOR ', '').lower()}"),
                       accent="blue", value_color=PASS_COLOR if ndt_lolos else FAIL_COLOR)
        with kk3:
            info_tile(tt("Margin Terkecil", "Smallest Margin"), f"{MARGIN_TERKECIL:+.1f}", unit="%",
                       explanation=tt("selisih terkecil terhadap ambang standar", "smallest margin against the standard threshold"),
                       accent="gold", value_color=PASS_COLOR if MARGIN_TERKECIL >= 0 else FAIL_COLOR)
        with kk4:
            info_tile(tt("ASTM Grain Size", "ASTM Grain Size"), f"{st.session_state['astm_G']:.1f}", unit="G",
                       explanation=tt("ambang minimum G ≥ 6.0", "minimum threshold G ≥ 6.0"),
                       accent="teal", value_color=PASS_COLOR if st.session_state['astm_G'] >= 6.0 else WARN_COLOR)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            hcol1, hcol2 = st.columns([2, 1.3])
            with hcol1:
                st.markdown(f"##### {tt('Skor per parameter pengujian', 'Score per test parameter')}")
                st.caption(tt("Garis putus-putus adalah ambang kelayakan (skor 0-100, murni visual).",
                              "The dashed line is the feasibility threshold (0-100 score, visual only)."))
            with hcol2:
                st.markdown(f"""
                <div class="score-legend">
                    <div class="score-legend-item"><span class="dot" style="background:{PASS_COLOR};"></span>{tt('Di atas ambang', 'Above threshold')}</div>
                    <div class="score-legend-item"><span class="dot" style="background:{WARN_COLOR};"></span>{tt('Mendekati ambang', 'Near threshold')}</div>
                    <div class="score-legend-item"><span class="dot" style="background:{FAIL_COLOR};"></span>{tt('Tidak memenuhi', 'Not meeting')}</div>
                </div>
                """, unsafe_allow_html=True)
            _labels_skor = [p[0] for p in PARAM_SKOR_RINGKAS]
            _nilai_skor = [p[1] for p in PARAM_SKOR_RINGKAS]
            _warna_skor = [skor_status_color(s, AMBANG_SKOR) for s in _nilai_skor]
            st.plotly_chart(bar_parameter_scores(_labels_skor, _nilai_skor, _warna_skor, AMBANG_SKOR),
                             use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        _temuan = [p for p in PARAM_SKOR_RINGKAS if p[1] < AMBANG_SKOR]
        st.markdown(f"**{tt('Temuan yang perlu ditindak', 'Findings that need action')}**")
        st.caption(tt(f"{len(_temuan)} dari 8 parameter di bawah ambang {AMBANG_SKOR:.0f}",
                      f"{len(_temuan)} of 8 parameters below the {AMBANG_SKOR:.0f} threshold"))
        if not _temuan:
            result_box("pass", tt("Tidak ada temuan", "No findings"),
                        tt(f"Seluruh parameter berada di atas ambang kelayakan untuk spesifikasi {selected_acsr_type}.",
                           f"All parameters are above the feasibility threshold for the {selected_acsr_type} specification."))
        else:
            for _chunk_start in range(0, len(_temuan), 3):
                _cols_temuan = st.columns(3)
                for _col, _p in zip(_cols_temuan, _temuan[_chunk_start:_chunk_start + 3]):
                    with _col:
                        result_box("fail" if _p[1] < AMBANG_SKOR - 20 else "warn",
                                    f"{_p[0]} — {tt('skor', 'score')} {_p[1]:.0f}", _p[2])

    with st.expander(f"📋 {tt('Rincian Tabel Status Tiap Parameter', 'Detailed Status Table per Parameter')}"):
        col_parameter, col_hasil, col_standar, col_status = (
            tt("Parameter", "Parameter"), tt("Hasil", "Result"), tt("Standar Min.", "Min. Standard"), tt("Status", "Status"))
        ringkas_rows = [{
            col_parameter: nama, col_hasil: hasil, col_standar: standar,
            col_status: (tt("✅ Lolos", "✅ Passed") if ok else tt("❌ Belum Lolos", "❌ Not Passed")),
        } for nama, hasil, standar, ok in PARAM_ROWS_RINGKAS]
        st.dataframe(pd.DataFrame(ringkas_rows), use_container_width=True, hide_index=True, column_config={
            col_parameter: st.column_config.TextColumn(width="medium"),
            col_hasil: st.column_config.TextColumn(width="medium"),
            col_standar: st.column_config.TextColumn(width="medium"),
            col_status: st.column_config.TextColumn(width="small"),
        })

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### {t('glossary_ndt_title')}")
    st.caption(t("glossary_caption"))
    ndt_terms = list(NDT_CLASS_DESC.items())
    col_a, col_b = st.columns(2)
    for i, (kelas, desk) in enumerate(ndt_terms):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            with st.expander(f"{NDT_CLASS_ICON.get(kelas, '❔')}  {kelas}"):
                # FIX: sebelumnya st.write(desk) menampilkan dict {id,en} MENTAH
                # (bukan teks sesuai bahasa aktif) -- sekarang di-resolve dulu.
                st.write(desk.get(st.session_state['lang'], desk.get('id', '')))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### {t('glossary_mech_title')}")
    st.caption(t("glossary_caption"))
    mech_terms = list(MECH_TERMS_DESC.items())
    col_c, col_d = st.columns(2)
    for i, (istilah, desk) in enumerate(mech_terms):
        target_col = col_c if i % 2 == 0 else col_d
        judul_istilah = (MECH_TERMS_LABEL_ID.get(istilah, istilah) if st.session_state['lang'] == 'id'
                          else MECH_TERMS_LABEL_EN.get(istilah, istilah))
        with target_col:
            # Sengaja TANPA ikon di depan judul expander (dulu pakai
            # MECH_TERMS_ICON) -- permintaan user supaya lebih ringkas &
            # profesional, konsisten dengan label tab DT di atas.
            with st.expander(judul_istilah):
                st.write(desk.get(st.session_state['lang'], desk.get('id', '')))

# ===========================================================================
# 1. MODUL NDT
# ===========================================================================
elif menu == "ndt":
    ic1, ic2, ic3, ic4 = st.columns([1, 1, 1, 0.6])
    with ic1:
        info_tile(t("status_model_label"), "Loaded ✅" if model is not None else "Not Found ❌", accent="green" if model is not None else "orange")
    with ic2:
        info_tile(t("akurasi_model_label"), f"{model_acc:.2f}%" if model_acc is not None else "N/A", accent="blue")
    with ic3:
        info_tile(t("batch_size_label"), f"{model_batch}" if model_batch is not None else "N/A", accent="purple")
    with ic4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(tt("🔄 Muat Ulang Model", "🔄 Reload Model"), use_container_width=True,
                     help=tt("Paksa coba muat ulang model dari Drive (kalau sebelumnya gagal karena Drive belum siap saat pertama kali dicoba)",
                             "Force reload the model from Drive (in case it previously failed because Drive wasn't ready on the first attempt)")):
            load_trained_model.clear()
            load_korosi_model.clear()
            st.rerun()

    if model is None and model_load_error:
        result_box("fail", tt("Alasan model gagal dimuat (bukan cuma 'Not Found')", "Reason the model failed to load (not just 'Not Found')"), model_load_error)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander(tt("🔧 Diagnostik (klik kalau model Not Found)", "🔧 Diagnostics (click if model Not Found)")):
        st.code(tt(f"MODEL_FILE (path yang dicari): {MODEL_FILE}", f"MODEL_FILE (searched path): {MODEL_FILE}"))
        st.code(f"os.path.exists(MODEL_FILE): {os.path.exists(MODEL_FILE)}")
        # v80: cek folder models/ LOKAL juga (dipakai di hosting permanen spt
        # Streamlit Community Cloud, BUKAN Google Drive) -- supaya diagnostik ini
        # tetap berguna di 2 jenis lingkungan (Colab & hosting permanen).
        _local_models_dir = os.path.join(BASE_DIR, "models")
        st.code(tt(f"Folder lokal 'models/' (utk hosting permanen spt Streamlit Community Cloud): {_local_models_dir}\n"
                    f"Ada?: {os.path.exists(_local_models_dir)} -- isi: "
                    f"{os.listdir(_local_models_dir) if os.path.exists(_local_models_dir) else '(folder tidak ada)'}",
                   f"Local 'models/' folder (used on permanent hosting like Streamlit Community Cloud): {_local_models_dir}\n"
                    f"Exists?: {os.path.exists(_local_models_dir)} -- contents: "
                    f"{os.listdir(_local_models_dir) if os.path.exists(_local_models_dir) else '(folder does not exist)'}"))
        st.code(f"os.path.exists('/content/drive/MyDrive'): {os.path.exists('/content/drive/MyDrive')}")
        raw_glob = glob.glob('/content/drive/**/cnn_kabel_model.pth', recursive=True)
        st.code(f"glob.glob('/content/drive/**/cnn_kabel_model.pth', recursive=True):\n{raw_glob}")
        try:
            drive_listing = os.listdir('/content/drive/MyDrive') if os.path.exists('/content/drive/MyDrive') else []
        except Exception as e:
            drive_listing = [f"ERROR: {e}"]
        st.code(tt(f"Isi /content/drive/MyDrive ({len(drive_listing)} item):\n", f"Contents of /content/drive/MyDrive ({len(drive_listing)} items):\n") + "\n".join(drive_listing))

        st.markdown("---")
        st.markdown(tt("**Tes langsung `torch.load()` (buka pesan error ASLI kalau ada):**",
                        "**Direct `torch.load()` test (shows the ORIGINAL error message if any):**"))
        if os.path.exists(MODEL_FILE):
            try:
                file_size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
                st.code(tt(f"Ukuran file terbaca: {file_size_mb:.2f} MB (harusnya ~106 MB)", f"File size read: {file_size_mb:.2f} MB (should be ~106 MB)"))
                _test_ckpt = torch.load(MODEL_FILE, map_location="cpu")
                st.success(tt("✅ torch.load() BERHASIL membaca file!", "✅ torch.load() SUCCESSFULLY read the file!"))
                if isinstance(_test_ckpt, dict):
                    st.code(tt(f"Key di dalam checkpoint: {list(_test_ckpt.keys())}", f"Keys inside checkpoint: {list(_test_ckpt.keys())}"))
                    if 'class_names' in _test_ckpt:
                        st.code(f"class_names: {_test_ckpt['class_names']}")
                    if 'arch' in _test_ckpt:
                        st.code(tt(f"arch (arsitektur asli training): {_test_ckpt['arch']}", f"arch (original training architecture): {_test_ckpt['arch']}"))

                    st.markdown(tt("**Tes `load_state_dict()` dengan arsitektur ConvNeXt-Tiny:**",
                                    "**`load_state_dict()` test with ConvNeXt-Tiny architecture:**"))
                    try:
                        _test_cnames = _test_ckpt.get('class_names', [])
                        _test_model = models.convnext_tiny(weights=None)
                        _test_infeat = _test_model.classifier[2].in_features
                        _test_model.classifier[2] = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(_test_infeat, len(_test_cnames)))
                        _test_model.load_state_dict(_test_ckpt.get('model_state_dict', _test_ckpt))
                        st.success(tt("✅ load_state_dict() BERHASIL dengan arsitektur ConvNeXt-Tiny!", "✅ load_state_dict() SUCCEEDED with the ConvNeXt-Tiny architecture!"))
                    except Exception as e2:
                        st.error(f"❌ load_state_dict() {tt('GAGAL', 'FAILED')}: {type(e2).__name__}: {e2}")
                else:
                    st.code(tt(f"Tipe objek checkpoint: {type(_test_ckpt)}", f"Checkpoint object type: {type(_test_ckpt)}"))
            except Exception as e:
                import traceback
                st.error(tt(f"❌ torch.load() GAGAL dengan error:\n\n{type(e).__name__}: {e}", f"❌ torch.load() FAILED with error:\n\n{type(e).__name__}: {e}"))
                st.code(traceback.format_exc())
        else:
            st.warning(tt("File tidak ditemukan, tidak bisa tes torch.load().", "File not found, cannot test torch.load()."))

        st.caption(tt("Kalau 'glob.glob' di atas KOSONG tapi kamu yakin file-nya ada di Drive, "
                   "kemungkinan besar Drive belum benar-benar ter-mount di proses Streamlit ini. "
                   "Kalau isinya ADA tapi Status Model tetap 'Not Found', lihat pesan error torch.load() "
                   "di atas -- itu penyebab sebenarnya.",
                   "If the 'glob.glob' above is EMPTY but you're sure the file exists in Drive, "
                   "Drive most likely isn't actually mounted in this Streamlit process yet. "
                   "If it's NOT empty but Model Status is still 'Not Found', check the torch.load() error "
                   "message above -- that's the real cause."))

    tab_single, tab_batch = st.tabs([t("tab_cek_satu"), t("tab_cek_batch")])

    with tab_single:
        col_u1, col_u2 = st.columns([1, 1.2])

        uploaded_file = None
        img_status_ndt = "neutral"
        with col_u1:
            upload_header("🩻", t("upload_xray_title"), t("upload_xray_sub"))
            uploaded_file = st.file_uploader(tt("Unggah gambar X-Ray...", "Upload X-Ray image..."), type=["jpg", "png", "tif", "jpeg"], label_visibility="collapsed")
            img = None
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")

        # Jalankan inferensi LEBIH DULU (sebelum render bingkai citra), supaya
        # warna bingkainya bisa langsung mengikuti status hasil (lolos/gagal).
        pred, conf = None, None
        if uploaded_file and model and img is not None:
            pred, conf = run_inference(img)
            img_status_ndt = "pass" if "NORMAL" in pred.strip().upper() else "fail"

        with col_u1:
            framed_image(
                img,
                caption=(tt(f"Citra X-Ray — {uploaded_file.name}", f"X-Ray Image — {uploaded_file.name}") if uploaded_file
                         else tt("Citra X-Ray", "X-Ray Image")),
                icon="🩻", status=img_status_ndt,
                placeholder=tt("unggah citra radiografi untuk memulai inspeksi", "upload a radiograph image to start the inspection"),
            )
            # Baris chip kelas NDT (read-only) -- menunjukkan semua kelas yang
            # bisa dideteksi model, kelas hasil prediksi disorot. Kelas aktif
            # tetap sepenuhnya ditentukan oleh inferensi model di atas.
            class_chip_row(list(NDT_CLASS_DESC.keys()), active_label=pred, status=img_status_ndt)

        with col_u2:
            if uploaded_file and model and pred is not None:
                st.session_state['ndt_defek'] = pred
                st.session_state['ndt_conf'] = conf

                with st.container(border=True):
                    xray_gauge_card(conf * 100, "confidence", status=img_status_ndt,
                                     kelas_label=pred, model_label="ConvNeXt-Tiny",
                                     badge_text=("NDT PASSED" if "NORMAL" in pred.strip().upper() else "NDT FAILED"))

                    # PERINGATAN CONFIDENCE RENDAH (v78) -- ditaruh SEBELUM
                    # kartu hasil PASSED/FAILED supaya user melihat peringatan
                    # ini duluan kalau confidence-nya di bawah ambang ragu.
                    # Tujuannya menangkap kasus seperti "citra mikrografi
                    # keliru diunggah ke tab NDT X-Ray" -- model tetap
                    # mengeluarkan 1 label (softmax selalu 100%), tapi
                    # confidence-nya biasanya jatuh rendah krn citranya memang
                    # bukan dari distribusi radiograf X-Ray yang dipelajari
                    # model. Ini SEKADAR sinyal untuk user, TIDAK mengubah
                    # ndt_lolos/status LOLOS-TIDAK LOLOS/keputusan akhir.
                    if conf < AMBANG_CONF_RAGU_NDT:
                        result_box(
                            "warn",
                            tt("Confidence rendah — periksa kembali citra yang diunggah",
                               "Low confidence — please double-check the uploaded image"),
                            tt(f"Model hanya {conf*100:.1f}% yakin dengan klasifikasi ini (di bawah ambang {AMBANG_CONF_RAGU_NDT*100:.0f}%). "
                               "Ini bisa jadi tanda citra yang diunggah BUKAN radiograf X-Ray yang valid — misalnya keliru mengunggah "
                               "citra mikrografi atau foto lain ke tab ini. Pastikan file yang diunggah benar sebelum menggunakan hasil di bawah.",
                               f"The model is only {conf*100:.1f}% confident in this classification (below the {AMBANG_CONF_RAGU_NDT*100:.0f}% threshold). "
                               "This can be a sign that the uploaded image is NOT a valid X-ray radiograph — for example, a micrography image "
                               "or another photo was uploaded to this tab by mistake. Confirm the uploaded file is correct before relying on the result below."))

                    pred_key = pred.strip().upper()
                    cond_icon = NDT_CLASS_ICON.get(pred_key, "❔")
                    _cond_desc_entry = NDT_CLASS_DESC.get(pred_key)
                    cond_desc = (_cond_desc_entry.get(st.session_state['lang'], _cond_desc_entry.get('id'))
                                 if _cond_desc_entry else t("desc_unavailable"))

                    if "NORMAL" in pred_key:
                        st.session_state['ndt_status'] = "LOLOS"
                        if st.session_state['lang'] == 'en':
                            reason = f"The model classified the image as '{pred}' with {conf*100:.1f}% confidence, meeting the required NORMAL class threshold."
                        else:
                            reason = f"Model mengklasifikasikan citra sebagai '{pred}' dengan confidence {conf*100:.1f}%, sesuai ambang batas kelas NORMAL yang dipersyaratkan."
                        result_box_detailed("pass", cond_icon, f"NDT PASSED — {pred}", reason,
                                             extra_label=t("keterangan_kondisi"), extra_text=cond_desc)
                    else:
                        st.session_state['ndt_status'] = "TIDAK LOLOS"
                        if st.session_state['lang'] == 'en':
                            reason = f"The model detected a defect indication '{pred}' with {conf*100:.1f}% confidence. This condition indicates internal deformation (not the NORMAL class), so the conductor needs further verification via DT testing."
                        else:
                            reason = f"Model mendeteksi indikasi cacat '{pred}' dengan confidence {conf*100:.1f}%. Kondisi ini menunjukkan deformasi internal (bukan kelas NORMAL) sehingga konduktor perlu diverifikasi lebih lanjut melalui uji DT."
                        result_box_detailed("fail", cond_icon, f"NDT FAILED — {pred}", reason,
                                             extra_label=t("keterangan_kondisi"), extra_text=cond_desc)
            elif uploaded_file and not model:
                result_box("warn", t("model_not_found_title"), t("model_not_found_body"))
            else:
                st.info(t("info_upload_xray_prompt"))

    with tab_batch:
        st.caption(tt("Periksa banyak citra sekaligus — bisa langsung dari folder Google Drive kamu (lebih praktis untuk ratusan citra), atau upload manual.",
                      "Check many images at once — either directly from your Google Drive folder (more practical for hundreds of images), or manual upload."))
        source_mode = st.radio(tt("Sumber citra:", "Image source:"),
                                [tt("📁 Folder di Google Drive", "📁 Google Drive Folder"), tt("⬆️ Upload manual", "⬆️ Manual upload")],
                                horizontal=True)

        batch_items = []  # list of (nama, PIL.Image)

        if source_mode == tt("📁 Folder di Google Drive", "📁 Google Drive Folder"):
            folder_path = st.text_input(
                tt("Path folder di Drive (berisi citra X-Ray):", "Drive folder path (containing X-Ray images):"),
                placeholder="/content/drive/MyDrive/Dataset/XRay_600",
            )
            if folder_path:
                if os.path.isdir(folder_path):
                    found = []
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                        found.extend(glob.glob(os.path.join(folder_path, ext)))
                    found = sorted(set(found))
                    st.write(tt(f"**{len(found)} citra ditemukan** di folder tersebut.", f"**{len(found)} images found** in that folder."))
                    batch_items = [(os.path.basename(fp), fp) for fp in found]
                else:
                    result_box("warn", tt("Folder tidak ditemukan", "Folder not found"),
                               tt(f"Path `{folder_path}` tidak ada atau Drive belum ter-mount. Pastikan Drive sudah di-mount dan path-nya benar (bisa di-copy dari panel file kiri Colab).",
                                  f"Path `{folder_path}` doesn't exist or Drive isn't mounted yet. Make sure Drive is mounted and the path is correct (you can copy it from Colab's left file panel)."))
        else:
            upload_header("📦", t("upload_batch_title"), t("upload_batch_sub"))
            batch_files = st.file_uploader(tt("Unggah beberapa citra X-Ray sekaligus", "Upload several X-Ray images at once"), type=["jpg", "png", "jpeg"],
                                            accept_multiple_files=True, key="batch_ndt", label_visibility="collapsed")
            if batch_files:
                batch_items = [(f.name, f) for f in batch_files]

        pred_col = tt("Prediksi", "Prediction")
        if batch_items and model:
            if st.button(tt(f"🚀 Jalankan Inspeksi Batch ({len(batch_items)} citra)", f"🚀 Run Batch Inspection ({len(batch_items)} images)")):
                progress = st.progress(0, text=tt("Memproses citra...", "Processing images..."))
                results = []
                for i, (name, src) in enumerate(batch_items):
                    try:
                        im = Image.open(src).convert("RGB")
                        pred, conf = run_inference(im)
                        results.append({"File": name, pred_col: pred, "Confidence (%)": round(conf * 100, 1)})
                    except Exception as e:
                        results.append({"File": name, pred_col: f"Error: {e}", "Confidence (%)": 0})
                    progress.progress((i + 1) / len(batch_items), text=tt(f"Memproses {i+1}/{len(batch_items)} citra...", f"Processing {i+1}/{len(batch_items)} images..."))
                progress.empty()

                df_batch = pd.DataFrame(results)
                st.dataframe(df_batch, use_container_width=True, hide_index=True)

                summary = df_batch[pred_col].value_counts()
                fig = go.Figure(go.Bar(
                    x=summary.index.tolist(), y=summary.values.tolist(),
                    marker_color=["#1E7F4E" if "NORMAL" in c.upper() else "#C8443C" for c in summary.index],
                    text=summary.values.tolist(), textposition="auto",
                ))
                fig.update_layout(title=tt("Ringkasan Hasil Inspeksi Batch", "Batch Inspection Result Summary"), height=320,
                                   xaxis_title=tt("Kelas Prediksi", "Predicted Class"), yaxis_title=tt("Jumlah Citra", "Number of Images"))
                st.plotly_chart(fig, use_container_width=True)

                n_normal = sum("NORMAL" in c.upper() for c in df_batch[pred_col])
                pct_normal = n_normal / len(df_batch) * 100
                if pct_normal >= 80:
                    result_box("pass", f"{pct_normal:.1f}% " + tt("batch dalam kondisi NORMAL", "of batch is in NORMAL condition"),
                               tt(f"Dari {len(df_batch)} citra yang diperiksa, {n_normal} citra ({pct_normal:.1f}%) terklasifikasi NORMAL — proporsi ini menunjukkan kondisi keseluruhan batch tergolong baik.",
                                  f"Of the {len(df_batch)} images checked, {n_normal} images ({pct_normal:.1f}%) were classified as NORMAL — this proportion indicates the overall batch condition is good."))
                else:
                    result_box("warn", f"{tt('Hanya', 'Only')} {pct_normal:.1f}% " + tt("batch dalam kondisi NORMAL", "of batch is in NORMAL condition"),
                               tt(f"Dari {len(df_batch)} citra, hanya {n_normal} ({pct_normal:.1f}%) yang NORMAL — proporsi cacat cukup tinggi, disarankan pemeriksaan DT menyeluruh pada spesimen terkait.",
                                  f"Of the {len(df_batch)} images, only {n_normal} ({pct_normal:.1f}%) are NORMAL — the defect proportion is quite high, a thorough DT inspection of the related specimen is recommended."))
        elif batch_items and not model:
            result_box("warn", tt("Model tidak ditemukan", "Model not found"), tt("Tidak bisa menjalankan batch check tanpa model ter-load.", "Cannot run batch check without a loaded model."))

# ===========================================================================
# 2. MODUL DT
# ===========================================================================
elif menu == "dt":
    t1, t2, t3, t4 = st.tabs([t("tab_mikrografi"), t("tab_tarik"), t("tab_torsi"), t("tab_lilit")])

    # --- MIKROGRAFI: analisis otomatis begitu citra diunggah ---
    with t1:
        upload_header("🔬", t("upload_mikro_title"), t("upload_mikro_sub"))
        img_m = st.file_uploader(tt("Unggah Mikrografi:", "Upload Micrograph:"), type=["jpg", "png"], label_visibility="collapsed")

        mag_used = st.number_input(
            t("mag_label"),
            min_value=50, max_value=1000, value=500, step=50,
            help=t("mag_help") + " " + tt(
                "Jumlah butir dideteksi otomatis dari citra (OpenCV) lalu dimasukkan ke formula Jeffries "
                "(ASTM E112) -- verifikasi konstanta rumus terhadap SOP lab kamu, lihat fungsi astm_grain_size() di app.py.",
                "Grain count is auto-detected from the image (OpenCV) then fed into the Jeffries formula "
                "(ASTM E112) -- verify the formula constants against your lab SOP, see the astm_grain_size() function in app.py.")
        )
        is_surface_only_mag = mag_used < 100

        # Hitung analisis LEBIH DULU (sebelum render bingkai citra), supaya
        # warna bingkainya bisa langsung mengikuti status hasil analisis.
        img_status = "neutral"
        corrosion_prob, is_smooth, detected_G, korosi_pred = 0.0, True, None, "-"
        grain_calc_note = ""
        if img_m is not None:
            pil_img_mikro = Image.open(img_m).convert("RGB")

            if korosi_model is not None:
                korosi_pred, korosi_conf = run_korosi_inference(pil_img_mikro)
                is_smooth = "Mulus" in korosi_pred
                corrosion_prob = (1 - korosi_conf) if is_smooth else korosi_conf
            else:
                korosi_pred, corrosion_prob, is_smooth = tt("Model tidak ditemukan", "Model not found"), 0.0, True

            if is_surface_only_mag:
                # 50x: butir belum kelihatan jelas -- jangan hitung grain size,
                # pakai nilai manual/sebelumnya yang tersimpan di session_state.
                detected_G = st.session_state.get('astm_G', None)
                grain_calc_note = tt(
                    "Perbesaran 50x dipakai khusus untuk evaluasi kondisi permukaan — "
                    "grain size TIDAK dihitung otomatis di perbesaran ini (butir belum terlihat jelas). "
                    "Naikkan ke 200x-1000x untuk perhitungan grain size otomatis.",
                    "50x magnification is used specifically for surface condition evaluation — "
                    "grain size is NOT auto-calculated at this magnification (grains aren't clearly visible yet). "
                    "Increase to 200x-1000x for automatic grain size calculation."
                )
            else:
                n_grains, _thresh_debug = count_grains_cv2(pil_img_mikro)
                raw_G = astm_grain_size(n_grains, mag_used)
                # Guard: kalau hasil di luar rentang wajar praktis (deteksi kontur
                # OpenCV kemungkinan gagal / kegagalan segmentasi), JANGAN tampilkan
                # angka mentah yang membingungkan (mis. negatif) -- tandai tidak valid,
                # minta koreksi manual, daripada menyesatkan pembaca jurnal.
                if raw_G is None or raw_G < 1.0 or raw_G > 14.0 or n_grains < 3:
                    detected_G = None
                    grain_calc_note = tt(
                        f"Deteksi otomatis jumlah butir kemungkinan gagal (terdeteksi {n_grains} kontur, "
                        f"hasil hitung G={raw_G} di luar rentang wajar). Ini biasanya karena kontras citra "
                        f"kurang jelas untuk segmentasi otomatis. Silakan isi nilai G secara manual di bawah.",
                        f"Automatic grain count detection likely failed (detected {n_grains} contours, "
                        f"computed G={raw_G} is out of the reasonable range). This is usually because the image "
                        f"contrast isn't clear enough for automatic segmentation. Please enter the G value manually below."
                    )
                else:
                    detected_G = raw_G

            img_status = "pass" if (is_smooth and detected_G is not None and detected_G >= 6.0) else "fail"
            if detected_G is not None:
                st.session_state['astm_G'] = detected_G
            st.session_state['korosi_status'] = "Mikro_Mulus" if is_smooth else "Mikro_Korosi"

        # Layout v74 mengikuti referensi tampilan terbaru yang dikirim user
        # (posisi persis, bukan cuma "mirip" seperti sebelumnya): KIRI = citra
        # (dalam bingkai) + DUA info-tile kecil di bawahnya (Klasifikasi
        # Permukaan, ASTM Grain Size G) -- dulu nilai ini cuma disebut di
        # teks caption kanan, sekarang jadi kartu tersendiri persis di bawah
        # citra. KANAN = donut probabilitas + teks keterangan digabung jadi
        # SATU kartu putih (st.container(border=True)), lalu kartu hasil
        # (result_box) tetap di bawahnya seperti sebelumnya.
        c1, c2 = st.columns(2)
        with c1:
            framed_image(
                pil_img_mikro if img_m else None,
                caption=(tt(f"Citra Mikrografi — {img_m.name}", f"Micrograph Image — {img_m.name}") if img_m
                         else tt("Citra Mikrografi", "Micrograph Image")),
                icon="🔬", status=img_status,
                placeholder=tt("unggah citra mikrografi, atau isi G di panel koreksi manual", "upload a micrograph image, or fill in G in the manual correction panel"),
            )
            if img_m is not None:
                tile_klas, tile_g = st.columns(2)
                with tile_klas:
                    info_tile(
                        tt("Klasifikasi Permukaan", "Surface Classification"), korosi_pred,
                        accent="green" if is_smooth else "orange",
                        value_color=(PASS_COLOR if is_smooth else FAIL_COLOR))
                with tile_g:
                    _g_ok_tile = detected_G is not None and detected_G >= 6.0
                    info_tile(
                        "ASTM Grain Size (G)", f"{detected_G:.1f}" if detected_G is not None else "—",
                        accent="green" if _g_ok_tile else "orange",
                        value_color=(PASS_COLOR if _g_ok_tile else None))
        with c2:
            with st.container(border=True):
                cd1, cd2 = st.columns([1, 1.3], vertical_alignment="center")
                if img_m is None:
                    with cd1:
                        st.plotly_chart(ring_chart(0.0, tt("Probabilitas Korosi", "Corrosion Probability"), threshold=50.0, invert_color=True),
                                         use_container_width=True, config={"displayModeBar": False})
                    with cd2:
                        st.markdown(f'<div class="donut-card-label">{tt("PROBABILITAS KOROSI PERMUKAAN", "SURFACE CORROSION PROBABILITY")}</div>', unsafe_allow_html=True)
                        st.caption(tt(
                            "Ambang peringatan 50%. Klasifikasi permukaan memakai model ResNet18 dua kelas "
                            "(Mikro_Mulus / Mikro_Korosi).",
                            "Warning threshold 50%. Surface classification uses a two-class ResNet18 model "
                            "(Mikro_Mulus / Mikro_Korosi)."))
                else:
                    with cd1:
                        st.plotly_chart(ring_chart(corrosion_prob * 100, tt("Probabilitas Korosi", "Corrosion Probability"), threshold=50.0, invert_color=True),
                                         use_container_width=True, config={"displayModeBar": False})
                    with cd2:
                        st.markdown(f'<div class="donut-card-label">{tt("PROBABILITAS KOROSI PERMUKAAN", "SURFACE CORROSION PROBABILITY")}</div>', unsafe_allow_html=True)
                        st.caption(tt(
                            f"Ambang peringatan 50%. Klasifikasi permukaan: **{korosi_pred}** · "
                            f"ASTM Grain Size G = **{detected_G:.1f}**." if detected_G is not None else
                            f"Ambang peringatan 50%. Klasifikasi permukaan: **{korosi_pred}** · Grain Size G belum terisi.",
                            f"Warning threshold 50%. Surface classification: **{korosi_pred}** · "
                            f"ASTM Grain Size G = **{detected_G:.1f}**." if detected_G is not None else
                            f"Warning threshold 50%. Surface classification: **{korosi_pred}** · Grain Size G not filled in."))

            if img_m is None:
                st.info(t("info_upload_mikro_prompt"))
            if img_m is not None:
                if korosi_model is None:
                    st.caption(f"⚠️ {tt('Model korosi tidak ditemukan di Drive', 'Corrosion model not found in Drive')} — "
                               f"{tt('cek path pencarian', 'check the search path for')} `cnn_mikro_korosi_model.pth` "
                               f"{tt('di fungsi', 'in the')} `load_korosi_model()`.")

                if grain_calc_note:
                    result_box("warn", tt("Catatan Perhitungan Grain Size", "Grain Size Calculation Note"), grain_calc_note)

                G_ok = detected_G is not None and detected_G >= 6.0
                if is_smooth and G_ok:
                    reason = tt(
                        f"Model mengklasifikasikan permukaan sebagai '{korosi_pred}', dan hasil hitung grain size G={detected_G:.1f} (perbesaran {mag_used}x) memenuhi ambang minimum G≥6.0 — struktur mikro tergolong baik.",
                        f"The model classified the surface as '{korosi_pred}', and the computed grain size G={detected_G:.1f} ({mag_used}x magnification) meets the minimum threshold G≥6.0 — the microstructure is classified as good.")
                    st.session_state['korosi_reason'] = reason
                    result_box("pass", tt("Kondisi Mikrografi: BAIK", "Micrography Condition: GOOD"), reason)
                elif not is_smooth:
                    reason = tt(
                        f"Model mendeteksi indikasi korosi pada permukaan (klasifikasi: '{korosi_pred}') — kondisi ini berpotensi menurunkan integritas mekanis konduktor dan perlu ditindaklanjuti.",
                        f"The model detected a corrosion indication on the surface (classification: '{korosi_pred}') — this condition may reduce the conductor's mechanical integrity and needs follow-up.")
                    st.session_state['korosi_reason'] = reason
                    result_box("fail", tt("Kondisi Mikrografi: TERINDIKASI KOROSI", "Micrography Condition: CORROSION INDICATED"), reason)
                elif detected_G is None:
                    reason = tt(
                        "Permukaan terklasifikasi mulus, namun nilai grain size G belum valid/belum diisi — lengkapi lewat koreksi manual di bawah sebelum kesimpulan akhir diambil.",
                        "The surface is classified as smooth, but the grain size G value isn't valid/filled in yet — complete it via the manual correction below before a final conclusion is drawn.")
                    st.session_state['korosi_reason'] = reason
                    result_box("warn", tt("Kondisi Mikrografi: G BELUM TERISI", "Micrography Condition: G NOT FILLED IN"), reason)
                else:
                    reason = tt(
                        f"Permukaan terklasifikasi mulus, namun ukuran butir G={detected_G:.1f} (perbesaran {mag_used}x) di bawah ambang minimum G≥6.0 — struktur butir tergolong kasar dan berpotensi menurunkan kekuatan mekanis.",
                        f"The surface is classified as smooth, but the grain size G={detected_G:.1f} ({mag_used}x magnification) is below the minimum threshold G≥6.0 — the grain structure is classified as coarse and may reduce mechanical strength.")
                    st.session_state['korosi_reason'] = reason
                    result_box("warn", tt("Kondisi Mikrografi: GRAIN SIZE KURANG", "Micrography Condition: GRAIN SIZE INSUFFICIENT"), reason)

                with st.expander(t("koreksi_manual")):
                    st.session_state['korosi_status'] = st.radio(
                        t("status_korosi_label"), ["Mikro_Mulus", "Mikro_Korosi"],
                        index=0 if st.session_state['korosi_status'] == "Mikro_Mulus" else 1,
                        format_func=lambda v: tt("Mulus (tidak ada korosi)", "Smooth (no corrosion)") if v == "Mikro_Mulus" else tt("Terindikasi Korosi", "Corrosion Indicated"))
                    st.session_state['astm_G'] = st.number_input(t("label_G_manual"), value=float(st.session_state['astm_G']), step=0.1)

    # Grafik benchmark (aktual vs standar minimum) SEKARANG SELALU tampil
    # berdampingan dengan input & kotak hasil (sebelumnya disembunyikan di
    # balik tombol "Tampilkan Grafik") -- supaya penyajian data hasil
    # pengujian langsung terlihat begitu nilai diisi, tidak perlu klik dulu.
    with t2:
        ca_l, ca_r = st.columns([1.15, 1])
        with ca_l:
            with st.container(border=True):
                st.markdown(f"##### {t('section_al')}")
                ca1, ca2 = st.columns(2)
                st.session_state['tensile_al'] = ca1.number_input(t("label_tensile_al"), value=float(st.session_state['tensile_al']))
                st.session_state['elong_al'] = ca2.number_input(t("label_elong_al"), value=float(st.session_state['elong_al']))

                p_al, r_al = evaluate_reason("Tensile Al", st.session_state['tensile_al'], acsr_spec['min_tensile_al'], "MPa")
                p_el, r_el = evaluate_reason("Elongasi Al", st.session_state['elong_al'], acsr_spec['min_elong_al'], "%")
                result_box("pass" if (p_al and p_el) else "fail", t("eval_al_title"), r_al + " " + r_el)
        with ca_r:
            st.plotly_chart(create_benchmark_chart(tt("Tensile Alumunium", "Aluminium Tensile"), st.session_state['tensile_al'], acsr_spec['min_tensile_al'], "MPa"),
                             use_container_width=True, config={"displayModeBar": False})

        cs_l, cs_r = st.columns([1.15, 1])
        with cs_l:
            with st.container(border=True):
                st.markdown(f"##### {t('section_steel')}")
                cs1, cs2 = st.columns(2)
                st.session_state['tensile_steel'] = cs1.number_input(t("label_tensile_steel"), value=float(st.session_state['tensile_steel']))
                st.session_state['elong_steel'] = cs2.number_input(t("label_elong_steel"), value=float(st.session_state['elong_steel']))

                p_st, r_st = evaluate_reason("Tensile Steel", st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], "MPa")
                p_els, r_els = evaluate_reason("Elongasi Steel", st.session_state['elong_steel'], acsr_spec['min_elong_steel'], "%")
                result_box("pass" if (p_st and p_els) else "fail", t("eval_steel_title"), r_st + " " + r_els)
        with cs_r:
            st.plotly_chart(create_benchmark_chart(tt("Tensile Inti Baja", "Steel Core Tensile"), st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], "MPa"),
                             use_container_width=True, config={"displayModeBar": False})

    with t3:
        tor_l, tor_r = st.columns([1.15, 1])
        with tor_l:
            with st.container(border=True):
                st.session_state['torsion'] = st.number_input(t("label_torsi"), value=float(st.session_state['torsion']))
                p_tor, r_tor = evaluate_reason("Torsi Puntir", st.session_state['torsion'], acsr_spec['min_torsi'], "N.m")
                result_box("pass" if p_tor else "fail", t("eval_torsi_title"), r_tor)
        with tor_r:
            st.plotly_chart(create_benchmark_chart(tt("Torsi Puntir", "Torsion"), st.session_state['torsion'], acsr_spec['min_torsi'], "N.m"),
                             use_container_width=True, config={"displayModeBar": False})

    with t4:
        lil_l, lil_r = st.columns([1.15, 1])
        with lil_l:
            with st.container(border=True):
                st.session_state['turns'] = st.number_input(t("label_lilit"), value=int(st.session_state['turns']))
                p_lil, r_lil = evaluate_reason("Jumlah Lilitan", st.session_state['turns'], acsr_spec['min_lilit'], "Putaran")
                result_box("pass" if p_lil else "fail", t("eval_lilit_title"), r_lil)
        with lil_r:
            st.plotly_chart(create_benchmark_chart(tt("Jumlah Lilitan", "Number of Wraps"), st.session_state['turns'], acsr_spec['min_lilit'], "Turns"),
                             use_container_width=True, config={"displayModeBar": False})

# ===========================================================================
# 3. MODUL DECISION
# ===========================================================================
elif menu == "keputusan":
    # --- Hitung semua evaluasi & alasan LEBIH DULU, supaya bisa ditampilkan
    #     langsung di kartu ringkasan (bukan cuma di expander bawah) ---
    al_pass, r_al = evaluate_reason("Tensile Al", st.session_state['tensile_al'], acsr_spec['min_tensile_al'], "MPa")
    al_pass2, r_al2 = evaluate_reason("Elongasi Al", st.session_state['elong_al'], acsr_spec['min_elong_al'], "%")
    steel_pass, r_st = evaluate_reason("Tensile Steel", st.session_state['tensile_steel'], acsr_spec['min_tensile_steel'], "MPa")
    steel_pass2, r_st2 = evaluate_reason("Elongasi Steel", st.session_state['elong_steel'], acsr_spec['min_elong_steel'], "%")
    tor_pass, r_tor = evaluate_reason("Torsi Puntir", st.session_state['torsion'], acsr_spec['min_torsi'], "N.m")
    lil_pass, r_lil = evaluate_reason("Jumlah Lilitan", st.session_state['turns'], acsr_spec['min_lilit'], "Putaran")
    al_ok, steel_ok, tor_ok = (al_pass and al_pass2), (steel_pass and steel_pass2), (tor_pass and lil_pass)

    # --- Info spesimen: ukuran & diameter (2 kartu) + Standar Acuan SEBAGAI
    # TABEL di bawahnya (sebelumnya 1 kartu sempit berisi 1 kalimat panjang
    # dalam font mono yang wrapping tidak rapi -- sekarang dipecah per baris
    # standar supaya mudah dipindai sekilas). Isinya sama untuk ketiga tipe
    # ACSR (acsr_spec['standar_acuan'] identik di ACSR_DATABASE), jadi aman
    # dipetakan statis di sini.
    lolos_txt, tidak_lolos_txt = tt("✅ Lolos", "✅ Passed"), tt("❌ Tidak Lolos", "❌ Not Passed")
    sp1, sp2 = st.columns(2)
    with sp1:
        info_tile(tt("Tipe & Ukuran Konduktor", "Conductor Type & Size"), selected_acsr_type, accent="blue")
    with sp2:
        info_tile(tt("Diameter Konduktor", "Conductor Diameter"), f"{acsr_spec['diameter_mm']} mm ({acsr_spec['jumlah_kawat']})", accent="blue")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    STANDAR_TABLE = [
        ("ASTM B232", tt("Konstruksi konduktor ACSR", "ACSR conductor construction")),
        ("ASTM E8 / B498", tt("Uji tarik kawat alumunium & inti baja", "Aluminium & steel core wire tensile test")),
        ("ASTM E112", tt("Mikrografi & ukuran butir (grain size)", "Micrography & grain size")),
        ("IEC 60888", tt("Uji torsi puntir & uji lilitan", "Torsion test & wrap test")),
    ]
    with st.container(border=True):
        st.markdown(f"###### 📐 {tt('Standar Acuan', 'Reference Standards')}")
        col_std, col_cakupan = tt("Standar", "Standard"), tt("Cakupan Pengujian", "Test Coverage")
        st.dataframe(
            pd.DataFrame([{col_std: kode, col_cakupan: cakupan} for kode, cakupan in STANDAR_TABLE]),
            use_container_width=True, hide_index=True,
            column_config={
                col_std: st.column_config.TextColumn(width="small"),
                col_cakupan: st.column_config.TextColumn(width="large"),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        info_tile("NDT X-Ray", f"{st.session_state['ndt_status']} ({st.session_state['ndt_defek']})",
                   accent="blue",
                   explanation=tt(f"Kondisi terdeteksi: {st.session_state['ndt_defek']}. Lihat Kamus Istilah NDT di Beranda untuk arti kondisi ini.",
                                  f"Detected condition: {st.session_state['ndt_defek']}. See the NDT Glossary on the Home page for what this condition means."))
    with m2:
        info_tile(tt("Mikrografi", "Micrography"), f"{st.session_state['korosi_status']} (G={st.session_state['astm_G']})",
                   accent="blue",
                   explanation=st.session_state['korosi_reason'] or tt("Belum dianalisis — unggah citra mikrografi di Modul DT.", "Not yet analyzed — upload a micrograph image in the DT Module."))
    with m3:
        info_tile(tt("Tarik Al / Steel", "Al / Steel Tensile"), f"{st.session_state['tensile_al']} / {st.session_state['tensile_steel']} MPa",
                   accent="blue",
                   explanation=(r_al if not al_pass else r_st) if not (al_ok and steel_ok) else tt("Kedua kawat (Al & Baja) memenuhi standar tarik & elongasi minimum.", "Both wires (Al & Steel) meet the minimum tensile & elongation standards."))
    with m4:
        info_tile(tt("Torsi / Lilit", "Torsion / Wrap"), f"{st.session_state['torsion']} N.m / {st.session_state['turns']} Turns",
                   accent="blue",
                   explanation=(r_tor if not tor_pass else r_lil) if not tor_ok else tt("Torsi puntir & jumlah lilitan memenuhi ambang batas minimum.", "Torsion & number of wraps meet the minimum threshold."))

    st.divider()
    final_pass = (st.session_state['ndt_status'] == "LOLOS") or (al_ok and steel_ok and tor_ok)

    all_reasons = [
        f"NDT X-Ray: {st.session_state['ndt_defek']} — status {st.session_state['ndt_status']}.",
        st.session_state['korosi_reason'] or "Mikrografi belum dianalisis.",
        r_al, r_al2, r_st, r_st2, r_tor, r_lil,
    ]

    if final_pass:
        reason_final = tt(
            "Seluruh parameter kritis (NDT dan/atau DT) memenuhi ambang batas standar yang dipersyaratkan untuk spesifikasi " + selected_acsr_type + ".",
            "All critical parameters (NDT and/or DT) meet the required standard thresholds for the " + selected_acsr_type + " specification.")
        result_box("pass", f"{tt('REKOMENDASI', 'RECOMMENDATION')}: {selected_acsr_type} — {tt('LAYAK PAKAI', 'FIT FOR USE')}", reason_final)
    else:
        failed_params_id = [name for name, ok in [("Tarik Alumunium", al_ok), ("Tarik Inti Baja", steel_ok), ("Torsi/Lilit", tor_ok)] if not ok]
        failed_params_en = [name for name, ok in [("Aluminium Tensile", al_ok), ("Steel Core Tensile", steel_ok), ("Torsion/Wrap", tor_ok)] if not ok]
        reason_final = tt(
            "Parameter yang TIDAK memenuhi standar: " + ", ".join(failed_params_id) + ". Lihat rincian alasan tiap parameter di atas untuk detail evaluasi.",
            "Parameters that do NOT meet the standard: " + ", ".join(failed_params_en) + ". See the reasoning detail for each parameter above for evaluation details.")
        result_box("fail", f"{tt('REKOMENDASI', 'RECOMMENDATION')}: {selected_acsr_type} — {tt('AFKIR (TIDAK LAYAK PAKAI)', 'REJECTED (NOT FIT FOR USE)')}", reason_final)

    st.markdown("<br>", unsafe_allow_html=True)
    col_parameter, col_hasil, col_standar, col_status, col_alasan = (
        tt("Parameter", "Parameter"), tt("Hasil", "Result"), tt("Standar Min.", "Min. Standard"),
        tt("Status", "Status"), tt("Alasan", "Reason"))
    # Dipindah dari dalam expander (tersembunyi) ke kartu SELALU TERBUKA --
    # ini tabel utama yang dicetak juga ke Sertifikat PDF, jadi presentasinya
    # dibuat lebih menonjol, bukan sesuatu yang harus diklik dulu untuk dilihat.
    with st.container(border=True):
        st.markdown(f"##### 📋 {tt('Rincian Data & Alasan Hasil Pengujian', 'Test Result Data & Reasoning Detail')}")
        st.caption(tt("Tabel ini yang dicetak pada Sertifikat Validasi PDF di bawah.",
                       "This table is what gets printed on the PDF Validation Certificate below."))
        detail_rows = [
            {col_parameter: "NDT X-Ray", col_hasil: st.session_state['ndt_defek'],
             col_standar: "Class NORMAL", col_status: lolos_txt if st.session_state['ndt_status'] == "LOLOS" else tidak_lolos_txt,
             col_alasan: f"NDT X-Ray: {st.session_state['ndt_defek']} — status {st.session_state['ndt_status']}."},
            {col_parameter: tt("Mikrografi", "Micrography"), col_hasil: st.session_state['korosi_status'],
             col_standar: tt("Mulus, G ≥ 6.0", "Smooth, G ≥ 6.0"), col_status: lolos_txt if "Mulus" in st.session_state['korosi_status'] else tidak_lolos_txt,
             col_alasan: st.session_state['korosi_reason'] or tt("Mikrografi belum dianalisis.", "Micrography not yet analyzed.")},
            {col_parameter: "Tensile Al", col_hasil: f"{st.session_state['tensile_al']:.2f} MPa",
             col_standar: f"{acsr_spec['min_tensile_al']:.2f} MPa", col_status: lolos_txt if al_pass else tidak_lolos_txt, col_alasan: r_al},
            {col_parameter: tt("Elongasi Al", "Al Elongation"), col_hasil: f"{st.session_state['elong_al']:.2f} %",
             col_standar: f"{acsr_spec['min_elong_al']:.2f} %", col_status: lolos_txt if al_pass2 else tidak_lolos_txt, col_alasan: r_al2},
            {col_parameter: "Tensile Steel", col_hasil: f"{st.session_state['tensile_steel']:.2f} MPa",
             col_standar: f"{acsr_spec['min_tensile_steel']:.2f} MPa", col_status: lolos_txt if steel_pass else tidak_lolos_txt, col_alasan: r_st},
            {col_parameter: tt("Elongasi Steel", "Steel Elongation"), col_hasil: f"{st.session_state['elong_steel']:.2f} %",
             col_standar: f"{acsr_spec['min_elong_steel']:.2f} %", col_status: lolos_txt if steel_pass2 else tidak_lolos_txt, col_alasan: r_st2},
            {col_parameter: tt("Torsi Puntir", "Torsion"), col_hasil: f"{st.session_state['torsion']:.2f} N.m",
             col_standar: f"{acsr_spec['min_torsi']:.2f} N.m", col_status: lolos_txt if tor_pass else tidak_lolos_txt, col_alasan: r_tor},
            {col_parameter: tt("Jumlah Lilitan", "Number of Wraps"), col_hasil: tt(f"{st.session_state['turns']:.2f} Putaran", f"{st.session_state['turns']:.2f} Turns"),
             col_standar: tt(f"{acsr_spec['min_lilit']:.2f} Putaran", f"{acsr_spec['min_lilit']:.2f} Turns"), col_status: lolos_txt if lil_pass else tidak_lolos_txt, col_alasan: r_lil},
        ]
        df_detail = pd.DataFrame(detail_rows)
        st.dataframe(
            df_detail, use_container_width=True, hide_index=True,
            column_config={
                col_parameter: st.column_config.TextColumn(width="small"),
                col_hasil: st.column_config.TextColumn(width="small"),
                col_standar: st.column_config.TextColumn(width="small"),
                col_status: st.column_config.TextColumn(width="small"),
                col_alasan: st.column_config.TextColumn(width="large"),
            },
        )

    st.divider()
    # Kartu CTA gradient navy untuk cetak PDF -- dibungkus st.container(key=...)
    # supaya CSS ".st-key-pdf_cta_card" (lihat apply_custom_css di utils/ui.py)
    # bisa menyasarnya secara ANDAL, pola yang sama seperti header_nav_tabs &
    # language_toggle. st.download_button tetap dirender sebagai widget asli
    # DI DALAM kartu (bukan HTML statis) supaya tombolnya tetap bisa diklik.
    # v74: teks judul & tombol dibuat PERSIS seperti referensi yang dikirim
    # user -- emoji "📄"/"📥" dihapus (referensinya polos tanpa ikon).
    with st.container(key="pdf_cta_card"):
        st.markdown(f"""
        <div class="pdf-cta-title">{tt('Cetak bukti validasi resmi', 'Print official validation proof')}</div>
        <p class="pdf-cta-desc">{tt(
            'Sertifikat memuat tabel rincian, grafik pendukung uji mekanis, kesimpulan naratif, '
            'serta kolom tanda tangan peneliti dan kepala laboratorium.',
            'The certificate includes a detail table, supporting mechanical test charts, a narrative '
            'conclusion, and signature columns for the researcher and lab head.'
        )}</p>
        """, unsafe_allow_html=True)
        pdf_file = generate_pdf_report(all_reasons)
        st.download_button(
            label=tt("Download Sertifikat Validasi PDF", "Download PDF Validation Certificate"),
            data=pdf_file,
            file_name=f"Sertifikat_Validasi_ACSR_{selected_acsr_type.split()[1]}_PLN_UNDIP.pdf",
            mime="application/pdf",
            type="primary"
        )
    # Tombol "Simpan ke Riwayat Pengujian" yang dulu di sini (v73) SUDAH
    # PINDAH ke header utama ("Simpan Pengujian", lihat blok header di atas
    # app.py) supaya bisa diklik dari halaman manapun, bukan cuma di sini --
    # tidak diduplikasi lagi di halaman Keputusan.

# ===========================================================================
# 4. RIWAYAT PENGUJIAN -- daftar snapshot hasil pengujian yang disimpan lewat
# tombol di halaman Keputusan. HANYA session_state (tidak ditulis ke file/
# Drive), sesuai keputusan eksplisit user -- riwayat hilang begitu sesi
# browser/Colab berakhir.
# ===========================================================================
elif menu == "riwayat":
    st.markdown(f"""
    <div class="page-title-block">
        <h1>{t("nav_riwayat")}</h1>
        <p class="page-subtitle">{tt(
            "Daftar hasil pengujian yang sudah disimpan lewat tombol 'Simpan Pengujian' di header selama sesi ini.",
            "List of test results saved via the 'Save Test' header button during this session.")}</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state['riwayat_list']:
        st.info(tt(
            "Belum ada riwayat pengujian yang disimpan pada sesi ini. Klik tombol 'Simpan Pengujian' "
            "di header (atas halaman manapun) setelah hasil evaluasi muncul.",
            "No test history saved yet in this session. Click the 'Save Test' button in the header "
            "(at the top of any page) once the evaluation result appears."))
    else:
        st.caption(tt(
            f"{len(st.session_state['riwayat_list'])} entri tersimpan — riwayat ini hanya berlaku selama sesi "
            "berjalan (tidak disimpan permanen ke file/Drive).",
            f"{len(st.session_state['riwayat_list'])} entries saved — this history only lasts for the current "
            "session (not permanently saved to a file/Drive)."))
        df_riwayat = pd.DataFrame(st.session_state['riwayat_list'])
        st.dataframe(df_riwayat, use_container_width=True, hide_index=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button(tt("🗑️ Bersihkan Riwayat", "🗑️ Clear History"), key="btn_clear_riwayat"):
            st.session_state['riwayat_list'] = []
            st.rerun()
