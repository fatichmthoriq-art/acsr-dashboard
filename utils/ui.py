import streamlit as st
import os
import io
import base64
import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# TEMA — Navy Undip x PLN Blue x Emas (diselaraskan dengan paket desain baru
# yang dikirim user: "Dashboard ACSR NDT-DT" referensi UNDIP x PLN). Sebelumnya
# skema ini ungu/indigo -- diganti total ke navy+emas+biru PLN sesuai referensi
# baru, TAPI seluruh nama fungsi & signature di file ini TETAP SAMA supaya
# app.py tidak perlu diubah sama sekali (nol risiko merusak fitur yang sudah
# jalan -- bahasa ID/EN, sidebar nav, dsb). Satu skema warna tetap, tidak ada
# toggle gelap/terang.
# ---------------------------------------------------------------------------
NAVY_DEEP = "#001B3D"   # paling gelap -- gradien header/sidebar
NAVY = "#002B5C"        # warna utama brand (dulu PRIMARY ungu)
BLUE_MID = "#00558F"
PLN_BLUE = "#0093DD"    # aksen biru PLN
GOLD = "#F5B921"        # aksen emas Undip/PLN -- garis atas header, ring aktif

P = {"bg": "#EEF2F7", "card": "#FFFFFF", "card_alt": "#F5F8FB", "ink": "#0F1B2D", "ink_soft": "#7A889C",
     "border": "rgba(0,43,92,.10)", "pass_bg": "#EAF6EF", "pass_ink": "#0F5C39", "fail_bg": "#FBEDEC",
     "fail_ink": "#8A2A24", "warn_bg": "#FDF6E7", "warn_ink": "#7A5205"}
GREEN, RED, BLUE, AMBER = "#1E7F4E", "#C8443C", "#0093DD", "#C77E12"
PASS, FAIL, WARN = GREEN, RED, AMBER
# PRIMARY/PRIMARY_DARK dipertahankan (dipakai di seluruh CSS sidebar/tombol/aksen
# "brand" di bawah) -- sekarang menunjuk ke Navy Undip, bukan ungu lagi.
PRIMARY = NAVY
PRIMARY_DARK = NAVY_DEEP
NAVY_GRAD = f"linear-gradient(102deg, {NAVY_DEEP} 0%, {NAVY} 46%, #004E86 100%)"
LEAF_GRAD = f"linear-gradient(135deg, {NAVY_DEEP} 0%, {NAVY} 60%, {BLUE_MID} 100%)"

FONT_SANS = "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'SFMono-Regular', monospace"


def apply_custom_css(compact=False):
    p = P
    sidebar_width = "88px" if compact else "285px"
    # CSS tambahan mode compact untuk daftar nav sidebar baru (v74/v75) --
    # dibangun sebagai string BIASA (bukan f-string bersarang) supaya kurung
    # kurawal di dalamnya tidak perlu di-escape ganda saat disisipkan ke
    # f-string besar di bawah. v74 sempat coba teks label diperkecil+wrap
    # supaya tetap kebaca di 88px -- HASILNYA malah pecah jadi tumpukan huruf
    # per baris ("Rin/gk/asa/n"), tidak rapi (laporan user). v75: nav_button()
    # SEKARANG ganti ke IKON SAJA saat compact=True (lihat app.py, teks label
    # penuh cuma dipakai sbg tooltip help=) -- jadi CSS di sini cukup bikin
    # tombolnya jadi kotak ikon rapi (font besar, satu baris, center),
    # BUKAN lagi menyusutkan+mewrap teks panjang.
    _navlist_compact_css = """
        .st-key-sidebar_nav_list .stButton button {
            font-size: 1.35rem !important; padding: 10px 2px !important;
            text-align: center !important; justify-content: center !important;
            white-space: nowrap !important; line-height: 1 !important;
            min-height: 44px !important;
        }
    """ if compact else ""
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {{ --sidebar-w: {sidebar_width}; }}
        .stApp {{ background: linear-gradient(165deg, #F5F7FA 0%, #EEF2F7 45%, #F6F8FB 100%); background-attachment: fixed; }}
        header[data-testid="stHeader"] {{ background-color: {p['bg']} !important; }}
        div[data-testid="stToolbar"] {{ background-color: transparent !important; }}
        div[data-testid="stDecoration"] {{ background: transparent !important; }}
        html, body, [class*="css"] {{ font-family: {FONT_SANS}; }}
        /* Setelah beberapa kali coba "menengahkan" konten pakai margin:auto /
           flexbox (gagal terus, kemungkinan besar salah tebak selector di
           versi Streamlit ini), sekarang pendekatannya diganti total: HAPUS
           batas lebar (max-width) sepenuhnya. Tanpa cap, komponen otomatis
           mengisi penuh ruang yang tersedia -- responsif alami terhadap
           sidebar dibuka/ditutup, tidak butuh logika "center" sama sekali. */
        .block-container, .stMainBlockContainer {{
            padding-top: 1.1rem; padding-bottom: 2.2rem;
            padding-left: clamp(1rem, 3vw, 2.4rem); padding-right: clamp(1rem, 3vw, 2.4rem);
            max-width: 100% !important;
            width: 100% !important;
        }}

        /* Paksa kolom st.columns() meregang sama tinggi -- supaya card seperti
           .info-tile yang isinya beda panjang teks tetap terlihat seragam,
           tidak berantakan (satu pendek satu panjang). */
        div[data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
        div[data-testid="column"] {{ display: flex !important; flex-direction: column !important; }}
        div[data-testid="column"] > div {{ display: flex; flex-direction: column; flex: 1; }}

        h1, h2, h3, h4, h5 {{ color: {p['ink']}; font-family: {FONT_SANS}; letter-spacing: -.01em; }}
        p, span, label, li {{ color: {p['ink']}; }}

        /* ===== HEADER ===== */
        .header-card {{
            background: {NAVY_GRAD};
            border-top: 4px solid {GOLD};
            padding: 26px clamp(20px, 3vw, 34px) !important;
            border-radius: 16px 16px 0 0;
            margin-bottom: 0;
            box-shadow: 0 8px 24px rgba(0,27,61,0.22);
            text-align: left;
            box-sizing: border-box;
        }}
        .header-card > div {{ margin: 0 !important; }}
        .header-title {{
            color: #FFFFFF; font-size: clamp(20px, 2.4vw, 25px); font-weight: 800;
            line-height: 1.3; letter-spacing: 0.01em; margin: 0 !important;
            text-shadow: 0 1px 3px rgba(0,0,0,0.35);
        }}
        .header-subtitle {{
            color: #FFFFFF; font-size: clamp(13px, 1.4vw, 15px);
            margin: 8px 0 0 0 !important;
            opacity: 0.92; text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}

        /* ===== STEP CARD (1-2-3 dinamis) ===== */
        .step-card {{
            background: linear-gradient(180deg, {p['card_alt']} 0%, {p['card']} 100%);
            border: 1px solid {p['border']};
            border-top: 3px solid {PLN_BLUE};
            border-radius: 16px;
            padding: 20px 20px 18px 20px;
            box-shadow: 0 4px 14px rgba(0,43,92,0.10);
            height: 100%;
            position: relative;
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }}
        .step-card:hover {{ box-shadow: 0 10px 26px rgba(0,43,92,0.16); transform: translateY(-3px); }}
        .step-num {{
            position: absolute; top: -14px; left: 18px;
            background: {NAVY_GRAD}; color: white; font-weight: 800; font-size: 0.8rem;
            font-family: {FONT_MONO};
            width: 28px; height: 28px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 4px 10px rgba(0,43,92,0.45);
            border: 2px solid {p['bg']};
        }}
        .step-card .icon {{ font-size: 1.7rem; margin: 8px 0 8px 0; }}
        .step-card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; color: {NAVY}; font-weight: 700; }}
        .step-card p {{ margin: 0; font-size: 0.84rem; color: {p['ink_soft']}; line-height: 1.5; }}

        /* ===== RESULT CARD — warna EKSPLISIT, konsisten di semua tema ===== */
        .result-box {{
            border-radius: 12px; padding: 15px 17px; margin-top: 9px; border-left: 5px solid;
            box-shadow: 0 2px 8px rgba(0,43,92,0.06);
        }}
        .result-pass {{ background: {p['pass_bg']}; border-color: {PASS}; color: {p['pass_ink']} !important; }}
        .result-fail {{ background: {p['fail_bg']}; border-color: {FAIL}; color: {p['fail_ink']} !important; }}
        .result-warn {{ background: {p['warn_bg']}; border-color: {WARN}; color: {p['warn_ink']} !important; }}
        .result-box, .result-box * {{ color: inherit !important; }}
        .result-box b {{ font-weight: 800; }}
        .result-box .reason {{ font-size: 0.83rem; margin-top: 5px; opacity: 0.92; line-height: 1.5; }}
        .result-divider {{ height: 1px; background: currentColor; opacity: 0.18; margin: 12px 0 10px 0; }}
        .result-extra-label {{
            font-family: {FONT_MONO};
            font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
            opacity: 0.85; margin-bottom: 3px;
        }}

        /* ===== INFO TILE (kartu angka kecil) — dengan aksen atas berwarna ===== */
        .info-tile {{
            background: {p['card']}; border: 1px solid {p['border']}; border-radius: 14px;
            padding: 0.9rem 1.05rem; height: 100%; min-height: 128px;
            display: flex; flex-direction: column;
            border-top: 3px solid {p['ink_soft']};
            box-shadow: 0 3px 10px rgba(0,43,92,0.07);
        }}
        .info-tile .lbl {{
            font-family: {FONT_MONO}; font-size: 0.62rem; color: {p['ink_soft']}; font-weight: 600;
            text-transform: uppercase; letter-spacing: .09em;
        }}
        .info-tile .val {{
            font-family: {FONT_MONO}; font-size: clamp(1.0rem, 1.6vw, 1.3rem); font-weight: 600; color: {NAVY};
            margin-top: 6px; word-break: break-word; line-height: 1.25;
        }}
        .info-tile .expl {{ font-size: 0.78rem; color: {p['ink_soft']}; margin-top: auto; padding-top: 6px; line-height: 1.5; }}
        /* Varian warna aksen -- supaya kartu tidak seragam/monoton, tiap kategori beda warna */
        .info-tile.accent-blue {{ border-top-color: {PLN_BLUE}; }}
        .info-tile.accent-green {{ border-top-color: {PASS}; }}
        .info-tile.accent-purple {{ border-top-color: {BLUE_MID}; }}
        .info-tile.accent-orange {{ border-top-color: {WARN}; }}
        .info-tile.accent-teal {{ border-top-color: #0D7E8C; }}
        .info-tile.accent-pink {{ border-top-color: #B23A6B; }}
        .info-tile.accent-gold {{ border-top-color: {GOLD}; }}

        /* ===== LABEL MINI KARTU DONUT (mikrografi, v74) =====
           Dipakai di kartu "PROBABILITAS KOROSI PERMUKAAN" tab Mikrografi
           (donut + teks jadi satu kartu putih, sesuai referensi tampilan
           yang dikirim user) -- style-nya sengaja disamakan dgn .info-tile
           .lbl (mono, kecil, huruf besar) supaya konsisten satu bahasa
           visual dengan kartu info lain. */
        .donut-card-label {{
            font-family: {FONT_MONO}; font-size: 0.62rem; color: {p['ink_soft']}; font-weight: 600;
            text-transform: uppercase; letter-spacing: .09em; margin-bottom: 6px;
        }}

        /* ===== VERDICT POLOS + DAFTAR SUB-SKOR (kartu donut Ringkasan
           Beranda, v77) =====
           plain_verdict(): teks judul besar+bold+warna rata tengah + deskripsi
           abu-abu di bawahnya, TANPA kotak/border berwarna (beda dari
           result_box() yang dipakai di tempat lain aplikasi) -- sesuai
           referensi tampilan yang dikirim user untuk kartu donut. */
        .plain-verdict {{
            text-align: center; font-size: 1.15rem; font-weight: 800; margin-top: 10px;
        }}
        .plain-verdict-reason {{
            text-align: center; font-size: 0.85rem; color: {p['ink_soft']};
            line-height: 1.55; margin: 6px auto 0 auto; max-width: 320px;
        }}
        /* mini_score_bar(): daftar sub-skor per kategori (NDT/Mikrografi/Tarik
           Al/Tarik Baja/Torsi/Lilit) dgn bar tipis, dipasang di bawah divider
           kartu donut -- label lebar tetap supaya bar & angka tetap sejajar
           rapi walau panjang label beda-beda. */
        .mini-score-row {{
            display: flex; align-items: center; gap: 10px; margin-bottom: 11px;
        }}
        .mini-score-row:last-child {{ margin-bottom: 0; }}
        .mini-score-label {{
            flex: 0 0 108px; font-size: 0.8rem; color: {NAVY}; font-weight: 600;
        }}
        .mini-score-track {{
            flex: 1; height: 7px; background: #E7EDF4; border-radius: 999px; overflow: hidden;
        }}
        .mini-score-fill {{ display: block; height: 100%; border-radius: 999px; }}
        .mini-score-value {{
            flex: 0 0 26px; text-align: right; font-family: {FONT_MONO}; font-weight: 700;
            color: {NAVY}; font-size: 0.82rem;
        }}

        /* ===== LEGENDA WARNA (kartu "Skor per parameter pengujian", v77) =====
           3 titik warna + label kecil di pojok kanan-atas kartu bar chart,
           sesuai referensi tampilan yang dikirim user. */
        .score-legend {{
            display: flex; flex-direction: column; gap: 4px; align-items: flex-start; padding-top: 3px;
        }}
        .score-legend-item {{
            display: flex; align-items: center; gap: 6px; font-size: 0.72rem;
            color: {p['ink_soft']}; font-weight: 600; white-space: nowrap;
        }}
        .score-legend-item .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}

        /* ===== KAMUS ISTILAH — expander interaktif ===== */
        div[data-testid="stExpander"] {{
            border: 1px solid {p['border']} !important; border-radius: 12px !important;
            background: {p['card']} !important; margin-bottom: 8px;
        }}
        div[data-testid="stExpander"] summary {{ font-weight: 700 !important; color: {NAVY} !important; }}

        /* ===== SIDEBAR — nav lebih profesional ===== */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(178deg, {NAVY_DEEP} 0%, {NAVY} 62%, #013C6E 100%);
            min-width: var(--sidebar-w) !important;
            max-width: var(--sidebar-w) !important;
            transition: min-width 0.25s ease, max-width 0.25s ease;
        }}
        section[data-testid="stSidebar"] .block-container {{ padding: 1.1rem {"0.4rem" if compact else "1rem"}; }}
        .sidebar-collapse-btn {{
            position: relative; width: 34px; height: 34px; border-radius: 50%;
            background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 10px auto; cursor: pointer;
        }}
        /* Tombol bendera bahasa -- bendera itu sendiri digambar via st.markdown
           (<img>, lihat language_toggle() di bawah file ini) supaya PASTI
           tampil di versi Streamlit manapun -- percobaan sebelumnya memasang
           bendera sebagai CSS background-image tombol via class ".st-key-..."
           ternyata TIDAK match di Streamlit versi kamu (sama seperti kasus
           ikon nav sidebar yang sempat merah), jadi bendera tidak kelihatan
           sama sekali. Sekarang gambar bendera + cincin "aktif" SUDAH PASTI
           benar (ditentukan langsung di Python, bukan CSS kind="primary").
           CSS di bawah ini HANYA polesan kosmetik opsional untuk tombol klik
           kecil di bawah bendera -- kalau tidak match pun tidak masalah,
           bendera & fungsinya tetap jalan normal. */
        /* Selector rangkap: class .st-key-... (tidak selalu match di versi
           Streamlit user) DITAMBAH selector berbasis atribut title="..."
           (dari parameter help= di st.button(), SELALU jadi atribut HTML
           native apapun versi Streamlit-nya -- jaring pengaman utama).
           Semuanya di-scope ke dalam sidebar supaya spesifisitasnya PASTI
           menang dibanding aturan tombol global ".stButton button[kind=...]"
           di bawah (bahasa toggle ini sekarang dirender di dalam sidebar). */
        /* .stButton.stButton (class diulang) SENGAJA -- trik menaikkan
           spesifisitas CSS supaya PASTI menang melawan aturan tombol sidebar
           kind="secondary" di bawah (yang juga men-scope ke sidebar & py
           sama-sama pakai kombinasi elemen+atribut+class serupa; tanpa trik
           ini keduanya sama spesifik dan urutan penulisan CSS yang menang,
           terlalu rapuh untuk diandalkan). */
        /* .st-key-lang_toggle_id/en (dari st.container(key=...) yang membungkus
           tombol klik bendera, lihat language_toggle() di bawah) adalah selector
           UTAMA sekarang -- terbukti reliable di Streamlit >=1.34 (dipakai juga
           di .st-key-header_nav_tabs). Selector button[title=...] & .st-key-
           lang_id_btn/en_btn (langsung ke tombolnya, BUKAN container-nya) tetap
           dipasang sebagai jaring pengaman tambahan, tidak menggantikan. */
        .st-key-lang_toggle_id button, .st-key-lang_toggle_en button,
        .st-key-lang_id_btn button, .st-key-lang_en_btn button,
        section[data-testid="stSidebar"] .stButton.stButton button[title="Bahasa Indonesia"],
        section[data-testid="stSidebar"] .stButton.stButton button[title="English"] {{
            min-height: 20px !important; height: 20px !important; padding: 0 !important;
            margin: 2px auto 0 auto !important; background: transparent !important;
            border: none !important; box-shadow: none !important; font-size: 0 !important;
        }}
        .st-key-lang_toggle_id button:hover, .st-key-lang_toggle_en button:hover,
        .st-key-lang_id_btn button:hover, .st-key-lang_en_btn button:hover,
        section[data-testid="stSidebar"] .stButton.stButton button[title="Bahasa Indonesia"]:hover,
        section[data-testid="stSidebar"] .stButton.stButton button[title="English"]:hover {{
            background: rgba(245,185,33,0.14) !important; border-radius: 6px !important;
        }}
        /* Container pembungkus sendiri -- hilangkan margin/gap bawaan supaya
           tidak menambah ruang kosong di bawah bendera. */
        .st-key-lang_toggle_id, .st-key-lang_toggle_en {{ margin-top: 0 !important; }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{ color: #E6EDF5 !important; }}
        /* Fix spesifik: label selectbox ("Pilih Tipe Konduktor ACSR:") kadang
           teksnya dibungkus <p> di dalam label yang tidak ikut ke-cover aturan
           di atas -- paksa terang di sini, TAPI JANGAN kena bagian dropdown
           putihnya (itu diatur terpisah di bawah supaya tetap teks gelap). */
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p,
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label span {{
            color: #E6EDF5 !important;
        }}
        section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.13); margin: 0.9rem 0; }}
        section[data-testid="stSidebar"] .stButton>button {{
            background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
            font-size: 0.83rem; color: #E6EDF5 !important; border-radius: 9px;
        }}
        section[data-testid="stSidebar"] .stButton>button:hover {{
            background: rgba(255,255,255,0.16); border-color: rgba(255,255,255,0.32);
        }}

        div[role="radiogroup"] {{ gap: 0.2rem !important; }}
        div[role="radiogroup"] label {{
            font-size: 0.9rem !important; padding: 9px 10px !important; border-radius: 9px;
            transition: background 0.15s ease; width: 100%;
        }}
        div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.07); }}

        /* ===== NAV PILL (tombol navigasi sidebar kustom) -- DIHAPUS (v73) =====
           Blok ini dulu menstyle nav_button()/nav_section_label() (daftar
           tombol navigasi vertikal di sidebar), TAPI sejak header_tabs()
           menggantikan navigasi sidebar (lihat app.py, "Navigasi sekarang
           jadi baris tab di bawah header card"), nav_button() tidak pernah
           dipanggil lagi -- fungsinya masih ada di file ini tapi jadi dead
           code. Masalahnya, selector generik "section[stSidebar] .stButton
           button[kind=secondary/primary]" di blok lama itu TETAP menyasar
           SEMUA tombol biasa lain di sidebar (toggle bahasa & tombol ciutkan
           sidebar), dan spesifisitasnya lebih tinggi daripada selector
           ".st-key-lang_toggle_id/en" -- akibatnya tombol bendera bahasa
           dipaksa min-height:68px + padding 14px + flex-column ala kartu
           besar, persis bug "tumpang tindih"/kotak besar pecah yang
           dilaporkan user di sidebar saat diciutkan. Blok CSS-nya dihapus
           total di sini supaya tidak ada lagi konflik; kalau nav_button()
           dipakai lagi nanti, style-nya harus dibungkus st.container(key=...)
           dan di-scope ke ".st-key-..." sendiri, bukan selector generik
           kind="secondary"/"primary" yang berlaku sidebar-lebar begini. */

        .nav-section-label {{
            font-family: {FONT_MONO};
            font-size: 0.65rem; font-weight: 600; color: #8FA3C2; text-transform: uppercase;
            letter-spacing: 0.12em; margin: 14px 0 6px 4px;
        }}

        /* ===== TOMBOL AREA UTAMA (di luar sidebar) -- termasuk baris tab
           navigasi header (header_tabs), tombol reload/batch/download, dsb.
           Satu gaya konsisten navy+emas supaya seluruh dashboard terasa satu
           kesatuan yang profesional. SENGAJA TIDAK di-scope ke ".main" atau
           semacamnya (nama class wrapper utama Streamlit berubah-ubah antar
           versi) -- sebagai gantinya, aturan sidebar di atas (yang diawali
           `section[data-testid="stSidebar"] ...`) otomatis MENANG di dalam
           sidebar karena spesifisitas selector-nya lebih tinggi, terlepas
           urutan penulisan CSS. Dual selector kind=/data-testid= tetap
           dipasang sebagai jaring pengaman ganda (versi Streamlit user
           terbukti tidak selalu match salah satu pola saja). */
        /* PENTING: warna teks harus dipasang juga ke `... *` (SEMUA elemen
           turunan tombol), bukan cuma ke elemen <button> itu sendiri --
           Streamlit membungkus label tombol dalam <p>/<div> di DALAM
           <button>, dan blanket rule "p, span, label, li {{ color: ... }}"
           di atas (dipakai supaya teks halaman lain kontras di background
           terang) otomatis menimpa warna itu kalau tidak ikut ditarget di
           sini -- itu sebabnya tab navigasi AKTIF sempat teksnya nyaris tak
           terbaca (putih yang dimaksud ketiban warna gelap dari rule blanket
           tsb, karena spesifisitasnya kena elemen <p> secara langsung). */
        .stButton button[kind="secondary"], .stButton button[kind="secondary"] *,
        .stButton button[data-testid="stBaseButton-secondary"], .stButton button[data-testid="stBaseButton-secondary"] * {{
            color: #0F2A4A !important;
        }}
        .stButton button[kind="secondary"],
        .stButton button[data-testid="stBaseButton-secondary"] {{
            background: #FFFFFF !important; border: 1.5px solid rgba(0,43,92,.16) !important;
            font-weight: 600 !important; border-radius: 10px !important;
            box-shadow: 0 1px 3px rgba(0,43,92,.06) !important; padding: 0.55rem 1rem !important;
        }}
        .stButton button[kind="secondary"]:hover,
        .stButton button[data-testid="stBaseButton-secondary"]:hover {{
            background: #F5F8FC !important; border-color: {PLN_BLUE} !important;
        }}
        .stButton button[kind="secondary"]:hover *,
        .stButton button[data-testid="stBaseButton-secondary"]:hover * {{
            color: {NAVY} !important;
        }}
        .stButton button[kind="primary"], .stButton button[kind="primary"] *,
        .stButton button[data-testid="stBaseButton-primary"], .stButton button[data-testid="stBaseButton-primary"] * {{
            color: #FFFFFF !important;
        }}
        .stButton button[kind="primary"],
        .stButton button[data-testid="stBaseButton-primary"] {{
            background: {NAVY} !important; border: none !important;
            font-weight: 700 !important; border-radius: 10px !important;
            border-bottom: 3px solid {GOLD} !important; padding: 0.55rem 1rem !important;
            box-shadow: 0 4px 14px rgba(0,43,92,.22) !important;
        }}
        .stButton button[kind="primary"]:hover,
        .stButton button[data-testid="stBaseButton-primary"]:hover {{
            background: {PRIMARY_DARK} !important;
        }}
        div[data-testid="stDownloadButton"] button, div[data-testid="stDownloadButton"] button * {{
            color: {NAVY} !important;
        }}
        div[data-testid="stDownloadButton"] button {{
            background: {GOLD} !important; border: none !important;
            font-weight: 700 !important; border-radius: 10px !important; padding: 0.7rem 1.2rem !important;
            box-shadow: 0 4px 14px rgba(245,185,33,.30) !important;
        }}
        div[data-testid="stDownloadButton"] button:hover {{ background: #E5A80F !important; }}

        /* ===== KARTU CTA CETAK PDF (pdf_cta_card) =====
           Dibungkus lewat st.container(key="pdf_cta_card") di app.py, sama
           persis polanya dengan ".st-key-header_nav_tabs" di bawah -- supaya
           CSS ini PASTI menyasar kartunya (bukan mengandalkan selector atribut
           yang riwayatnya tidak selalu match di semua versi Streamlit). Isinya
           gradient navy + judul putih tebal + deskripsi abu-abu terang, sesuai
           referensi tampilan "Cetak bukti validasi resmi" yang dikirim user.
           Tombol download di dalamnya tetap otomatis emas dari aturan
           "div[data-testid=stDownloadButton] button" di atas. */
        .st-key-pdf_cta_card {{
            background: {NAVY_GRAD};
            border-radius: 16px;
            border-top: 4px solid {GOLD};
            padding: 26px clamp(20px, 3vw, 32px) 24px clamp(20px, 3vw, 32px);
            box-shadow: 0 8px 24px rgba(0,27,61,0.22);
            margin-top: 4px;
        }}
        .st-key-pdf_cta_card .pdf-cta-title {{
            color: #FFFFFF !important; font-weight: 800; font-size: 1.08rem;
            margin: 0 0 9px 0; text-shadow: 0 1px 3px rgba(0,0,0,0.25);
        }}
        .st-key-pdf_cta_card .pdf-cta-desc {{
            color: rgba(255,255,255,0.86) !important; font-size: 0.87rem;
            line-height: 1.65; margin: 0 0 18px 0; max-width: 640px;
        }}
        .st-key-pdf_cta_card div[data-testid="stDownloadButton"] button {{
            padding: 0.75rem 1.4rem !important; font-size: 0.92rem !important;
        }}

        /* ===== HEADER UTAMA BARU (v74) -- main_header_card =====
           Menggantikan .header-card + header_tabs() lama: sekarang SATU kartu
           gradient navy yang memuat brand row (logo chip + "UNDIP x PLN" +
           toggle bahasa), badge kode spesimen + baris info (Haspel/panjang/
           tanggal/tim), judul dashboard KONSTAN (tidak lagi berubah per
           halaman), lalu baris interaktif: selectbox tipe ACSR + tombol
           "Data Pengujian"/"Simpan Pengujian". Dibungkus st.container(key=
           "main_header_card") (bukan HTML statis) supaya widget Streamlit
           (selectbox/button) tetap hidup DI DALAM kartu -- pola sama seperti
           pdf_cta_card. Navigasi halaman (dulu baris tab header_tabs di sini)
           SUDAH PINDAH ke sidebar (lihat .st-key-sidebar_nav_list di bawah),
           sesuai permintaan user v74 supaya lebih rapi/tidak dobel dengan
           header. */
        .st-key-main_header_card {{
            background: {NAVY_GRAD};
            border-top: 4px solid {GOLD};
            border-radius: 16px;
            padding: 22px clamp(18px, 3vw, 30px) 20px clamp(18px, 3vw, 30px);
            box-shadow: 0 8px 24px rgba(0,27,61,0.22);
            margin-bottom: 20px;
        }}
        .st-key-main_header_card .header-brand-row {{
            display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
        }}
        .st-key-main_header_card .header-logo-chip {{
            width: 15px; height: 15px; border-radius: 5px; display: inline-block;
        }}
        .st-key-main_header_card .header-brand-text {{
            font-family: {FONT_MONO}; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
            color: rgba(255,255,255,0.75); text-transform: uppercase; margin-left: 2px;
        }}
        .st-key-main_header_card .header-spec-row {{
            display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;
        }}
        .st-key-main_header_card .header-spec-badge {{
            background: {GOLD}; color: {NAVY}; font-family: {FONT_MONO}; font-weight: 800;
            font-size: 0.74rem; letter-spacing: 0.04em; padding: 4px 12px; border-radius: 999px;
        }}
        .st-key-main_header_card .header-spec-sub {{
            color: rgba(255,255,255,0.78); font-size: 0.8rem;
        }}
        .st-key-main_header_card .header-main-title {{
            color: #FFFFFF; font-size: clamp(19px, 2.3vw, 24px); font-weight: 800;
            line-height: 1.3; margin: 2px 0 16px 0; text-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}
        /* Selectbox jadi pil putih (kontras dgn navy). Tombol "Data Pengujian"
           jadi chip translucent (sekunder), "Simpan Pengujian" jadi emas
           (primer) -- override aturan tombol global kind=secondary/primary
           (yang defaultnya untuk background TERANG, tidak cocok kalau
           dipasang langsung di atas kartu navy ini). */
        .st-key-main_header_card div[data-testid="stSelectbox"] > div {{
            background: #FFFFFF !important; border-radius: 10px !important;
        }}
        .st-key-main_header_card .stButton button[kind="secondary"],
        .st-key-main_header_card .stButton button[data-testid="stBaseButton-secondary"] {{
            background: rgba(255,255,255,0.12) !important; border: 1px solid rgba(255,255,255,0.30) !important;
            box-shadow: none !important;
        }}
        .st-key-main_header_card .stButton button[kind="secondary"] *,
        .st-key-main_header_card .stButton button[data-testid="stBaseButton-secondary"] * {{
            color: #FFFFFF !important;
        }}
        .st-key-main_header_card .stButton button[kind="secondary"]:hover,
        .st-key-main_header_card .stButton button[data-testid="stBaseButton-secondary"]:hover {{
            background: rgba(255,255,255,0.22) !important;
        }}
        .st-key-main_header_card .stButton button[kind="primary"],
        .st-key-main_header_card .stButton button[data-testid="stBaseButton-primary"] {{
            background: {GOLD} !important; border: none !important; box-shadow: 0 4px 14px rgba(245,185,33,.30) !important;
        }}
        .st-key-main_header_card .stButton button[kind="primary"] *,
        .st-key-main_header_card .stButton button[data-testid="stBaseButton-primary"] * {{
            color: {NAVY} !important;
        }}
        .st-key-main_header_card .stButton button[kind="primary"]:hover,
        .st-key-main_header_card .stButton button[data-testid="stBaseButton-primary"]:hover {{
            background: #E5A80F !important;
        }}
        /* BUG v75 -- toggle bahasa "kotak besar" lagi, kali ini di HEADER:
           language_toggle() dipindah ke DALAM .st-key-main_header_card (kolom
           kanan brand row) di v74, sehingga tombol klik benderanya (di dalam
           .st-key-lang_toggle_id/en) kena DUA aturan sekaligus -- aturan kecil
           yang dimaksud (".st-key-lang_toggle_id button", spesifisitas 0-1-1)
           DAN aturan sekunder kartu header persis di atas ini
           (".st-key-main_header_card .stButton button[kind=secondary]",
           spesifisitas 0-3-1, LEBIH TINGGI) -- yang menang otomatis aturan
           kartu header, bikin tombol bendera balik jadi kotak besar
           translucent (persis bug NAV PILL v73, tapi lokasinya beda). FIX:
           selector di bawah ini SENGAJA dibuat lebih spesifik lagi (4 level:
           main_header_card + lang_toggle_id/en + stButton + [kind=secondary])
           supaya PASTI menang atas aturan kartu header di atas, mengembalikan
           tombol bendera ke ukuran kecil transparan aslinya. */
        .st-key-main_header_card .st-key-lang_toggle_id .stButton button[kind="secondary"],
        .st-key-main_header_card .st-key-lang_toggle_en .stButton button[kind="secondary"],
        .st-key-main_header_card .st-key-lang_toggle_id button,
        .st-key-main_header_card .st-key-lang_toggle_en button {{
            min-height: 20px !important; height: 20px !important; padding: 0 !important;
            margin: 2px auto 0 auto !important; background: transparent !important;
            border: none !important; box-shadow: none !important; font-size: 0 !important;
        }}
        .st-key-main_header_card .st-key-lang_toggle_id button:hover,
        .st-key-main_header_card .st-key-lang_toggle_en button:hover {{
            background: rgba(245,185,33,0.18) !important; border-radius: 6px !important;
        }}
        /* Kolom toggle bahasa disejajarkan presisi dgn brand row di sebelah
           kiri (rata kanan, rata tengah vertikal) -- sebelumnya menempel ke
           kiri kolomnya sendiri sehingga terlihat "mengambang" tidak sejajar.
           v77: selector diganti dari :has(.st-key-lang_toggle_id) ke
           :has(.st-key-lang_toggle_row) -- alasan sama dgn catatan bug v77 di
           bawah (:has(.st-key-lang_toggle_id) overmatch krn ada 2 level
           kolom bersarang: kolom LUAR col_lang, DAN kolom DALAM col_id/col_en
           yang masing2 langsung membungkus .st-key-lang_toggle_id/en --
           keduanya sama2 "punya keturunan" .st-key-lang_toggle_id sehingga
           dulu SEMUA level kolom ikut ke-flex-column, bukan cuma col_lang
           yang dimaksud). .st-key-lang_toggle_row cuma ada SATU level di atas
           (langsung di dalam col_lang), jadi :has() ini sekarang cuma match
           col_lang tunggal, tidak overmatch ke kolom bendera individual. */
        .st-key-main_header_card div[data-testid="column"]:has(.st-key-lang_toggle_row) {{
            display: flex !important; flex-direction: column !important;
            align-items: flex-end !important; justify-content: center !important;
        }}
        /* BUG v76 -- 2 bendera terlalu berjauhan: language_toggle() sendiri
           membuat st.columns(2) INTERNAL (satu kolom per bendera), dan kolom
           itu otomatis melebar mengisi PENUH lebar kolom induknya (yg di
           app.py cukup lebar, st.columns([3,1]) dari total lebar header yang
           sudah tanpa batas max-width) -- tiap bendera jadi ke-CENTER di
           tengah kolomnya masing2 yang lebar, hasilnya kelihatan berjauhan
           padahal maksudnya cuma toggle kecil berdampingan. FIX v76 (SALAH,
           diperbaiki lagi di v77 -- lihat catatan di bawah): baris horizontal
           INTERNAL itu di-target via :has(.st-key-lang_toggle_id).

           BUG BARU ke-2 yang BARU KETAHUAN di v77 dari screenshot user: fix
           :has() di atas TERNYATA juga menimpa baris horizontal LUAR (brand
           row "UNDIP x PLN" + kolom toggle bahasa, dari st.columns([3,1]) di
           app.py) -- karena :has() mencocokkan keturunan di KEDALAMAN
           BERAPAPUN, bukan cuma anak langsung, dan .st-key-lang_toggle_id
           tetap jadi "keturunan" baris luar itu (lewat col_lang -> baris
           dalam -> container-nya). Akibatnya baris luar IKUT dipaksa
           shrink-to-fit + rata-kanan + margin-left:auto, sehingga teks
           "UNDIP x PLN" ikut "tersedot" ke kanan menempel bendera alih-alih
           tetap di kiri kartu header seperti seharusnya.
           FIX v77: language_toggle() sekarang membungkus st.columns(2)-nya
           sendiri di dalam st.container(key="lang_toggle_row") (lihat
           utils/ui.py) -- jadi baris horizontal internalnya bisa disasar
           LANGSUNG lewat ".st-key-lang_toggle_row div[stHorizontalBlock]"
           TANPA :has() sama sekali, dijamin TIDAK PERNAH match baris luar
           yang mana pun karena scoping-nya sekarang eksplisit lewat class
           container, bukan deteksi keturunan. PELAJARAN BARU: :has() itu
           mencocokkan keturunan di SEMUA kedalaman -- kalau elemen target
           bisa muncul lagi di dalam elemen bersarang (nested columns di
           dalam columns), :has() gampang overmatch ke leluhur yang lebih
           luar dari yang dimaksud. Solusi paling aman: bungkus elemen yang
           mau ditarget dgn st.container(key=...) sendiri dan sasar lewat
           class ".st-key-<key>" langsung, HINDARI :has() kalau strukturnya
           bersarang berlapis seperti ini. */
        .st-key-main_header_card .st-key-lang_toggle_row div[data-testid="stHorizontalBlock"] {{
            display: flex !important; flex-wrap: nowrap !important;
            justify-content: flex-end !important; align-items: center !important;
            gap: 10px !important; width: auto !important; margin-left: auto !important;
        }}
        .st-key-main_header_card .st-key-lang_toggle_row div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
            flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
        }}

        /* ===== NAV SIDEBAR BARU (v74) -- sidebar_nav_list =====
           Menggantikan header_tabs() sebagai navigasi utama (dipindah ke
           sidebar, permintaan user). Dibungkus st.container(key=
           "sidebar_nav_list") supaya CSS ini TERISOLASI hanya ke tombol nav
           di dalamnya -- TIDAK menyasar tombol sidebar lain (toggle ciutkan)
           seperti bug lama blok "NAV PILL" generik yang sudah dihapus di v73
           (lihat catatan panjang soal itu tepat di atas .nav-section-label).
           TANPA ikon (permintaan user sebelumnya: tampilan profesional,
           tanpa ikon di label nav). */
        .st-key-sidebar_nav_list {{ margin: 6px 0 4px 0; }}
        .st-key-sidebar_nav_list .stButton {{ margin-bottom: 6px; }}
        .st-key-sidebar_nav_list .stButton button[kind="secondary"],
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-secondary"],
        .st-key-sidebar_nav_list .stButton button[kind="secondary"] *,
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-secondary"] * {{
            color: #E6EDF5 !important;
        }}
        .st-key-sidebar_nav_list .stButton button[kind="secondary"],
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-secondary"] {{
            background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.16) !important;
            font-weight: 600 !important; font-size: 0.83rem !important;
            padding: 11px 12px !important; border-radius: 10px !important;
            box-shadow: none !important; text-align: left !important; justify-content: flex-start !important;
        }}
        .st-key-sidebar_nav_list .stButton button[kind="secondary"]:hover,
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-secondary"]:hover {{
            background: rgba(255,255,255,0.16) !important; border-color: rgba(255,255,255,0.32) !important;
        }}
        .st-key-sidebar_nav_list .stButton button[kind="primary"],
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-primary"],
        .st-key-sidebar_nav_list .stButton button[kind="primary"] *,
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-primary"] * {{
            color: {NAVY} !important;
        }}
        .st-key-sidebar_nav_list .stButton button[kind="primary"],
        .st-key-sidebar_nav_list .stButton button[data-testid="stBaseButton-primary"] {{
            background: #FFFFFF !important; border: none !important; font-weight: 700 !important;
            padding: 11px 12px !important; border-radius: 10px !important;
            border-left: 3px solid {GOLD} !important;
            box-shadow: 0 4px 12px rgba(0,27,61,0.28) !important;
            text-align: left !important; justify-content: flex-start !important;
        }}
        {_navlist_compact_css}

        /* ===== BARIS TAB NAVIGASI HEADER (header_tabs) -- SUDAH TIDAK DIPAKAI
           sejak v74 (navigasi pindah ke sidebar, lihat sidebar_nav_list di
           atas) -- fungsi header_tabs() & CSS di bawah ini DIBIARKAN ada
           (dead code, tidak dipanggil dari app.py) supaya kalau suatu saat
           navigasi ingin dikembalikan ke header, tinggal panggil lagi tanpa
           perlu menulis ulang. =====
           Di-scope ke ".st-key-header_nav_tabs" (dari st.container(key=...)
           di header_tabs()) supaya gaya "folder tab" ini HANYA berlaku di
           baris navigasi header, TIDAK ikut mengubah tombol lain di halaman
           utama (yang tetap pakai gaya kind=primary/secondary global di
           atas). Spesifisitas selector lebih tinggi (ada kelas tambahan)
           otomatis MENANG atas aturan global itu, sama seperti pola yang
           sudah terbukti jalan di override tombol sidebar. */
        .st-key-header_nav_tabs {{
            background: {NAVY_GRAD};
            margin-top: -14px;               /* menutup celah bawaan Streamlit di bawah header-card */
            margin-bottom: 22px;
            padding: 0 clamp(14px, 2.4vw, 26px) 8px clamp(14px, 2.4vw, 26px);
            border-radius: 0 0 14px 14px;
            box-shadow: 0 8px 20px rgba(0,27,61,0.18);
        }}
        .st-key-header_nav_tabs .stButton {{ margin-bottom: 0 !important; }}
        .st-key-header_nav_tabs .stButton button[kind="secondary"],
        .st-key-header_nav_tabs .stButton button[data-testid="stBaseButton-secondary"] {{
            background: rgba(255,255,255,0.09) !important;
            border: none !important;
            border-radius: 10px 10px 0 0 !important;
            box-shadow: none !important;
            padding: 0.6rem 1rem !important;
            font-weight: 600 !important;
        }}
        .st-key-header_nav_tabs .stButton button[kind="secondary"] *,
        .st-key-header_nav_tabs .stButton button[data-testid="stBaseButton-secondary"] * {{
            color: rgba(255,255,255,0.68) !important;
        }}
        .st-key-header_nav_tabs .stButton button[kind="secondary"]:hover,
        .st-key-header_nav_tabs .stButton button[data-testid="stBaseButton-secondary"]:hover {{
            background: rgba(255,255,255,0.17) !important;
        }}
        .st-key-header_nav_tabs .stButton button[kind="secondary"]:hover *,
        .st-key-header_nav_tabs .stButton button[data-testid="stBaseButton-secondary"]:hover * {{
            color: #FFFFFF !important;
        }}
        .st-key-header_nav_tabs .stButton button[kind="primary"],
        .st-key-header_nav_tabs .stButton button[data-testid="stBaseButton-primary"] {{
            background: #FFFFFF !important;
            border: none !important;
            border-bottom: 3px solid {GOLD} !important;
            border-radius: 10px 10px 0 0 !important;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.08) !important;
            padding: 0.6rem 1rem !important;
            font-weight: 700 !important;
        }}
        .st-key-header_nav_tabs .stButton button[kind="primary"] *,
        .st-key-header_nav_tabs .stButton button[data-testid="stBaseButton-primary"] * {{
            color: {NAVY} !important;
        }}

        /* ===== JUDUL HALAMAN (theme-aware, tidak hardcode) ===== */
        .page-title-block {{ text-align: center; margin-bottom: 22px; }}
        .page-title-block h1 {{
            font-size: clamp(1.4rem, 3vw, 2.1rem); font-weight: 800; margin-bottom: 4px; color: {NAVY};
        }}
        .page-title-block .page-subtitle {{ font-size: 0.95rem; color: {p['ink_soft']}; margin: 0; }}

        /* FIX: dropdown/selectbox — teks & background KONTRAS, terlepas dari tema sidebar */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important; border-radius: 9px !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] * {{ color: {p['ink']} !important; }}
        div[data-baseweb="popover"] * {{ color: {p['ink']} !important; }}
        div[data-baseweb="popover"] ul {{ background-color: #FFFFFF !important; }}

        div[data-testid="stMetric"] {{
            background: {p['card']}; border: 1px solid {p['border']}; border-radius: 12px;
            padding: 0.75rem 0.95rem;
        }}
        div[data-testid="stMetric"] label {{ color: {p['ink_soft']} !important; }}
        div[data-testid="stMetric"] div {{ color: {p['ink']} !important; }}

        /* ===== FILE UPLOADER -- restyle jadi kartu rounded + drop-zone rapi
           (Streamlit generate teks internalnya sendiri, jadi tidak 100% bisa
           diganti persis, tapi tampilan luarnya kita bikin senada). ===== */
        div[data-testid="stFileUploader"] {{
            background: {p['card']}; border: 1px solid {p['border']}; border-radius: 16px;
            padding: 14px; box-shadow: 0 3px 12px rgba(0,43,92,0.06);
        }}
        div[data-testid="stFileUploader"] > label {{
            font-weight: 700 !important; font-size: 0.95rem !important; color: {p['ink']} !important;
            margin-bottom: 8px !important;
        }}
        div[data-testid="stFileUploader"] section {{
            background: #F7FAFD; border: 2px dashed #BBD3E8 !important; border-radius: 14px !important;
            padding: 28px 16px !important;
        }}
        div[data-testid="stFileUploader"] section > div {{
            display: flex !important; flex-direction: column !important; align-items: center !important;
        }}
        div[data-testid="stFileUploader"] section svg {{
            width: 32px !important; height: 32px !important; color: {BLUE_MID} !important; fill: {BLUE_MID} !important;
        }}
        div[data-testid="stFileUploader"] section small {{ color: {p['ink_soft']} !important; }}
        div[data-testid="stFileUploader"] section button {{
            background: #FFFFFF !important; border: 1.5px solid {BLUE_MID} !important; color: {BLUE_MID} !important;
            border-radius: 999px !important; font-weight: 700 !important; padding: 6px 20px !important;
            margin-top: 10px !important;
        }}
        div[data-testid="stFileUploader"] section button:hover {{
            background: {BLUE_MID} !important; color: #FFFFFF !important;
        }}

        /* ===== BINGKAI CITRA (preview gambar X-Ray/Mikrografi) ===== */
        .img-frame {{
            background: linear-gradient(180deg, #F0F5FA 0%, #FBFCFE 100%);
            border: 2px solid {PLN_BLUE}; border-radius: 14px;
            padding: 14px; box-shadow: 0 4px 14px rgba(0,147,221,0.15); margin-top: 8px;
            display: flex; flex-direction: column;
        }}
        .img-frame .img-caption {{
            font-family: {FONT_MONO};
            font-size: 0.68rem; font-weight: 600; color: {PLN_BLUE}; text-transform: uppercase;
            letter-spacing: 0.08em; margin-bottom: 10px; display: flex; align-items: center; gap: 5px;
            flex-shrink: 0;
        }}
        /* Tinggi bingkai citra v74: dulu DIKUNCI TETAP 420px apapun ukuran
           gambarnya -- keluhan user, citra jadi terlihat "berlebihan"/terlalu
           besar dipaksakan mengisi kotak setinggi itu. Sekarang pakai
           min/max-height (bukan height tetap): kotak menyesuaikan ke ukuran
           kontennya (lewat flex:1 dari kolom yang di-stretch sama tinggi
           dengan panel sebelahnya), TIDAK PERNAH melebihi 380px, dan tidak
           pernah lebih pendek dari 200px supaya kartu tetap konsisten
           sebelum citra diunggah. Gambar sendiri tetap object-fit: contain
           (jaga rasio asli, tidak pernah stretch/distorsi). */
        .img-frame .img-body {{
            flex: 1; display: flex; align-items: center; justify-content: center;
            min-height: 200px; max-height: 380px; background: rgba(0,43,92,0.03); border-radius: 10px;
            overflow: hidden;
        }}
        .img-frame img {{
            border-radius: 8px; max-width: 100%; max-height: 100%; width: auto; height: auto;
            display: block; object-fit: contain;
            border: 1px solid rgba(0,43,92,0.15);
            box-shadow: 0 2px 8px rgba(0,43,92,0.08);
        }}
        /* Kotak placeholder (belum ada citra diunggah) -- garis putus-putus +
           motif garis diagonal halus + label pil di tengah, supaya kartu
           sudah punya bentuk & ukuran final SEJAK AWAL, bukan kosong melompong. */
        .img-frame.img-frame-placeholder {{
            border-style: dashed; border-color: rgba(0,43,92,0.28);
            background: linear-gradient(180deg, #F7F9FB 0%, #FCFDFE 100%);
            box-shadow: none;
        }}
        .img-body.placeholder-stripes {{
            background-image: repeating-linear-gradient(45deg, rgba(0,43,92,0.05) 0 10px, rgba(0,43,92,0.015) 10px 20px);
        }}
        .img-placeholder-text {{
            background: #FFFFFF; border: 1px solid rgba(0,43,92,.16); border-radius: 999px;
            padding: 7px 18px; font-size: 0.78rem; color: {p['ink_soft']};
            font-family: {FONT_MONO};
            box-shadow: 0 2px 6px rgba(0,43,92,.08);
        }}

        /* ===== NOTICE MODE SIMULASI -- kotak rapi, bukan span/caption polos ===== */
        .sim-notice {{
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border: 1.5px solid {WARN}; border-left: 4px solid {WARN};
            border-radius: 12px; padding: 12px 15px; margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(199,126,18,0.12);
        }}
        .sim-notice-title {{ font-weight: 800; color: {p['warn_ink']}; font-size: 0.88rem; margin-bottom: 4px; }}
        .sim-notice-body {{ font-size: 0.78rem; color: #78350F; line-height: 1.5; }}
        .sim-notice-body code {{ background: rgba(122,82,5,0.1); padding: 1px 5px; border-radius: 4px; font-size: 0.75rem; }}

        /* ===== PANEL HASIL (pasangan bingkai citra di kolom sebelah -- gauge + result box) ===== */
        .result-panel {{
            background: {p['card']}; border: 2px solid {p['border']}; border-radius: 14px;
            padding: 16px; box-shadow: 0 4px 14px rgba(0,43,92,0.08); margin-top: 8px;
        }}

        /* ===== WADAH INPUT (grup number_input dsb dalam st.container(border=True)) =====
           Catatan: sebelumnya pakai warna card_alt yang nyaris sama dengan bg halaman
           (nyaris tak kelihatan bedanya) -- sekarang pakai warna biru muda jelas + aksen
           kiri berwarna, supaya wadahnya benar-benar terlihat sebagai kotak terpisah. */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stContainer"] {{
            background: linear-gradient(135deg, #F7FAFD 0%, #EDF3F9 100%) !important;
            border: 1.5px solid #D6E3EF !important; border-left: 4px solid {PLN_BLUE} !important;
            border-radius: 14px !important; padding: 14px 16px !important; margin-bottom: 16px !important;
            box-shadow: 0 3px 12px rgba(0,147,221,0.10) !important;
        }}
        /* Input angka -- kasih background & border supaya tidak "melayang" di atas putih polos */
        div[data-testid="stNumberInput"] input {{
            background: #FFFFFF !important; border: 1.5px solid #D6E3EF !important;
            border-radius: 9px !important; color: {p['ink']} !important; font-weight: 600 !important;
            font-family: {FONT_MONO} !important;
        }}
        div[data-testid="stNumberInput"] label p {{ font-weight: 700 !important; color: {p['ink']} !important; }}
        div[data-testid="stNumberInput"] button {{
            background: #EDF3F9 !important; border: 1.5px solid #D6E3EF !important; color: {BLUE_MID} !important;
        }}

        /* ===== TABEL (st.dataframe) -- border & radius senada tema ===== */
        div[data-testid="stDataFrame"] {{
            border: 1.5px solid {p['border']} !important; border-radius: 12px !important;
            overflow: hidden;
        }}

        /* Tab lebih rapi */
        /* ===== TAB (st.tabs) -- selaras dengan skema navy/PLN-blue, bukan merah default ===== */
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{
            background: {p['card_alt']}; border-radius: 12px; padding: 5px;
            box-shadow: 0 2px 8px rgba(0,43,92,0.06); gap: 2px;
            border: 1px solid {p['border']};
        }}
        button[data-baseweb="tab"] {{
            font-size: 0.88rem; border-radius: 9px !important; color: {p['ink_soft']} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: {p['card']} !important; color: {NAVY} !important; font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(0,43,92,0.10) !important;
        }}
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{ background-color: {PLN_BLUE} !important; }}
        div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{ display: none; }}

        /* Responsif: di layar sempit, columns Streamlit sudah auto-stack; pastikan card tetap nyaman */
        @media (max-width: 640px) {{
            .step-card, .info-tile {{ padding: 14px; }}
        }}
        </style>
    """, unsafe_allow_html=True)


GT_LOGO_SVG = """
<svg width="68" height="68" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="navyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#002B5C"/>
      <stop offset="100%" stop-color="#001B3D"/>
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="46" fill="url(#navyGrad)" stroke="#F5B921" stroke-width="2.5"/>
  <ellipse cx="74" cy="26" rx="16" ry="7" fill="#FFFFFF" opacity="0.14" transform="rotate(45 74 26)"/>
  <ellipse cx="26" cy="74" rx="14" ry="6" fill="#FFFFFF" opacity="0.10" transform="rotate(45 26 74)"/>
  <text x="50" y="61" font-family="Segoe UI, Arial, sans-serif" font-size="32" font-weight="800"
        fill="#F5B921" text-anchor="middle">GT</text>
</svg>
"""


def render_sidebar(logo_path=None, compact=False):
    logo_size = "44px" if compact else "72px"
    logo_svg_sized = GT_LOGO_SVG.replace('width="68" height="68"', f'width="{logo_size}" height="{logo_size}"')

    if compact:
        st.sidebar.markdown(
            f"<div style='text-align:center;padding:10px 0 4px 0;'>{logo_svg_sized}</div>",
            unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            f"<div style='text-align:center;padding:10px 0 4px 0;'>{logo_svg_sized}"
            f"<div style='font-weight:800;font-size:1.05rem;color:#FFFFFF;margin-top:4px;letter-spacing:.02em;'>"
            f"GREENER TEKNO</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.66rem;letter-spacing:.14em;color:#F5B921;'>SOLUTIONS</div></div>",
            unsafe_allow_html=True)
        st.sidebar.caption("Konsorsium Riset PT PLN (Persero) x Universitas Diponegoro")
    st.sidebar.divider()


def sidebar_collapse_toggle():
    """Tombol bulat gaya 'SideBar UI' referensi -- klik untuk toggle mode
    compact (ikon saja) / mode penuh (ikon + label). Return True kalau diklik."""
    compact = st.session_state.get('sidebar_compact', False)
    icon = "▶" if compact else "◀"
    return st.sidebar.button(icon, key="sidebar_collapse_btn", help="Ciutkan/perbesar sidebar")


def render_header(title, subtitle, align="left"):
    """Header-card generik. align='center' dipakai khusus Beranda, halaman
    lain (NDT/DT/Keputusan) default 'left'."""
    st.markdown(f"""
    <div class="header-card" style="text-align:{align};">
        <div class="header-title">{title}</div>
        <div class="header-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def step_card(number, icon, title, desc):
    st.markdown(f"""
    <div class="step-card">
        <div class="step-num">{number}</div>
        <div class="icon">{icon}</div>
        <h4>{title}</h4>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)


def info_tile(label, value, accent="gray", explanation="", value_color=None, unit=""):
    """accent: 'gray'|'blue'|'green'|'purple'|'orange'|'teal'|'pink'|'gold' -- supaya kartu
    tidak terlihat seragam. explanation: teks singkat opsional di bawah nilai.
    value_color: override warna teks NILAI secara independen dari accent (accent
    tetap mengontrol garis aksen di atas kartu) -- dipakai supaya mis. kartu ber-
    accent "blue" tetap bisa menampilkan angka hijau/merah sesuai kondisi lolos/
    gagalnya, bukan cuma satu warna navy tetap. unit: satuan kecil opsional di
    sebelah nilai (mis. "%", "/ 8")."""
    cls = f"info-tile accent-{accent}" if accent != "gray" else "info-tile"
    expl_html = f'<div class="expl">{explanation}</div>' if explanation else ""
    val_style = f' style="color:{value_color};"' if value_color else ""
    unit_html = f' <span style="font-size:0.68em; font-weight:600; opacity:0.75;">{unit}</span>' if unit else ""
    st.markdown(f"""<div class="{cls}"><div class="lbl">{label}</div><div class="val"{val_style}>{value}{unit_html}</div>{expl_html}</div>""",
                unsafe_allow_html=True)


def donut_score(skor, warna, label_atas="SKOR KELAYAKAN AGREGAT"):
    """Donut skor agregat 0-100 gaya referensi terbaru -- dipakai KHUSUS untuk
    presentasi visual ringkasan Beranda, TIDAK memengaruhi keputusan LAYAK/
    TIDAK LAYAK (lihat catatan di utils/scoring.py)."""
    track = "#E7EDF4"
    fig = go.Figure(go.Pie(
        values=[skor, max(100 - skor, 0.001)], hole=0.74, sort=False, direction="clockwise",
        marker=dict(colors=[warna, track], line=dict(color="#fff", width=2)),
        textinfo="none", showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title={"text": label_atas, "x": 0.5, "font": {"size": 11, "color": P["ink_soft"], "family": FONT_MONO}},
        annotations=[dict(text=f"<b>{skor:.0f}</b><br><span style='font-size:11px;color:{P['ink_soft']}'>dari 100</span>",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=34, color=NAVY, family=FONT_MONO))],
        height=250, margin=dict(l=10, r=10, t=44, b=10), paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def bar_parameter_scores(labels, skor, warna_list, ambang=70.0):
    """Bar chart skor per parameter (0-100) + garis putus-putus ambang
    kelayakan -- versi visual dari tabel ringkasan, dipakai di Beranda."""
    fig = go.Figure(go.Bar(
        x=labels, y=skor, marker_color=warna_list,
        text=[f"{s:.0f}" for s in skor], textposition="outside",
        textfont=dict(family=FONT_MONO, size=11, color=P["ink"]), hovertemplate="%{x}: %{y:.0f}<extra></extra>",
    ))
    fig.add_hline(y=ambang, line_dash="dash", line_color=FAIL, opacity=0.55,
                  annotation_text=f"ambang {ambang:.0f}", annotation_position="top right",
                  annotation_font=dict(size=10, color=FAIL))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=26, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 112], gridcolor="rgba(0,43,92,.07)",
                                                 title=dict(text="skor", font=dict(size=10, color=P["ink_soft"]))),
        xaxis=dict(tickfont=dict(size=10, color="#5A6A80")), bargap=0.42, showlegend=False,
    )
    return fig


def result_box(status, title, reason=""):
    """status: 'pass' | 'fail' | 'warn'. Warna EKSPLISIT, tidak akan invisible di tema manapun."""
    cls = {"pass": "result-pass", "fail": "result-fail", "warn": "result-warn"}[status]
    icon = {"pass": "✅", "fail": "❌", "warn": "⚠️"}[status]
    reason_html = f'<div class="reason">{reason}</div>' if reason else ""
    st.markdown(f'<div class="result-box {cls}">{icon} <b>{title}</b>{reason_html}</div>', unsafe_allow_html=True)


def plain_verdict(status, title, reason=""):
    """Versi POLOS (tanpa kotak/border/background berwarna) dari result_box()
    -- teks judul besar+warna+bold di tengah, deskripsi abu-abu di bawahnya.
    Dipakai KHUSUS di kartu donut Ringkasan Beranda (v77) supaya sesuai
    referensi tampilan yang dikirim user (kartu donut hasilnya cuma teks
    polos rata tengah, bukan kotak result_box() bergaris warna seperti yang
    dipakai di tempat lain aplikasi -- result_box() itu sendiri TIDAK diubah
    supaya tempat lain yang masih memakainya tidak ikut berubah)."""
    warna = {"pass": PASS, "fail": FAIL, "warn": WARN}[status]
    reason_html = f'<div class="plain-verdict-reason">{reason}</div>' if reason else ""
    st.markdown(
        f'<div class="plain-verdict" style="color:{warna};">{title}</div>{reason_html}',
        unsafe_allow_html=True)


def mini_score_bar(label, skor, warna):
    """Baris skor mini: label kiri + bar tipis + angka kanan -- dipakai di
    daftar sub-skor per kategori dalam kartu donut Ringkasan Beranda (v77),
    sesuai referensi tampilan yang dikirim user."""
    pct = max(0.0, min(100.0, skor))
    st.markdown(f"""
    <div class="mini-score-row">
        <span class="mini-score-label">{label}</span>
        <span class="mini-score-track"><span class="mini-score-fill" style="width:{pct:.1f}%; background:{warna};"></span></span>
        <span class="mini-score-value">{skor:.0f}</span>
    </div>
    """, unsafe_allow_html=True)


def result_box_detailed(status, icon, title, reason, extra_label="", extra_text=""):
    """Versi result_box yang menyatukan status + alasan + keterangan tambahan
    (misal deskripsi kondisi) dalam SATU kartu warna, supaya tidak terlihat
    terputus seperti dua kotak terpisah."""
    cls = {"pass": "result-pass", "fail": "result-fail", "warn": "result-warn"}[status]
    extra_html = (
        f'<div class="result-divider"></div>'
        f'<div class="result-extra-label">{extra_label}</div>'
        f'<div class="reason">{extra_text}</div>'
    ) if extra_text else ""
    st.markdown(
        f'<div class="result-box {cls}">'
        f'<div style="font-size:1.3rem;line-height:1;">{icon}</div>'
        f'<b style="font-size:1.02rem;">{title}</b>'
        f'<div class="reason">{reason}</div>'
        f'{extra_html}'
        f'</div>', unsafe_allow_html=True)


def nav_section_label(text):
    st.sidebar.markdown(f'<div class="nav-section-label">{text}</div>', unsafe_allow_html=True)


def nav_button(label, active, key, compact=False, icon=""):
    """Tombol navigasi vertikal sidebar. Mode NORMAL: teks label saja, TANPA
    ikon (permintaan user v72: tampilan profesional, tanpa ikon). Mode
    compact=True (sidebar diciutkan ke 88px, v75): tampilkan HANYA `icon`
    (emoji), teks label dipindah jadi tooltip (help=) -- v74 sempat coba
    memperkecil+mewrap teks label saat compact, hasilnya malah pecah jadi
    tumpukan huruf per baris (laporan user), jadi diganti ikon polos yang
    jelas & tidak pernah wrap aneh. HARUS dipanggil di dalam
    `with st.container(key="sidebar_nav_list"):` di app.py supaya CSS
    ".st-key-sidebar_nav_list" bisa menyasarnya secara TERISOLASI (lihat
    catatan panjang di apply_custom_css soal bug lama CSS nav generik yang
    dulu ikut menghajar tombol sidebar lain)."""
    display_text = icon if (compact and icon) else label
    return st.button(display_text, key=key, use_container_width=True,
                      type="primary" if active else "secondary",
                      help=label if compact else None)


def header_tabs(items, active_key, key_prefix="tab"):
    """Baris tombol tab navigasi utama -- ditaruh langsung di bawah header
    card, MENGGANTIKAN daftar tombol navigasi di sidebar (dulu nav_button()
    di sidebar) supaya tata letak lebih mirip referensi dashboard terbaru
    (navigasi di header, sidebar dipakai murni untuk parameter & bahasa).
    `items` = [(kunci_internal, label_tampilan), ...]. Return kunci yang baru
    diklik, atau None kalau tidak ada klik pada render ini.

    Dibungkus st.container(key="header_nav_tabs") supaya Streamlit memberi
    class pembungkus ".st-key-header_nav_tabs" yang dipakai CSS (lihat
    apply_custom_css) untuk membuat baris tab "menempel" langsung di bawah
    header-card (satu blok navy menyatu, tab aktif putih + garis emas,
    gaya folder-tab) -- BUKAN gaya tombol biasa yang dipakai tombol lain
    di halaman utama."""
    with st.container(key="header_nav_tabs"):
        cols = st.columns(len(items))
        clicked = None
        for col, (key, label) in zip(cols, items):
            with col:
                if st.button(label, key=f"{key_prefix}_{key}", use_container_width=True,
                             type="primary" if key == active_key else "secondary"):
                    clicked = key
    return clicked


def gauge_chart(value_pct, title="Confidence Score", threshold=50.0):
    p = P
    bar_color = GREEN if value_pct >= threshold else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 34, "color": p["ink"], "family": FONT_MONO}},
        title={"text": title, "font": {"size": 13, "color": p["ink_soft"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": p["ink_soft"]},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold], "color": "#F6E7E5"},
                {"range": [threshold, 100], "color": "#E6F0F7"},
            ],
            "threshold": {"line": {"color": p["ink"], "width": 2}, "thickness": 0.75, "value": threshold},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=15, r=15, t=35, b=5),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": p["ink"]})
    return fig


def framed_image(pil_image, caption="Citra", icon="🖼️", status="neutral", placeholder=None):
    """Tampilkan citra dalam bingkai kartu + caption. Warna bingkai mengikuti
    status hasil analisis: 'pass' (hijau), 'fail' (merah), 'neutral' (biru PLN,
    dipakai sebelum ada hasil analisis / saat baru upload).

    pil_image=None + placeholder="teks..." -> tampilkan kotak placeholder
    bergaris putus-putus (belum ada citra) alih-alih tidak menampilkan apa-apa
    sama sekali -- supaya tata letak kartu sudah terbentuk & konsisten SEJAK
    AWAL (sebelum citra diunggah), bukan baru muncul setelah upload."""
    if pil_image is None:
        ph_text = placeholder or "..."
        st.markdown(
            f'<div class="img-frame img-frame-placeholder">'
            f'<div class="img-caption" style="color:{P["ink_soft"]};">{icon} {caption}</div>'
            f'<div class="img-body placeholder-stripes"><span class="img-placeholder-text">{ph_text}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    colors_map = {
        "pass": (PASS, "linear-gradient(180deg, #EAF6EF 0%, #F6FBF8 100%)", "rgba(30,127,78,0.15)"),
        "fail": (FAIL, "linear-gradient(180deg, #FBEDEC 0%, #FEF7F7 100%)", "rgba(200,68,60,0.15)"),
        "neutral": (PLN_BLUE, "linear-gradient(180deg, #F0F5FA 0%, #FBFCFE 100%)", "rgba(0,147,221,0.15)"),
    }
    border_c, bg_grad, shadow_c = colors_map.get(status, colors_map["neutral"])

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<div class="img-frame" style="border-color:{border_c}; background:{bg_grad}; box-shadow:0 4px 14px {shadow_c};">'
        f'<div class="img-caption" style="color:{border_c};">{icon} {caption}</div>'
        f'<div class="img-body" style="background:rgba(0,0,0,0.03);"><img src="data:image/png;base64,{b64}" style="border-color:{shadow_c};"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def ring_chart(value_pct, title="Probabilitas", threshold=50.0, invert_color=True):
    """Cincin lingkaran PENUH (donut) dengan persentase di tengah -- khusus
    dipakai untuk hasil Mikrografi (beda gaya dari gauge setengah-lingkaran
    NDT X-Ray yang tetap dipertahankan seperti semula).

    invert_color=True: nilai TINGGI berarti BURUK (mis. probabilitas korosi)
    -> merah kalau >= threshold, hijau kalau di bawahnya.
    invert_color=False: nilai TINGGI berarti BAIK -> sebaliknya.
    """
    p = P
    is_bad = value_pct >= threshold
    if invert_color:
        ring_color = RED if is_bad else GREEN
    else:
        ring_color = GREEN if is_bad else RED

    fig = go.Figure(go.Pie(
        values=[value_pct, max(100 - value_pct, 0.001)],
        hole=0.72,
        marker=dict(colors=[ring_color, "#E7EDF4"], line=dict(color="#FFFFFF", width=2)),
        textinfo="none",
        sort=False,
        direction="clockwise",
        rotation=0,
        showlegend=False,
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": p["ink_soft"]}, "x": 0.5},
        annotations=[dict(
            text=f"<b>{value_pct:.1f}%</b>", x=0.5, y=0.5,
            font=dict(size=26, color=p["ink"], family=FONT_MONO), showarrow=False,
        )],
        height=230, margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def upload_header(icon, title, subtitle):
    """Header kecil di atas st.file_uploader -- ikon bulat + judul + subjudul,
    supaya terasa seperti kartu upload modal, bukan widget polos."""
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">'
        f'<div style="width:42px; height:42px; border-radius:50%; background:rgba(0,147,221,0.10); '
        f'display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex-shrink:0;">{icon}</div>'
        f'<div><div style="font-weight:800; font-size:1rem;">{title}</div>'
        f'<div style="font-size:0.78rem; color:{P["ink_soft"]};">{subtitle}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def xray_gauge_card(value_pct, title="Confidence Score", status="neutral",
                     kelas_label=None, model_label=None, badge_text=None):
    """Kartu hasil NDT gaya referensi terbaru: cincin gauge WARNA SOLID sesuai
    status (pass=hijau, fail=merah, neutral=biru PLN) -- TANPA jarum penunjuk
    dan tanpa gradasi warna seperti versi sebelumnya, dengan persentase +
    label "confidence" di tengah cincin. Kalau kelas_label/model_label/
    badge_text diisi, cincin ini digabung dalam SATU kartu dengan blok teks
    klasifikasi + lencana status di sampingnya (kiri-kanan, flexbox) --
    menyatukan apa yang sebelumnya 2 komponen terpisah (gauge polos + kotak
    hasil) supaya tampilannya lebih ringkas & profesional. Kalau parameter
    itu di-kosongkan, hanya cincin + `title` yang tampil (kompatibel dengan
    pemakaian lama)."""
    color_map = {"pass": PASS, "fail": FAIL, "warn": WARN, "neutral": PLN_BLUE}
    ring_c = color_map.get(status, color_map["neutral"])

    CARD_BG = "#FFFFFF"
    TRACK = "#E7EDF4"
    PCT_C = "#0F1B2D"

    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor(CARD_BG)
    fig.patch.set_alpha(0)
    ax.set_facecolor(CARD_BG)

    start_angle, end_angle = 225, -45  # sapuan 270 derajat, celah di bawah
    total_sweep = start_angle - end_angle
    value_frac = max(0.0, min(value_pct, 100.0)) / 100.0
    value_angle = start_angle - value_frac * total_sweep

    # track abu-abu muda sebagai latar penuh gauge
    ax.add_patch(mpatches.Wedge((0, 0), 1.0, end_angle, start_angle, width=0.22,
                                 facecolor=TRACK, edgecolor="none"))
    # arc progres -- WARNA SOLID mengikuti status (bukan gradasi), sesuai
    # referensi tampilan terbaru yang dikirim user.
    if value_frac > 0:
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, value_angle, start_angle, width=0.22,
                                     facecolor=ring_c, edgecolor="none"))
        # penanda kecil (notch) di ujung nilai -- pengganti jarum, gaya "slider handle"
        rad = np.deg2rad(value_angle)
        hx, hy = 0.89 * np.cos(rad), 0.89 * np.sin(rad)
        ax.add_patch(plt.Circle((hx, hy), 0.045, facecolor="#FFFFFF", edgecolor=ring_c,
                                 linewidth=2.2, zorder=6))

    ax.text(0, 0.06, f"{value_pct:.1f}%", ha="center", va="center",
             fontsize=21, fontweight="bold", color=PCT_C)
    ax.text(0, -0.14, (title or "confidence").lower(), ha="center", va="center",
             fontsize=9, color="#7A889C")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.35, 1.05)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()

    if kelas_label is None:
        # Mode lama: cincin saja dalam kartu kecil (dipertahankan untuk
        # kompatibilitas kalau ada pemanggilan tanpa info klasifikasi).
        st.markdown(
            f'<div style="background:{CARD_BG}; border:2px solid {ring_c}; border-radius:16px; '
            f'padding:14px; box-shadow:0 4px 14px rgba(0,43,92,0.10); max-width:260px; margin:0 auto;">'
            f'<img src="data:image/png;base64,{b64}" style="width:100%; display:block;">'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    model_tag = (model_label or "").upper()
    st.markdown(
        f'<div style="background:{CARD_BG}; border:2px solid {ring_c}; border-radius:16px; '
        f'padding:18px 20px; box-shadow:0 4px 14px rgba(0,43,92,0.10); '
        f'display:flex; align-items:center; gap:16px; flex-wrap:wrap;">'
        f'<img src="data:image/png;base64,{b64}" style="width:150px; height:150px; flex-shrink:0;">'
        f'<div style="flex:1; min-width:140px;">'
        f'<div style="font-family:{FONT_MONO}; font-size:0.66rem; font-weight:600; color:#7A889C; '
        f'text-transform:uppercase; letter-spacing:.08em; line-height:1.5;">KLASIFIKASI<br>{model_tag}</div>'
        f'<div style="font-size:1.3rem; font-weight:800; color:{ring_c}; line-height:1.18; margin:6px 0 12px 0;">{kelas_label}</div>'
        f'<span style="background:{ring_c}; color:#FFFFFF; font-weight:700; font-size:0.76rem; '
        f'padding:5px 15px; border-radius:999px; display:inline-block; letter-spacing:.02em;">{badge_text}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def class_chip_row(class_labels, active_label=None, status="neutral"):
    """Baris chip/pill HANYA UNTUK TAMPILAN (read-only, bukan input) yang
    menampilkan semua kelas NDT yang bisa dideteksi model, dengan kelas hasil
    prediksi disorot -- meniru referensi terbaru yang dikirim user (baris
    NORMAL/BENDING/MEKAR/MULUR/PUTUS di bawah citra, kelas terprediksi
    ditonjolkan). Kelas aktif TETAP ditentukan sepenuhnya oleh hasil inferensi
    model saat upload (parameter `active_label`) -- komponen ini tidak
    menambah input manual apa pun, sejalan dengan keputusan sebelumnya untuk
    tidak mengadopsi panel input manual dari paket desain baru.

    `class_labels`: iterable nama kelas penuh (mis. "KONDUKTOR NORMAL", ...).
    Ditampilkan dipendekkan (tanpa prefiks "KONDUKTOR ") supaya ringkas.
    `status`: status kelas aktif ("pass"/"fail"/"warn"/"neutral") -- menentukan
    warna sorot chip yang aktif (hijau kalau NORMAL/pass, merah kalau fail)."""
    color_map = {"pass": PASS, "fail": FAIL, "warn": WARN, "neutral": PLN_BLUE}
    active_c = color_map.get(status, color_map["neutral"])
    active_norm = (active_label or "").strip().upper()

    chips_html = []
    for lbl in class_labels:
        lbl_norm = lbl.strip().upper()
        short = lbl_norm.replace("KONDUKTOR ", "").strip()
        is_active = active_norm and lbl_norm == active_norm
        if is_active:
            chips_html.append(
                f'<span style="background:{active_c}; color:#FFFFFF; font-weight:700; '
                f'font-size:0.72rem; padding:6px 14px; border-radius:999px; '
                f'box-shadow:0 2px 8px rgba(0,43,92,0.18); letter-spacing:.02em; '
                f'white-space:nowrap;">{short}</span>'
            )
        else:
            chips_html.append(
                f'<span style="background:#F1F4F9; color:{P["ink_soft"]}; font-weight:600; '
                f'font-size:0.72rem; padding:6px 14px; border-radius:999px; '
                f'border:1px solid {P["border"]}; letter-spacing:.02em; '
                f'white-space:nowrap;">{short}</span>'
            )

    st.markdown(
        '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; justify-content:center;">'
        + "".join(chips_html) +
        '</div>',
        unsafe_allow_html=True,
    )


# SVG bendera sederhana (BUKAN emoji) -- supaya render konsisten sebagai
# gambar di semua device, tidak jadi fallback teks "ID"/"GB" seperti emoji
# regional-indicator yang tidak didukung sebagian font/OS. Di-encode base64
# dan ditampilkan via st.markdown(<img>) di language_toggle() -- GAMBAR HTML
# BIASA, bukan CSS background-image lewat class ".st-key-..." seperti versi
# sebelumnya (itu ternyata tidak match di Streamlit versi kamu, jadi bendera
# sempat tidak kelihatan sama sekali). <img> pasti tampil di versi apapun.
FLAG_ID_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
<rect width="24" height="12" y="0" fill="#DC2626"/>
<rect width="24" height="12" y="12" fill="#FFFFFF"/>
</svg>"""

FLAG_EN_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
<rect width="24" height="24" fill="#1E3A8A"/>
<rect width="24" height="4" y="10" fill="#FFFFFF"/>
<rect width="4" height="24" x="10" fill="#FFFFFF"/>
<rect width="24" height="2" y="11" fill="#DC2626"/>
<rect width="2" height="24" x="11" fill="#DC2626"/>
</svg>"""

FLAG_ID_B64 = base64.b64encode(FLAG_ID_SVG.encode("utf-8")).decode("ascii")
FLAG_EN_B64 = base64.b64encode(FLAG_EN_SVG.encode("utf-8")).decode("ascii")


def _flag_img_html(b64, is_active):
    """Tag <img> bendera bulat kecil. Cincin "aktif" ditentukan LANGSUNG di
    Python lewat is_active (bukan CSS kind="primary" yang terbukti tidak
    reliable di semua versi Streamlit) -- jadi selalu benar apapun versinya.
    Cincin aktif dibuat EMAS (aksen Undip/PLN) supaya senada tema navy baru."""
    ring = f"box-shadow:0 0 0 2px {GOLD}, 0 1px 3px rgba(0,27,61,0.35);" if is_active else "box-shadow:0 1px 3px rgba(0,27,61,0.20);"
    return (f'<img src="data:image/svg+xml;base64,{b64}" '
            f'style="width:19px;height:19px;border-radius:50%;display:block;margin:0 auto;{ring}">')


def language_toggle(lang):
    """Toggle bahasa ID/EN -- ikon bendera bulat kecil, tanpa teks "ID"/"EN"
    tambahan. Bendera digambar via st.markdown (dijamin tampil), tombol klik
    kecil tanpa teks ditaruh rapat di bawahnya sebagai area klik. Panggil di
    kolom sempit (lihat pemakaian di app.py). Return bahasa terpilih ('id'/'en').

    v77 FIX: seluruh baris (st.columns(2) internal) dibungkus st.container(
    key="lang_toggle_row") -- BUKAN cuma kosmetik, ini memperbaiki bug nyata:
    CSS lama menyasar div[data-testid="stHorizontalBlock"]:has(.st-key-lang_toggle_id)
    tanpa pembungkus khusus ini, dan :has() mencocokkan keturunan di
    KEDALAMAN BERAPAPUN -- akibatnya bukan cuma baris horizontal internal
    (2 kolom bendera) yang kena, tapi baris horizontal LUAR (col_brand+
    col_lang di app.py) juga ikut match karena .st-key-lang_toggle_id tetap
    jadi keturunannya (lewat col_lang). Hasilnya SELURUH baris header (brand
    row "UNDIP x PLN" + toggle bahasa) dipaksa shrink-to-fit & rata-kanan,
    bikin teks brand ikut "tersedot" ke kanan nempel bendera. Dengan
    container key khusus ini, CSS di apply_custom_css() bisa menyasar
    ".st-key-lang_toggle_row div[data-testid='stHorizontalBlock']" secara
    LANGSUNG (tanpa :has()), jadi PASTI cuma baris 2-bendera internal yang
    kena, baris luar (brand+lang) tidak tersentuh sama sekali."""
    with st.container(key="lang_toggle_row"):
        col_id, col_en = st.columns(2, gap="small")
        with col_id:
            st.markdown(_flag_img_html(FLAG_ID_B64, lang == "id"), unsafe_allow_html=True)
            # Dibungkus st.container(key=...) supaya CSS bisa menyasar tombol ini
            # secara ANDAL lewat ".st-key-lang_toggle_id" (lihat apply_custom_css)
            # -- sebelumnya cuma mengandalkan atribut title=/key= tombol langsung
            # yang di beberapa kondisi terbukti tidak match, membuat tombol jatuh
            # ke ukuran default Streamlit (kotak besar) dan terlihat tumpang tindih
            # dengan bendera di atasnya.
            with st.container(key="lang_toggle_id"):
                if st.button("", key="lang_id_btn", help="Bahasa Indonesia", use_container_width=True):
                    return "id"
        with col_en:
            st.markdown(_flag_img_html(FLAG_EN_B64, lang == "en"), unsafe_allow_html=True)
            with st.container(key="lang_toggle_en"):
                if st.button("", key="lang_en_btn", help="English", use_container_width=True):
                    return "en"
    return lang
