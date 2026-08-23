"""LANGKAH 1 -- menjalankan seluruh lengan eksperimen.

    python 1_eksperimen.py --lihat          # periksa rencananya dulu
    python 1_eksperimen.py                  # 30 hari  (klaim pokok)
    python 1_eksperimen.py --horizon 90d    # 90 hari  (uji ketahanan)

Tiga lengan x tiga tingkat kepercayaan awal = 9 pelatihan per horizon:

    h6b_utama        penyatuan + penyeimbang   <- yang diuji
    h1a_pemerataan   pemerataan saja           <- sistem asal 1
    h2a_selera       selera saja               <- sistem asal 2

Baseline greedy (queue & util) dihitung otomatis di tiap lengan, tak perlu dijalankan
terpisah.

Kenapa ketiga lengan harus dijalankan, bukan cuma yang diuji: klaim tesis adalah
penyatuan mengungguli KEDUA sistem asal. Tanpa kedua sistem asal dijalankan pada kondisi
yang sama persis, klaim itu tak punya pembanding -- membandingkan dengan angka dari lini
eksperimen lama tidak sah karena protokolnya berbeda.

Kenapa tiga tingkat kepercayaan, bukan satu: rumusan masalah tesis ini adalah meratakan
beban TANPA meruntuhkan kepercayaan. Kalau semua dijalankan pada satu tingkat
kepercayaan, dinamika kepercayaan cuma jadi latar, tak pernah jadi variabel yang diuji.
Biaya tambahannya hanya 3x, dan analisis kurvanya gratis -- `2_analisis.py` membaca
hasil yang sama.

Tahan-putus: lengan yang berkasnya sudah ada akan dilewati. Aman dijalankan ulang.
"""
import sys, os, argparse, subprocess, datetime, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kakas as K

PIPELINE = os.path.join(HERE, "_pipeline_hipotesis.py")
STAMP = datetime.date.today().strftime("%Y%m%d")

# Pengaturan yang WAJIB sama di seluruh lengan. Dikunci di sini, bukan disalin ke tiap
# baris perintah -- menyalin dengan tangan adalah cara termudah membuat dua lengan tak
# sepadan tanpa disadari, dan itu sudah pernah terjadi (`K3_gap` tak terpakai karena dua
# hal berubah bersamaan).
BERSAMA = ["--forecaster", "vwf", "--reward-preset", "seimbang4x", "--dataset", "4x"]
N_EVAL_SEED = 10

# Rancangan 2x2 atas DUA teknik: teknik preferensi x pemisahan penilai.
#
#                      teknik preferensi   penilai
#   h1a_pemerataan     mati                3 terpisah   <- koordinasi saja
#   h2a_selera         hidup               1 gabungan   <- preferensi saja
#   h6b_utama          hidup               3 terpisah   <- keduanya
#
# TEKNIK PREFERENSI = modul selera (`--pref --pref-feature-mode`) DITAMBAH imbalan
# kepatuhan (`--alpha-accept 1.0`). Keduanya satu paket, bukan dua hal terpisah:
# modul menyediakan REPRESENTASI (siapa pengguna ini, apa yang ia suka), imbalan
# kepatuhan menyediakan OBJEKTIFNYA (apakah pencocokan itu benar-benar berbuah
# kepatuhan). Tanpa imbalan kepatuhan, modul preferensi belajar mencocokkan selera
# tanpa pernah diberi tahu apakah pencocokan itu berguna -- teknik yang tak lengkap.
#
# Keputusan 2026-08-23. Sebelumnya `--alpha-accept` hanya dipasang di h6b, sehingga h6b
# berbeda dari TIAP induk dalam dua hal sekaligus dan komponen penyebab kemenangan tak
# dapat disimpulkan. Dengan memasukkannya ke dalam paket teknik preferensi, kedua
# perbandingan menjadi SATU FAKTOR:
#     h6b vs h1a (Gini)       -> bedanya teknik preferensi
#     h6b vs h2a (penerimaan) -> bedanya pemisahan penilai
#
# Konsekuensi yang diterima: klaim "imbalan kepatuhan menyelamatkan penyatuan naif"
# gugur, karena suku itu kini ada di baseline juga. Ditukar dengan hasil yang bisa
# diatribusikan -- klaim lama pun belum pernah terverifikasi.
#
# ASUMSI yang datanya akan terlihat sendiri: framing ini benar bila erosi penerimaan
# memang KHAS preferensi. `h1a_pemerataan` sengaja tetap tanpa imbalan kepatuhan, jadi
# bila penerimaannya ternyata ikut tergerus parah, berarti suku itu perbaikan umum --
# bukan milik teknik preferensi -- dan framing ini harus dicatat sebagai keterbatasan.
#
# `--no-hist` WAJIB ada di ketiganya. Bawaan kelas kebijakan adalah `use_hist=True`,
# jadi lengan yang tak menyebutkannya diam-diam mendapat penyandi riwayat interaksi
# (proxy kepercayaan berbasis LSTM) yang tak dimiliki lengan lain. Sempat terjadi pada
# `h1a_pemerataan` dan baru ketahuan saat memeriksa ulang.
PAKET_PREFERENSI = ["--pref", "--pref-feature-mode", "--alpha-accept", "1.0"]

LENGAN = [
    ("h6b_utama",
     PAKET_PREFERENSI + ["--no-hist", "--n-critics", "3"],
     "preferensi + penilai terpisah (yang diuji)"),
    ("h1a_pemerataan",
     ["--no-hist", "--n-critics", "3"],
     "koordinasi saja, tanpa teknik preferensi"),
    ("h2a_selera",
     PAKET_PREFERENSI + ["--no-hist", "--n-critics", "1"],
     "preferensi saja, tanpa penilai terpisah"),
]


# Hasil mode cepat diberi awalan tersendiri. Tanpa ini, uji rangkaian yang dijalankan
# pada hari yang sama akan menghasilkan nama berkas IDENTIK dengan run sungguhan --
# sehingga run sungguhan dilewati ("sudah ada") dan yang terbaca `2_analisis.py` adalah
# hasil 3-pembaruan yang tak berarti. Jebakan yang sangat mudah tak disadari, karena
# semuanya tampak berjalan normal.
AWALAN_CEPAT = "zzcepat_"


def nama_berkas(tag, it, horizon, cepat=False):
    suf = "" if horizon == "30d" else f"_{horizon}"
    pre = AWALAN_CEPAT if cepat else ""
    return f"{pre}{tag}__it{it.replace('.', '')}{suf}_{STAMP}_eval.json"


def sudah_ada(tag, it, horizon, cepat=False):
    return os.path.exists(os.path.join(K.OUTDIR, nama_berkas(tag, it, horizon, cepat)))


def periksa_kesepadanan():
    """Menjaga agar perbedaan antar-lengan hanya yang DISENGAJA.

    Cacat yang dicegah di sini sudah pernah terjadi: `--no-hist` tertinggal di satu
    lengan, sehingga lengan itu diam-diam memakai penyandi riwayat yang tak dimiliki
    lengan lain. Tak ada galat, tak ada peringatan -- angkanya saja yang jadi tak
    sepadan. Diperiksa di kode, bukan dipercayakan pada ketelitian membaca."""
    bendera = {tag: {x for x in khas if x.startswith("--")} for tag, khas, _ in LENGAN}
    nilai = {tag: dict(zip(khas[::1][:-1], khas[1:]))
             for tag, khas, _ in LENGAN}

    # 1. `--no-hist` wajib di SEMUA lengan (bawaan kelas kebijakan use_hist=True).
    tanpa = [t for t, b in bendera.items() if "--no-hist" not in b]
    assert not tanpa, (
        f"`--no-hist` tertinggal di: {tanpa}. Bawaan kelas kebijakan adalah "
        f"use_hist=True, jadi lengan itu akan diam-diam memakai penyandi riwayat "
        f"yang tak dimiliki lengan lain -- perbandingannya jadi tak sepadan.")

    # 2. Teknik preferensi adalah SATU PAKET. Modul selera dan imbalan kepatuhan harus
    #    selalu hidup bersama atau mati bersama -- kalau terpisah, perbandingan
    #    h6b-vs-induk kembali jadi dua faktor dan tak bisa diatribusikan.
    for tag, khas, _ in LENGAN:
        punya_modul = "--pref" in khas
        punya_accept = float(nilai[tag].get("--alpha-accept", 0.0) or 0.0) != 0.0
        assert punya_modul == punya_accept, (
            f"Paket teknik preferensi pecah di `{tag}`: modul selera="
            f"{punya_modul}, imbalan kepatuhan={punya_accept}. Keduanya harus hidup "
            f"bersama atau mati bersama -- modul menyediakan representasi, imbalan "
            f"kepatuhan menyediakan objektifnya. Memisahkannya membuat h6b berbeda "
            f"dari induknya dalam DUA hal, dan penyebab kemenangan tak dapat "
            f"disimpulkan.")

    # 3. Tepat SATU lengan tanpa teknik preferensi (yaitu koordinasi-saja).
    tanpa_pref = [t for t, khas, _ in LENGAN if "--pref" not in khas]
    assert tanpa_pref == ["h1a_pemerataan"], (
        f"Rancangan 2x2 rusak: yang TANPA teknik preferensi seharusnya hanya "
        f"h1a_pemerataan, tetapi yang ditemukan {tanpa_pref}.")

    # 4. Tepat SATU lengan memakai penilai gabungan (yaitu preferensi-saja).
    gabungan = [t for t, v in nilai.items() if v.get("--n-critics") == "1"]
    assert gabungan == ["h2a_selera"], (
        f"Rancangan 2x2 rusak: yang memakai penilai GABUNGAN seharusnya hanya "
        f"h2a_selera, tetapi yang ditemukan {gabungan}.")


def main():
    periksa_kesepadanan()
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=str, default="30d", choices=["30d", "90d"])
    p.add_argument("--hanya", type=str, default=None, help="jalankan satu lengan saja (nama tag)")
    p.add_argument("--lihat", action="store_true", help="tampilkan rencana, jangan jalankan")
    p.add_argument("--cepat", action="store_true",
                   help="1 seed x 3 pembaruan x 2 seed evaluasi -- hanya memastikan "
                        "rangkaiannya utuh. HASILNYA TIDAK SAH dilaporkan.")
    p.add_argument("--n-train-seed", type=int, default=5)
    p.add_argument("--n-updates", type=int, default=300)
    args = p.parse_args()

    n_seed = 1 if args.cepat else args.n_train_seed
    n_upd = 3 if args.cepat else args.n_updates
    # Evaluasi ikut dipangkas di mode cepat. Tanpa ini `--cepat` nyaris tak lebih cepat:
    # tiap lengan menjalankan n_eval_seed simulasi kebijakan + 2 x n_eval_seed simulasi
    # greedy, dan simulasi 30 hari jauh lebih lama daripada 3 pembaruan pelatihan.
    n_eval = 2 if args.cepat else N_EVAL_SEED

    ds = ["--dataset", K.DATASET_90D] if args.horizon == "90d" else []
    horizon_arg = ["--horizon", args.horizon]

    pilih = LENGAN if not args.hanya else [l for l in LENGAN if l[0] == args.hanya]
    if not pilih:
        sys.exit(f"tag tak dikenal: {args.hanya}\npilihan: " + ", ".join(l[0] for l in LENGAN))

    tugas = [(tag, khas, it, ket) for tag, khas, ket in pilih for it in K.TINGKAT_TRUST]

    K.judul(f"RENCANA -- {len(tugas)} lengan, horizon {args.horizon}, "
            f"{n_seed} seed x {n_upd} pembaruan x {n_eval} seed evaluasi")
    if args.cepat:
        print("!! MODE CEPAT: hasilnya TIDAK SAH untuk dilaporkan\n")
    lewati = 0
    for tag, khas, it, ket in tugas:
        ada = sudah_ada(tag, it, args.horizon, args.cepat)
        lewati += ada
        print(f"  {'LEWATI' if ada else 'jalan '}  {tag:<16} it={it}   {ket}")
    print(f"\n  {len(tugas)-lewati} akan dijalankan, {lewati} dilewati (sudah ada)")
    if args.cepat:
        print(f"  berkasnya berawalan `{AWALAN_CEPAT}` -- tak akan tertukar dengan hasil "
              f"sungguhan, dan diabaikan `2_analisis.py`")

    if args.lihat:
        print("\n(--lihat: tidak ada yang dijalankan)")
        return

    t0 = time.time()
    for i, (tag, khas, it, ket) in enumerate(tugas, 1):
        if sudah_ada(tag, it, args.horizon, args.cepat):
            print(f"\n[{i}/{len(tugas)}] {tag} it={it} -- LEWATI", flush=True)
            continue
        tag_jalan = (AWALAN_CEPAT + tag) if args.cepat else tag
        cmd = ([sys.executable, PIPELINE, "--tag", tag_jalan, "--initial-trust", it]
               + khas + BERSAMA + ds + horizon_arg
               + ["--n-train-seed", str(n_seed), "--n-updates", str(n_upd),
                  "--n-eval-seed", str(n_eval)])
        print(f"\n[{i}/{len(tugas)}] {tag} it={it}  ({time.time()-t0:.0f}s berlalu)", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\n!! GAGAL pada {tag} it={it} (kode {r.returncode}).")
            print("   Jalankan ulang perintah yang sama -- yang sudah selesai akan dilewati.")
            sys.exit(r.returncode)

    K.judul(f"SELESAI dalam {time.time()-t0:.0f}s")
    print(f"Lanjut:  python 2_analisis.py --horizon {args.horizon}")


if __name__ == "__main__":
    main()
