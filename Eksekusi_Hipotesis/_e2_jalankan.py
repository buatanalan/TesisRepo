"""E2/E3 -- menjalankan seluruh lengan berurutan, satu perintah.

Menggantikan penyalinan gelung bash: di sini urutan, penamaan, dan pengaturan tiap
lengan sudah terkunci di kode, sehingga tak mungkin ada lengan yang tak sengaja
dijalankan dengan pengaturan berbeda. Itu bukan kerapian belaka -- E3 (uji atribusi)
hanya sah bila lengan `h6b_tanpa_accept` identik dengan `h6b_utama` KECUALI satu
pengaturan. Menyalin perintah dengan tangan adalah cara paling mudah merusak syarat itu,
dan sudah pernah terjadi (`K3_gap` tak terpakai karena dua hal berubah sekaligus).

Tahan-putus: lengan yang berkasnya sudah ada akan dilewati, jadi aman dijalankan ulang
setelah terputus.

    python _e2_jalankan.py                 # E2 penuh: 3 lengan x 3 tingkat kepercayaan
    python _e2_jalankan.py --hanya e3      # cuma lengan atribusi E3
    python _e2_jalankan.py --lihat         # tampilkan rencana, tidak menjalankan
    python _e2_jalankan.py --cepat         # anggaran mini, untuk memastikan rangkaian utuh
"""
import sys, os, json, argparse, subprocess, datetime, time

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(_HERE, "outputs")
PIPELINE = os.path.join(_HERE, "_pipeline_hipotesis.py")
STAMP = datetime.date.today().strftime("%Y%m%d")

# Pengaturan bersama SELURUH lengan. Apa pun yang ada di sini tidak boleh berbeda
# antar-lengan, kalau tidak perbandingannya tidak bisa diatribusikan.
BERSAMA = ["--forecaster", "vwf", "--reward-preset", "seimbang4x",
           "--n-eval-seed", "10", "--horizon", "30d", "--dataset", "4x"]

TINGKAT = ["0.3", "0.5", "0.7"]

# (tag, pengaturan khas lengan, tingkat kepercayaan yang dijalankan, keterangan)
LENGAN = [
    ("h6b_utama",
     ["--pref", "--pref-feature-mode", "--no-hist", "--alpha-accept", "1.0", "--n-critics", "3"],
     TINGKAT, "penyatuan + penyeimbang (yang diuji)"),
    ("h1a_pemerataan",
     ["--n-critics", "3"],
     TINGKAT, "pemerataan saja, tanpa modul selera (H1a)"),
    ("h2a_selera",
     ["--pref", "--pref-feature-mode", "--no-hist", "--n-critics", "1"],
     TINGKAT, "selera saja, tanpa penilai terpisah (H2a)"),
    ("h6b_tanpa_accept",
     ["--pref", "--pref-feature-mode", "--no-hist", "--alpha-accept", "0.0", "--n-critics", "3"],
     ["0.5"], "E3 -- atribusi: sama persis h6b_utama KECUALI alpha-accept"),
]


def nama_berkas(tag, it):
    return f"{tag}__it{it.replace('.', '')}_{STAMP}_eval.json"


def sudah_ada(tag, it):
    return os.path.exists(os.path.join(OUTDIR, nama_berkas(tag, it)))


def periksa_atribusi():
    """Menjaga syarat E3: `h6b_tanpa_accept` harus identik dengan `h6b_utama` kecuali
    tepat satu pengaturan. Diperiksa di sini, bukan dipercayakan pada ketelitian membaca."""
    a = dict(zip(*[iter([x for x in LENGAN[0][1] if x != "--pref" and x != "--pref-feature-mode"
                         and x != "--no-hist"])] * 2))
    b = dict(zip(*[iter([x for x in LENGAN[3][1] if x != "--pref" and x != "--pref-feature-mode"
                         and x != "--no-hist"])] * 2))
    beda = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    bendera_a = {x for x in LENGAN[0][1] if x.startswith("--") and x not in a}
    bendera_b = {x for x in LENGAN[3][1] if x.startswith("--") and x not in b}
    assert bendera_a == bendera_b, (
        f"E3 rusak: bendera lengan berbeda -- hanya di h6b_utama {bendera_a - bendera_b}, "
        f"hanya di h6b_tanpa_accept {bendera_b - bendera_a}")
    assert beda == {"--alpha-accept"}, (
        f"E3 rusak: yang berbeda antara h6b_utama dan h6b_tanpa_accept seharusnya HANYA "
        f"--alpha-accept, tetapi yang berbeda adalah {beda}. Uji atribusi tidak sah "
        f"selama lebih dari satu hal berubah.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hanya", type=str, default=None,
                   help="jalankan sebagian: 'e2' (3 lengan pokok) | 'e3' (lengan atribusi) "
                        "| nama tag persis")
    p.add_argument("--lihat", action="store_true", help="tampilkan rencana, jangan jalankan")
    p.add_argument("--cepat", action="store_true",
                   help="anggaran mini (1 seed, 3 pembaruan) -- hanya untuk memastikan "
                        "rangkaiannya utuh, HASILNYA TIDAK SAH untuk dilaporkan")
    p.add_argument("--n-train-seed", type=int, default=5)
    p.add_argument("--n-updates", type=int, default=300)
    args = p.parse_args()

    periksa_atribusi()

    n_seed = 1 if args.cepat else args.n_train_seed
    n_upd = 3 if args.cepat else args.n_updates

    pilih = LENGAN
    if args.hanya == "e2":
        pilih = LENGAN[:3]
    elif args.hanya == "e3":
        pilih = LENGAN[3:]
    elif args.hanya:
        pilih = [l for l in LENGAN if l[0] == args.hanya]
        if not pilih:
            sys.exit(f"tag tak dikenal: {args.hanya}")

    tugas = []
    for tag, khas, tingkat, ket in pilih:
        for it in tingkat:
            tugas.append((tag, khas, it, ket))

    print("=" * 72)
    print(f"RENCANA -- {len(tugas)} lengan, {n_seed} seed x {n_upd} pembaruan")
    if args.cepat:
        print("!! MODE CEPAT: hasilnya TIDAK SAH untuk dilaporkan")
    print("=" * 72)
    lewati = 0
    for tag, khas, it, ket in tugas:
        ada = sudah_ada(tag, it)
        lewati += ada
        print(f"  {'LEWATI' if ada else 'jalan '}  {tag:20s} it={it}   {ket}")
    print(f"\n{len(tugas)-lewati} akan dijalankan, {lewati} dilewati (sudah ada)")

    if args.lihat:
        print("\n(--lihat: tidak ada yang dijalankan)")
        return

    t0 = time.time()
    for i, (tag, khas, it, ket) in enumerate(tugas, 1):
        if sudah_ada(tag, it):
            print(f"\n[{i}/{len(tugas)}] {tag} it={it} -- LEWATI", flush=True)
            continue
        cmd = ([sys.executable, PIPELINE, "--tag", tag, "--initial-trust", it]
               + khas + BERSAMA
               + ["--n-train-seed", str(n_seed), "--n-updates", str(n_upd)])
        print(f"\n[{i}/{len(tugas)}] {tag} it={it}  ({time.time()-t0:.0f}s berlalu)", flush=True)
        print("  " + " ".join(cmd[1:]), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\n!! GAGAL pada {tag} it={it} (kode {r.returncode}). "
                  f"Jalankan ulang perintah ini -- yang sudah selesai akan dilewati.")
            sys.exit(r.returncode)

    print("\n" + "=" * 72)
    print(f"SEMUA SELESAI dalam {time.time()-t0:.0f}s")
    print("Lanjut:  python _e2_banding.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
