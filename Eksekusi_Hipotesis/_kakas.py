"""Perkakas bersama seluruh skrip Eksekusi_Hipotesis.

Bukan skrip yang dijalankan langsung (awalan `_`). Berisi hal-hal yang kalau ditulis
ulang di tiap skrip akan pelan-pelan menyimpang satu sama lain -- terutama pemuatan
hasil, format tabel, dan ukuran efek. Perbedaan kecil di situ membuat angka antar-tahap
tak lagi sepadan tanpa ada yang menyadarinya.
"""
import sys, os, json, glob, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Eksekusi_RL"))
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(HERE, "outputs")
LOGDIR = os.path.join(HERE, "logs")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

DATASET_30D = "scenario_dataset_klaster12_4x.json"
DATASET_90D = "scenario_dataset_klaster12_4x_90d.json"
DATASET_1X = "scenario_dataset_klaster12.json"

TINGKAT_TRUST = ["0.3", "0.5", "0.7"]

# Empat metrik inti. Arah "lebih baik": gini turun, penerimaan naik, tunggu turun,
# kepercayaan naik. Dipakai seragam di seluruh tabel supaya tanda panah tak pernah
# ditafsirkan terbalik antar-tahap.
METRIK = [("gini", "Gini", -1), ("acc", "Terima", +1),
          ("wait", "Tunggu", -1), ("trust", "Percaya", +1)]


@contextlib.contextmanager
def mode_trust(mode):
    """Alihkan aturan penalti kepercayaan: 'abs' (dua arah, meleset ke arah mana pun
    menghukum) atau 'signed' (satu arah, hanya keterlambatan menghukum).

    Aturan 'signed' SUDAH menjadi bawaan kode (`user.TRUST_PENALTY_MODE`). E1 memakai
    konteks ini untuk mengembalikan sementara ke 'abs' sebagai pembanding -- tanpa
    pembanding pada horizon yang SAMA, tiap perbedaan tak bisa diatribusikan ke aturan."""
    import marl_spklu.env.user as U
    assert mode in ("abs", "signed"), f"mode aturan tak dikenal: {mode}"
    asli = U.TRUST_PENALTY_MODE
    U.TRUST_PENALTY_MODE = mode
    try:
        yield
    finally:
        U.TRUST_PENALTY_MODE = asli


def metrik_sim(sim):
    """Empat metrik inti + sebaran kepercayaan. SATU definisi dipakai semua skrip --
    disalin dari `_pipeline_hipotesis._metrik` dengan sengaja identik."""
    import common
    sv = np.array([s.total_served for s in sim.spklus.values()], float)
    w = np.array([l["wait_time"] for l in sim.logs], float)
    c = np.array([bool(l["complied"]) for l in sim.logs], bool)
    tr = np.array([u.trust for u in sim.users], float)
    return dict(
        gini=common.gini(sv), served=int(sv.sum()),
        acc=float(c.mean()) if c.size else 0.0,
        wait=float(w.mean()) if w.size else 0.0,
        trust=float(tr.mean()), trust_sd=float(tr.std()),
        trust_p10=float(np.percentile(tr, 10)), trust_p50=float(np.percentile(tr, 50)),
        trust_p90=float(np.percentile(tr, 90)),
        trust_frac_bawah_03=float((tr < 0.3).mean()),
    )


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def muat_eval(tag, it=None):
    """Muat hasil `_pipeline_hipotesis.py` untuk satu lengan. Bila ada beberapa tanggal,
    yang TERBARU dipakai, dan sisanya disebutkan -- supaya tak diam-diam memakai hasil
    lama tanpa pemberitahuan."""
    suf = f"__it{it.replace('.', '')}" if it else ""
    pola = os.path.join(OUTDIR, f"{tag}{suf}_*_eval.json")
    berkas = sorted(glob.glob(pola))
    if not berkas:
        return None
    if len(berkas) > 1:
        print(f"    ! {tag}{suf}: {len(berkas)} tanggal ditemukan, memakai "
              f"{os.path.basename(berkas[-1])}", flush=True)
    with open(berkas[-1], encoding="utf-8") as f:
        d = json.load(f)
    d["_berkas"] = os.path.basename(berkas[-1])
    return d


def simpan(obj, nama):
    path = os.path.join(OUTDIR, nama)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"\n-> outputs/{nama}", flush=True)
    return path


def judul(teks, lebar=76):
    print("\n" + "=" * lebar)
    print(teks)
    print("=" * lebar, flush=True)


def baris_metrik(nama, r, lebar=22):
    if r is None:
        return f"  {nama:<{lebar}}  {'(belum ada)':>40}"
    return (f"  {nama:<{lebar}}  {r['gini']:>8.4f}  {r['acc']:>7.3f}  "
            f"{r['wait']:>8.1f}  {r['trust']:>8.3f}")


def kepala_metrik(lebar=22):
    return f"  {'':<{lebar}}  {'Gini':>8}  {'Terima':>7}  {'Tunggu':>8}  {'Percaya':>8}"


def vonis(lulus, teks_lulus, teks_gagal):
    tanda = "LULUS" if lulus else "TIDAK LULUS"
    return f"[{tanda}] {teks_lulus if lulus else teks_gagal}"


def butuh(kondisi, pesan):
    """Berhenti dengan pesan yang menjelaskan APA yang harus dijalankan lebih dulu,
    bukan sekadar melempar galat berkas-tak-ditemukan."""
    if not kondisi:
        print(f"\n!! {pesan}", flush=True)
        sys.exit(1)
