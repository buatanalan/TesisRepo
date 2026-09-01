"""Pemuat & penghitung bersama untuk analisis Bab V (RENCANA_ANALISIS_HASIL.md).

SATU sumber angka untuk SELURUH blok analisis -- tabel di Bab V dijamin konsisten
dan dapat direproduksi. Jangan menghitung statistik langsung di notebook; panggil
fungsi di sini.

DISIPLIN UNIT ANALISIS (RENCANA §2.1) -- alasan modul ini ada:
    Lengan RL dilatih 3 seed (-> 3 checkpoint kebijakan) lalu dievaluasi 10 kali
    dengan checkpoint BERGILIR. 10 run eval BUKAN 10 sampel independen: run yang
    berbagi checkpoint mengevaluasi kebijakan yang PERSIS SAMA. Karena itu SD yang
    dilaporkan = SD dari rerata PER-CHECKPOINT (n=3, atau 5 utk `it0.7`), BUKAN SD
    dari 10 run mentah (yang akan meremehkan variansi antar-kebijakan).

    `greedy` tidak punya checkpoint (heuristik, tanpa pelatihan) -- 3 seed-nya adalah
    3 realisasi stokastik lingkungan yang independen. Unitnya = run itu sendiri.
    Keduanya berujung n=3, TAPI maknanya beda: RL = 3 kebijakan hasil latih berbeda,
    greedy = 3 realisasi lingkungan. Catat perbedaan ini saat menafsirkan SD.
"""
from __future__ import annotations

import json
import os


import numpy as np
import pandas as pd

DIR_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Metrik inti + arah "lebih baik". Dipakai untuk penyorotan & pemeriksaan tanda.
ARAH_BAIK = {
    "gini": "min", "gini_wait_pengguna": "min", "gini_trip_pengguna": "min",
    "wait": "min", "w_p50": "min", "w_p90": "min", "w_p95": "min",
    "w_frac_gt60": "min", "w_frac_gt120": "min",
    "served": "max", "acc": "max",
    "trust": "max", "trust_min": "max", "frac_trust_rendah": "min",
    "rec_entropy": "max", "entropi_spklu_pengguna": "max",
    "herding": "min", "flocking": "min",
    "jn_expost_frac_untung": "max", "jn_expost_mean": "max",
    "jnpatuh_expost_frac_untung": "max", "jnpatuh_expost_mean": "max",
    "pct_telat": "min", "pct_tepat": "max",
}
METRIK_INTI = ["gini", "acc", "wait", "trust", "rec_entropy", "herding"]


# --------------------------------------------------------------------------- muat
def _path_uji(tag: str, horizon: str = "90d") -> str:
    return os.path.join(DIR_OUT, f"uji_{tag}_metrik_{horizon}.json")


def muat(tag: str, horizon: str = "90d") -> dict:
    """Muat berkas hasil evaluasi. `tag` tanpa awalan 'uji_'/akhiran '_metrik_*'."""
    with open(_path_uji(tag, horizon), encoding="utf-8") as f:
        return json.load(f)


def daftar_lengan(d: dict) -> list:
    """Nama lengan (kunci `agregat`) di dalam satu berkas. Berkas RL biasanya 1;
    berkas greedy memuat 8 (2 heuristik x 2 mode x 2 rezim trust)."""
    return list(d["agregat"].keys())


# ------------------------------------------------- acuan greedy per kondisi (Blok E)
# Blok E semula membandingkan 4 sel PURE3 lintas 8 kondisi TANPA pembanding non-RL,
# sehingga "arsitektur ini lebih tahan" tak terpisah dari "kondisi ini memang lebih
# sulit bagi agen apa pun". Greedy tak perlu dilatih, jadi acuannya dapat dilengkapi
# lewat evaluasi ulang saja (`_uji_greedy_setara_metrik.py <seed> 90d 3 <kondisi>`).
#
# Kunci = akhiran TAG_ARM lengan RL; nilai = tag greedy yang setara lingkungannya.
GREEDY_KONDISI = {
    "":            "greedy_setara_k3",              # baku 4x
    "_it0.3":      "greedy_setara_k3_it0.3",
    "_it0.7":      "greedy_setara_k3_it0.7",
    "_gw0.0279513": "greedy_setara_k3_gw0.0279513",
    "_gw0.111805": "greedy_setara_k3_gw0.111805",
    # `_load6x` -> acuan 4x, BUKAN 6x. Lengan RL `_load6x` DILATIH pada 6x tetapi
    # DIEVALUASI pada 4x (lihat peringatan di bawah), jadi pembanding yang setara
    # lingkungannya adalah greedy 4x. Acuan 6x sejati ada di `greedy_setara_k3_load6x`
    # dan dilaporkan TERPISAH sbg ukuran beratnya rezim itu sendiri.
    "_load6x":     "greedy_setara_k3",
    # `_g0.95` / `_g0.999` adalah faktor diskon PPO/GAE -- parameter ALGORITMA, bukan
    # lingkungan. Greedy tak punya fungsi nilai sehingga tak berubah; acuannya SENGAJA
    # dipetakan ke `baku`, dan keidentikan itu sendiri yang perlu dinyatakan saat
    # melaporkan (perbedaan apa pun di sana murni milik agen RL).
    "_g0.95":      "greedy_setara_k3",
    "_g0.999":     "greedy_setara_k3",
}

# PERINGATAN yang WAJIB dibawa ke pelaporan: `eval_pure3_beban6x_metrik.sh` memanggil
# `_uji_master_pure_hybrid_ppo_metrik.py` dgn TAG=90d, dan baris 57 skrip itu mengunci
# `K.DS` ke dataset 4x. Berbeda dari `_it`/`_gw` yang diturunkan dari tag, `_load6x`
# TIDAK punya penurunan dataset. Terverifikasi lewat `served` (16.392 pada lengan 6x vs
# 25.106 permintaan yang ada di dataset 6x). Jadi lengan `_load6x` adalah uji TRANSFER
# (dilatih 6x, diuji 4x), BUKAN uji beban berat. Acuan greedy di bawah dijalankan pada
# dataset 6x yang sebenarnya, sehingga keduanya TIDAK sebanding secara langsung.
GREEDY_KONDISI_CATATAN = {
    "_load6x": "RL dilatih 6x tapi DIEVALUASI 4x -> ini uji TRANSFER; acuan greedy "
               "sengaja 4x agar setara. Acuan 6x sejati: greedy_setara_k3_load6x",
    "_g0.95":  "gamma = parameter algoritma; acuan greedy identik dgn baku",
    "_g0.999": "gamma = parameter algoritma; acuan greedy identik dgn baku",
}


def greedy_untuk(akhiran: str = "") -> str:
    """Tag greedy yang lingkungannya setara dgn kondisi `akhiran` (mis. `_it0.3`)."""
    if akhiran not in GREEDY_KONDISI:
        raise KeyError(f"kondisi {akhiran!r} tak dikenal. "
                       f"Pilihan: {sorted(GREEDY_KONDISI)}")
    return GREEDY_KONDISI[akhiran]


MODE_BAKU = "signed|dinamis"


def _pilih_lengan(d: dict, lengan: str = None) -> str:
    """Pilih kunci lengan.

    Sebagian berkas hasil memuat SATU lengan (dievaluasi mode `cepat` -> hanya
    `signed|dinamis`), sebagian memuat EMPAT (abs/signed x dinamis/beku). Bila
    `lengan` tak disebut dan ada lebih dari satu, `MODE_BAKU` dipilih otomatis --
    itu mode pelaporan baku di seluruh analisis ini, sehingga lengan ber-1 dan
    ber-4 mode dapat dicampur dalam satu tabel tanpa penanganan khusus di notebook.
    Ambiguitas yang TERSISA (mis. dua lengan berbeda dalam satu berkas, seperti
    `greedy_queue` vs `greedy_util`) tetap memunculkan galat -- di situ `lengan`
    memang harus disebut.
    """
    kunci = daftar_lengan(d)
    if lengan is None:
        if len(kunci) == 1:
            return kunci[0]
        cocok = [k for k in kunci if MODE_BAKU in k]
        if len(cocok) == 1:
            return cocok[0]
        raise ValueError(
            f"Berkas memuat {len(kunci)} lengan dan {len(cocok)} cocok dgn mode baku "
            f"{MODE_BAKU!r}; sebutkan `lengan=` secara eksplisit. Pilihan: {kunci}")
    if lengan in kunci:
        return lengan
    cocok = [k for k in kunci if lengan in k]
    if len(cocok) == 1:
        return cocok[0]
    raise KeyError(f"'{lengan}' tak cocok tunggal. Pilihan: {kunci}")


# ------------------------------------------------------------- statistik per-unit
def unit_stat(tag: str, metrik: str, horizon: str = "90d", lengan: str = None):
    """Rerata & SD pada UNIT ANALISIS YANG BENAR (RENCANA §2.1).

    Returns (mean, sd, nilai_per_unit, label_unit) dengan:
      - lengan RL  : unit = checkpoint; nilai_per_unit = rerata run per checkpoint
      - greedy/dll : unit = run; nilai_per_unit = nilai tiap run

    SD memakai ddof=0 (populasi dari unit yang ada), konsisten dgn pelaporan
    sebelumnya di sesi ini.
    """
    d = muat(tag, horizon)
    k = _pilih_lengan(d, lengan)
    ps = d["per_seed"][k]

    if "ckpt_per_seed" in d and d.get("n_checkpoint"):
        grup = {c: [] for c in range(d["n_checkpoint"])}
        for seed_str, c in d["ckpt_per_seed"].items():
            grup[c].append(ps[int(seed_str)][metrik])
        nilai = np.array([np.mean(v) for v in grup.values()], dtype=float)
        label = "checkpoint"
    else:
        nilai = np.array([r[metrik] for r in ps], dtype=float)
        label = "run"

    return float(np.mean(nilai)), float(np.std(nilai, ddof=0)), nilai, label


def tabel_banding(lengan_spec: list, metrik: list = None, horizon: str = "90d",
                  sertakan_unit: bool = False) -> pd.DataFrame:
    """Tabel perbandingan antar-lengan.

    `lengan_spec`: daftar (nama_tampil, tag) atau (nama_tampil, tag, lengan).
    """
    metrik = metrik or METRIK_INTI
    baris = []
    for spec in lengan_spec:
        nama, tag = spec[0], spec[1]
        lg = spec[2] if len(spec) > 2 else None
        r = {"lengan": nama}
        for m in metrik:
            mean, sd, nilai, label = unit_stat(tag, m, horizon, lg)
            r[m] = mean
            r[f"{m}_sd"] = sd
            if sertakan_unit:
                r[f"{m}_unit"] = np.round(nilai, 4).tolist()
        r["n_unit"] = len(nilai)
        r["unit"] = label
        baris.append(r)
    return pd.DataFrame(baris).set_index("lengan")


def sorot_terbaik(df: pd.DataFrame, metrik: list = None):
    """Styler: tebalkan nilai terbaik per metrik menurut ARAH_BAIK."""
    metrik = metrik or [c for c in df.columns if c in ARAH_BAIK]

    def _gaya(kol: pd.Series):
        if kol.name not in ARAH_BAIK:
            return [""] * len(kol)
        best = kol.min() if ARAH_BAIK[kol.name] == "min" else kol.max()
        return ["font-weight:bold;background-color:#e8f4e8" if v == best else ""
                for v in kol]

    return df.style.apply(_gaya, subset=metrik).format(precision=4)


# ------------------------------------------------------------------ ukuran efek
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d antar-dua himpunan unit (tak berpasangan, varians gabungan).
    n kecil (3-5) -- nilai ini INDIKATIF, bukan inferensi. Lihat RENCANA §2.2."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def banding_dua(tag_a: str, tag_b: str, metrik: list = None, horizon: str = "90d",
                lengan_a: str = None, lengan_b: str = None,
                nama_a: str = "A", nama_b: str = "B") -> pd.DataFrame:
    """Selisih + Cohen's d untuk dua lengan, per metrik.

    CATATAN (RENCANA §2.2): dengan n unit = 3-5, tidak ada uji signifikansi yang
    dapat mencapai alpha=0,05. Kolom `d` dilaporkan sebagai UKURAN EFEK deskriptif,
    dan `nilai_*` menampilkan seluruh unit apa adanya agar pembaca menilai sendiri.
    """
    metrik = metrik or METRIK_INTI
    baris = []
    for m in metrik:
        ma, sa, va, _ = unit_stat(tag_a, m, horizon, lengan_a)
        mb, sb, vb, _ = unit_stat(tag_b, m, horizon, lengan_b)
        arah = ARAH_BAIK.get(m, "?")
        selisih = ma - mb
        lebih_baik = (nama_a if ((selisih < 0) == (arah == "min")) else nama_b) \
            if arah in ("min", "max") else "?"
        baris.append({
            "metrik": m, "arah_baik": arah,
            nama_a: ma, f"{nama_a}_sd": sa,
            nama_b: mb, f"{nama_b}_sd": sb,
            "selisih(A-B)": selisih, "d": cohen_d(va, vb),
            "lebih_baik": lebih_baik,
            f"nilai_{nama_a}": np.round(va, 4).tolist(),
            f"nilai_{nama_b}": np.round(vb, 4).tolist(),
        })
    return pd.DataFrame(baris).set_index("metrik")


# --------------------------------------------------------------------- lintasan
def lintasan(tag: str, metrik: str, horizon: str = "90d", lengan: str = None,
             buang_warmup: int = 7) -> pd.DataFrame:
    """Seri harian satu metrik. Mengembalikan kolom [hari, nilai, sd].

    Entri hari -1 (kondisi awal, banyak NaN & trust=prior) SELALU dibuang.
    `buang_warmup` membuang N hari pertama (baku 7, sesuai fase warm-up di
    .docx Rancangan Simulasi) -- WAJIB sama untuk semua lengan yang dibandingkan.
    """
    d = muat(tag, horizon)
    k = _pilih_lengan(d, lengan)
    h = pd.DataFrame(d["harian"][k])
    h = h[h["hari"] >= buang_warmup]
    kol_sd = f"{metrik}_sd"
    out = h[["hari", metrik] + ([kol_sd] if kol_sd in h.columns else [])].copy()
    return out.rename(columns={metrik: "nilai", kol_sd: "sd"}).reset_index(drop=True)


def slope_dan_auc(tag: str, metrik: str, horizon: str = "90d", lengan: str = None,
                  buang_warmup: int = 7) -> dict:
    """Laju perubahan (regresi linear atas hari) + akumulasi (rerata & AUC).

    `slope` satuannya = perubahan metrik per HARI. Negatif pada `gini_harian`
    berarti ketimpangan membaik seiring waktu.
    """
    s = lintasan(tag, metrik, horizon, lengan, buang_warmup).dropna(subset=["nilai"])
    x, y = s["hari"].to_numpy(float), s["nilai"].to_numpy(float)
    if len(x) < 3:
        return {"slope": np.nan, "rerata": np.nan, "auc": np.nan, "n_hari": len(x)}
    slope, intersep = np.polyfit(x, y, 1)
    n = len(x)
    return {
        "slope": float(slope), "intersep": float(intersep),
        "rerata": float(y.mean()), "auc": float(np.trapezoid(y, x)),
        "awal_10h": float(y[:10].mean()), "akhir_10h": float(y[-10:].mean()),
        "slope_paruh1": float(np.polyfit(x[:n // 2], y[:n // 2], 1)[0]),
        "slope_paruh2": float(np.polyfit(x[n // 2:], y[n // 2:], 1)[0]),
        "n_hari": n,
    }


# ------------------------------------------------------------------ deteksi kolaps
# Pola kolaps (RENCANA §2.3): gini^ + wait^ + herding^ BERSAMAAN dgn rec_entropy_ + trust_
POLA_KOLAPS = {"gini": "naik", "wait": "naik", "herding": "naik",
               "rec_entropy": "turun", "trust": "turun"}


def deteksi_kolaps(tag: str, horizon: str = "90d", lengan: str = None,
                   ambang_rasio: float = 1.3, min_cocok: int = 4) -> pd.DataFrame:
    """Periksa kolaps per-checkpoint dgn ko-pergerakan metrik (RENCANA §2.3).

    KRITERIA = RASIO terhadap checkpoint TERBAIK, bukan skor-z. Alasan: dengan n=3
    checkpoint, checkpoint terburuk SELALU berjarak ~1,15 SD dari rerata secara
    mekanis, sehingga ambang berbasis z menandai apa pun -- termasuk selisih 4% yang
    tak berarti apa-apa. Kolaps sesungguhnya berukuran BESAR (teramati gini 2,4x
    lipat checkpoint terbaik), jadi rasio yang membedakannya, bukan posisi relatif.

    Ditandai `kolaps` bila >= `min_cocok` dari 5 metrik POLA_KOLAPS memburuk
    >= `ambang_rasio` kali checkpoint terbaik (untuk metrik "turun":
    <= 1/ambang_rasio kali). Tidak menuntut kelimanya karena `herding` terbatas pada
    [0,1] sehingga rentang dinamisnya sempit -- pada kolaps terkonfirmasi ia hanya
    mencapai ~1,2x, sementara `gini`/`wait` melonjak 2,4x/7,1x. Menuntut 5/5 akan
    melewatkan kolaps yang jelas-jelas nyata. Tabel ini alat bantu penyaring;
    keputusan akhir tetap dibaca manual lewat kolom nilai & rasionya.

    Checkpoint kolaps TIDAK dibuang dari analisis -- ia bagian dari hasil (Blok F).
    """
    d = muat(tag, horizon)
    k = _pilih_lengan(d, lengan)
    ps = d["per_seed"][k]

    if "ckpt_per_seed" not in d or not d.get("n_checkpoint"):
        return pd.DataFrame()   # greedy dll: tak ada checkpoint

    grup = {c: [] for c in range(d["n_checkpoint"])}
    for seed_str, c in d["ckpt_per_seed"].items():
        grup[c].append(int(seed_str))

    baris = []
    for c, seeds in grup.items():
        r = {"checkpoint": c, "n_run": len(seeds), "seeds": seeds}
        for m in POLA_KOLAPS:
            r[m] = float(np.mean([ps[s][m] for s in seeds]))
        baris.append(r)
    df = pd.DataFrame(baris).set_index("checkpoint")

    tanda = pd.DataFrame(index=df.index)
    for m, arah in POLA_KOLAPS.items():
        if arah == "naik":                       # makin besar makin buruk
            terbaik = df[m].min()
            rasio = df[m] / terbaik if terbaik > 0 else pd.Series(1.0, index=df.index)
            tanda[m] = rasio >= ambang_rasio
        else:                                    # makin kecil makin buruk
            terbaik = df[m].max()
            rasio = df[m] / terbaik if terbaik > 0 else pd.Series(1.0, index=df.index)
            tanda[m] = rasio <= 1.0 / ambang_rasio
        df[f"{m}_rasio"] = rasio.round(3)
    df["n_pola_cocok"] = tanda.sum(axis=1)
    df["kolaps"] = df["n_pola_cocok"] >= min_cocok
    return df


# ------------------------------------------------------------------ riwayat latih
def _path_latih(tag: str) -> str:
    return os.path.join(DIR_OUT, f"{tag}_training_results.json")


def riwayat_latih(tag: str) -> list:
    """Muat riwayat pelatihan: daftar per seed, tiap entri punya `history`
    (300 chunk) berisi beta/ret_mean/entropy/loss/grad_norm/n_backlog dll."""
    with open(_path_latih(tag), encoding="utf-8") as f:
        return json.load(f)


def kurva_latih(tag: str, kunci: str) -> pd.DataFrame:
    """DataFrame [iter x seed] untuk satu kunci skalar (`entropy`, `grad_norm`, ...)."""
    data = riwayat_latih(tag)
    return pd.DataFrame(
        {f"seed{e['seed']}": [c[kunci] for c in e["history"]] for e in data},
        index=[c["iter"] for c in data[0]["history"]]).rename_axis("iter")


def kurva_vektor(tag: str, kunci: str = "beta") -> dict:
    """Untuk kunci bernilai VEKTOR (`beta`, `ret_mean`): {seed: DataFrame[iter x aliran]}."""
    data = riwayat_latih(tag)
    out = {}
    for e in data:
        arr = np.array([c[kunci] for c in e["history"]], dtype=float)
        nama = (["wait", "gini", "accept"] if arr.shape[1] == 3
                else [f"aliran{i}" for i in range(arr.shape[1])])
        out[e["seed"]] = pd.DataFrame(
            arr, columns=nama, index=[c["iter"] for c in e["history"]]).rename_axis("iter")
    return out


def beta_akhir(tag: str, n_chunk: int = 50) -> pd.DataFrame:
    """Rerata bobot DGR pada `n_chunk` TERAKHIR, per seed.

    PENTING: pakai rerata jendela, JANGAN chunk terakhir tunggal -- chunk tunggal
    dapat menjadi pencilan ekstrem (teramati [0,997 0,002 0,002] pada chunk-299
    padahal rerata 50 chunk terakhir [0,80 0,17 0,03]).
    """
    baris = []
    for seed, df in kurva_vektor(tag, "beta").items():
        r = {"seed": seed}
        r.update(df.iloc[-n_chunk:].mean().to_dict())
        baris.append(r)
    return pd.DataFrame(baris).set_index("seed")


# ----------------------------------------------------------------- distribusi
# Dua jenis "distribusi" yang TIDAK boleh tertukar saat melaporkan:
#   (a) ANTAR-UNIT  -- sebaran nilai ringkas antar checkpoint/run independen.
#                      Menjawab "seberapa stabil hasilnya bila diulang?"
#   (b) DALAM-RUN   -- sebaran antar pengguna/trip DI DALAM satu run, sudah
#                      terhitung sebagai persentil (`w_p10`..`w_p99`, `tr_p*`,
#                      `galat_p*`). Menjawab "apakah beban ditanggung merata?"
# Rerata yang sama bisa menyembunyikan (b) yang sangat berbeda -- mis. wait rerata
# sama tetapi p90 jauh berbeda berarti ekor pengalaman buruk yang berbeda.

def sebaran_antar_unit(lengan_spec: list, metrik: list = None,
                       horizon: str = "90d") -> pd.DataFrame:
    """Statistik sebaran ANTAR-UNIT (checkpoint utk RL, run utk greedy).

    `cv` = koefisien variasi (sd/|mean|) -- membandingkan kestabilan antar metrik
    yang satuannya berbeda. `rentang_rel` = (maks-min)/|mean|.
    """
    metrik = metrik or METRIK_INTI
    baris = []
    for spec in lengan_spec:
        nama, tag = spec[0], spec[1]
        lg = spec[2] if len(spec) > 2 else None
        for m in metrik:
            mean, sd, v, label = unit_stat(tag, m, horizon, lg)
            baris.append({
                "lengan": nama, "metrik": m, "unit": label, "n": len(v),
                "mean": mean, "sd": sd,
                "min": float(np.min(v)), "median": float(np.median(v)),
                "maks": float(np.max(v)),
                "cv": (sd / abs(mean)) if mean != 0 else np.nan,
                "rentang_rel": ((np.max(v) - np.min(v)) / abs(mean)) if mean != 0 else np.nan,
                "nilai": np.round(v, 4).tolist(),
            })
    return pd.DataFrame(baris).set_index(["metrik", "lengan"])


def nilai_unit(tag: str, metrik: str, horizon: str = "90d", lengan: str = None):
    """Vektor nilai per-unit (checkpoint/run) -- bahan plot sebaran."""
    return unit_stat(tag, metrik, horizon, lengan)[2]


def profil_dalam_run(lengan_spec: list, dasar: str = "w", horizon: str = "90d") -> pd.DataFrame:
    """Persentil DALAM-RUN, dirata-ratakan antar unit.

    `dasar`: 'w' (wait semua trip) | 'wpatuh' | 'wtolak' | 'tr' (trust antar pengguna)
             | 'galat' (galat janji). Persentil yang tersedia: p10..p99 (lihat JSON).
    """
    pers = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "maks"]
    baris = []
    for spec in lengan_spec:
        nama, tag = spec[0], spec[1]
        lg = spec[2] if len(spec) > 2 else None
        r = {"lengan": nama}
        for p in pers:
            kunci = f"{dasar}_{p}"
            try:
                r[p] = unit_stat(tag, kunci, horizon, lg)[0]
            except (KeyError, TypeError):
                r[p] = np.nan
        baris.append(r)
    return pd.DataFrame(baris).set_index("lengan")


# --------------------------------------------------- KOREKSI: Gini utilisasi
def _gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# Metrik `gini` yang tersimpan di berkas hasil dihitung dari `total_served` (CACAH EV
# yang dilayani) -- lihat `_uji_konsolidasi.py:122`. Tujuan Kinerja 1 (III.2.3) menuntut
# ketimpangan UTILISASI, dan reward pelatihan memang memakai utilisasi
# (`rollout.py:607`, `s.get_utilization()`). Jadi yang keliru HANYA metrik evaluasi.
#
# Perbaikannya tidak memerlukan run ulang: `_stasiun` sudah menyimpan `util_mean`
# (rerata waktu okupansi konektor) dan `queue_mean` per stasiun untuk tiap run.
#
# CATATAN: `util_mean` adalah rerata-waktu dari cuplikan per jam (`station_log`),
# sedangkan reward memakai utilisasi SESAAT tiap langkah. Keduanya besaran yang sama,
# beda resolusi -- sah sebagai padanan saat evaluasi, dan wajib disebut demikian.
KUNCI_STASIUN = {"gini_utilisasi": "util_mean", "gini_antrean": "queue_mean"}


def gini_stasiun(tag: str, jenis: str = "gini_utilisasi", horizon: str = "90d",
                 lengan: str = None):
    """Gini antar-stasiun dari `_stasiun`, per unit analisis.

    `jenis`: 'gini_utilisasi' (okupansi konektor -- SESUAI Tujuan Kinerja 1) atau
             'gini_antrean' (panjang antrean).

    Returns (mean, sd, nilai_per_unit, label_unit) -- antarmuka sama `unit_stat`.
    """
    if jenis not in KUNCI_STASIUN:
        raise KeyError(f"jenis={jenis!r} tak dikenal; pilih {list(KUNCI_STASIUN)}")
    kunci = KUNCI_STASIUN[jenis]
    d = muat(tag, horizon)
    k = _pilih_lengan(d, lengan)
    ps = d["per_seed"][k]
    per_run = np.array([_gini([v[kunci] for v in r["_stasiun"].values()]) for r in ps])

    if "ckpt_per_seed" in d and d.get("n_checkpoint"):
        grup = {c: [] for c in range(d["n_checkpoint"])}
        for seed_str, c in d["ckpt_per_seed"].items():
            grup[c].append(per_run[int(seed_str)])
        nilai = np.array([np.mean(v) for v in grup.values()])
        label = "checkpoint"
    else:
        nilai = per_run
        label = "run"
    return float(np.mean(nilai)), float(np.std(nilai, ddof=0)), nilai, label
