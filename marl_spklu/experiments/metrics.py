"""Ekstraktor KPI untuk eksperimen skenario (S0/S1/S2/...).

Semua fungsi bersifat murni: menerima array / log / daftar aktor, mengembalikan
skalar atau dict. Sumber data:
  - deret Gini    : HistoryBuffer.util_history (kronologis bila window_size >= max_steps)
  - herding/entropi: Simulator.rec_distribution_log
  - acceptance    : User.compliance_history + User.trust
  - trust         : User.trust
  - waktu tunggu  : Simulator.logs
Memetakan ke KPI Rancangan Tahap 2.2 (Kelas 1 Level, Kelas 2 Dinamika, Kelas 3 Koordinasi).
"""
import numpy as np


# ----------------------------- Gini & tren (Kelas 1 & 2) -----------------------------

def gini(util_array) -> float:
    """Koefisien Gini untuk array 1D non-negatif."""
    a = np.clip(np.asarray(util_array, dtype=float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a)
    n = a.shape[0]
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))


def gini_series_from_history(history) -> np.ndarray:
    """Deret Gini per-step (kronologis). Mengasumsikan window_size >= jumlah step
    (tanpa wraparound) -- pola yang dipakai runner (window_size_15m = max_steps)."""
    n = history.current_step
    if n <= 0:
        return np.zeros(0)
    hist = history.util_history[:n, :]
    return np.array([gini(row) for row in hist])


def slope(series) -> float:
    """Slope regresi linear sederhana dari sebuah deret."""
    y = np.asarray(series, dtype=float)
    n = y.shape[0]
    if n < 2:
        return 0.0
    x = np.arange(n)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


def auc_gini(series) -> float:
    """Area di bawah kurva Gini (biaya ketimpangan kumulatif), dinormalisasi per-step
    -> setara rata-rata Gini sepanjang horizon. Independen dari panjang horizon."""
    y = np.asarray(series, dtype=float)
    if y.shape[0] == 0:
        return 0.0
    return float(np.trapz(y) / (y.shape[0] - 1)) if y.shape[0] > 1 else float(y[0])


def gini_final_smoothed(series, steps_per_day: int = 96) -> float:
    """Gini akhir sbg RATA-RATA hari terakhir (bukan satu snapshot langkah terakhir).
    `gini_final` (snapshot tunggal) sangat noisy di dataset ini: menjelang akhir horizon,
    jadwal kedatangan pengguna berhenti (tak ada spawn baru mengisi slot kosong) shg
    jumlah EV yg SEDANG charging pd langkah persis terakhir bisa tinggal 1-2 dari 8 SPKLU
    -- Gini dari sampel sekecil itu berayun liar (bisa 0.3 di satu langkah, 0.87 di
    langkah berikutnya) tanpa mencerminkan pemerataan jangka panjang. Rata-rata 1 hari
    terakhir (96 langkah @15 menit) meredam derau snapshot-tunggal ini sambil tetap
    representasi kondisi "akhir" (bukan rata-rata SELURUH horizon spt gini_mean)."""
    y = np.asarray(series, dtype=float)
    if y.shape[0] == 0:
        return 0.0
    window = min(steps_per_day, y.shape[0])
    return float(y[-window:].mean())


def _daily_mean(series, steps_per_day: int = 96) -> np.ndarray:
    """Rata-rata Gini per hari dari deret per-step."""
    y = np.asarray(series, dtype=float)
    n_days = y.shape[0] // steps_per_day
    if n_days == 0:
        return np.array([y.mean()]) if y.shape[0] else np.zeros(0)
    return y[:n_days * steps_per_day].reshape(n_days, steps_per_day).mean(axis=1)


def time_to_negative_slope(series, steps_per_day: int = 96,
                           window_days: int = 7, sustained_days: int = 7):
    """Hari pertama saat slope Gini (atas jendela `window_days` harian) menjadi negatif
    dan BERTAHAN negatif >= `sustained_days` berturut-turut. Mengembalikan indeks hari
    (int) atau None bila tak pernah tercapai. Operasionalisasi metrik
    'time-to-negative-slope' (Rancangan Kelas 2)."""
    daily = _daily_mean(series, steps_per_day)
    D = daily.shape[0]
    if D < window_days + 1:
        return None
    neg = np.zeros(D, dtype=bool)
    for d in range(window_days, D):
        neg[d] = slope(daily[d - window_days:d + 1]) < 0
    run = 0
    for d in range(D):
        run = run + 1 if neg[d] else 0
        if run >= sustained_days:
            return int(d - sustained_days + 1)
    return None


# ----------------------------- Koordinasi (Kelas 3) -----------------------------

def herding_index(rec_distribution_log, threshold: int = 3) -> float:
    """Herding Index (Rancangan Tahap 2.2 -- DOKUMEN DESAIN AWAL, BUKAN definisi final
    di draf tesis): fraksi window-rekomendasi di mana ADA >=threshold EV direkomendasikan
    ke SPKLU yang sama. Ternormalisasi (0..1), bukan hitungan mentah.

    CATATAN AUDIT (lihat rencana_pengujian_demonstrasi_evaluasi.md Bagian 4): metrik ini
    TIDAK SAMA dengan "Flocking Index" yang didefinisikan draf tesis Bab IV.2 (para. 686-
    698) sbg kriteria keberhasilan Objektif 1 (<=0,20). Untuk kriteria itu pakai
    flocking_index() di bawah, bukan fungsi ini. Fungsi ini dipertahankan sbg metrik
    diagnostik terpisah (herding_events/herding_index sudah dipakai di laporan-laporan
    sebelumnya) -- bukan dihapus, hanya diberi disambiguasi eksplisit."""
    windows = [w for w in rec_distribution_log if w.get("n_recs", 0) > 0]
    if not windows:
        return 0.0
    herd = sum(1 for w in windows if w["counts"] and max(w["counts"].values()) >= threshold)
    return herd / len(windows)


def flocking_index(rec_distribution_log) -> float:
    """Flocking Index sesuai definisi draf tesis Bab IV.2 (para. 686-698, kriteria
    keberhasilan Objektif 1 III.2.2: <=0,20): "rasio antara peningkatan beban puncak
    stasiun target dan jumlah rekomendasi yang diberikan pada stasiun tersebut dalam
    window 15 menit".

    Dihitung per (window, stasiun-target) sbg queue_delta[sid] / n_recs_ke_sid, lalu
    dirata-ratakan di seluruh pasangan (window, stasiun) yang menerima >=1 rekomendasi
    sepanjang horizon evaluasi. queue_delta = peningkatan antrean stasiun itu (>=0,
    diclip) dari awal ke akhir window yang sama -- direkam Simulator.step_once() ke
    field 'queue_delta' pada rec_distribution_log.

    PERINGATAN SATUAN: hasil TIDAK otomatis berada di [0,1] -- satuannya "peningkatan
    panjang-antrean per rekomendasi". Draf tesis tidak menspesifikasi skema normalisasi
    utk ambang 0,20; jangan bandingkan nilai ini terhadap ambang tsb tanpa memverifikasi
    dulu apakah ambang 0,20 dimaksudkan pada skala mentah ini atau versi ternormalisasi.
    Entri lama (sebelum field 'queue_delta' ditambahkan) diabaikan otomatis."""
    ratios = []
    for w in rec_distribution_log:
        counts = w.get("counts") or {}
        deltas = w.get("queue_delta")
        if not counts or deltas is None:
            continue
        for sid, n_recs in counts.items():
            if n_recs > 0 and sid in deltas:
                ratios.append(deltas[sid] / n_recs)
    return float(np.mean(ratios)) if ratios else 0.0


def recommendation_entropy(rec_distribution_log, n_stations: int = None) -> float:
    """Rata-rata entropi Shannon (base-2) distribusi rekomendasi per-window, di atas
    window dengan >=1 rekomendasi. Bila n_stations diberikan, dinormalisasi ke [0,1]
    (dibagi log2(n_stations)) -> entropi tinggi = rekomendasi beragam (anti-herding)."""
    windows = [w for w in rec_distribution_log if w.get("n_recs", 0) > 0]
    if not windows:
        return 0.0
    ents = []
    for w in windows:
        counts = np.array(list(w["counts"].values()), dtype=float)
        p = counts / counts.sum()
        ents.append(float(-np.sum(p * np.log2(p))))
    ent = float(np.mean(ents))
    if n_stations and n_stations > 1:
        ent /= np.log2(n_stations)
    return ent


# ----------------------------- Acceptance & trust (Kelas 2) -----------------------------

def overall_acceptance_rate(users) -> float:
    all_c = [c for u in users for c in u.compliance_history]
    return float(np.mean(all_c)) if all_c else 0.0


def acceptance_rate_by_tercile(users) -> dict:
    """Acceptance rate dikelompokkan per tersil trust T (rendah/sedang/tinggi). Model lama
    memakai w_i (kerentanan per-segmen, sudah dihapus); di model baru T = alpha/(alpha+beta)
    adalah SATU-SATUNYA bobot yang mengontrol seberapa besar P_rec memengaruhi pilihan
    (P = (1-T)*P_pref + T*P_rec, Model_Simulasi_Inti.md §3.2) -- pengganti langsung w_i."""
    act = [u for u in users if u.compliance_history]
    if not act:
        return {"low": None, "mid": None, "high": None}
    ts = np.array([u.trust for u in act])
    q1, q2 = np.quantile(ts, [1 / 3, 2 / 3])
    buckets = {"low": [], "mid": [], "high": []}
    for u in act:
        key = "low" if u.trust <= q1 else ("mid" if u.trust <= q2 else "high")
        buckets[key].extend(u.compliance_history)
    return {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}


def trust_stats(users) -> dict:
    """Distribusi trust populasi (KPI 'trust score' -- kini hidup setelah loop trust)."""
    t = np.array([u.trust for u in users], dtype=float)
    if t.size == 0:
        return {}
    return {
        "mean": float(t.mean()),
        "p10": float(np.quantile(t, 0.10)),
        "p50": float(np.quantile(t, 0.50)),
        "p90": float(np.quantile(t, 0.90)),
        "frac_below_0.5": float(np.mean(t < 0.5)),
        "n_moved_from_init": int(np.sum(np.abs(t - 0.5) > 1e-9)),
    }


# ----------------------------- Level (Kelas 1) -----------------------------

def wait_stats(logs) -> dict:
    """Statistik waktu tunggu dari Simulator.logs (per sesi selesai)."""
    w = np.array([L["wait_time"] for L in logs], dtype=float)
    if w.size == 0:
        return {"mean": 0.0, "p90": 0.0, "n": 0}
    return {"mean": float(w.mean()), "p90": float(np.quantile(w, 0.90)), "n": int(w.size)}


def wait_stats_by_compliance(logs) -> dict:
    """Waktu tunggu dipecah per grup patuh/default -- basis kriteria keberhasilan
    Objektif 2 (III.2.2, para. 484): "Waktu tunggu rata-rata pengguna yang mengikuti
    rekomendasi tidak lebih buruk dari 20% dibandingkan waktu tunggu pengguna yang
    mempertahankan pilihan bawaannya". Sumber: Simulator.logs, field 'complied' yang
    sudah direkam per sesi selesai (simulator.py, on_charge_complete) -- tidak perlu
    instrumentasi baru, data sudah tersedia sejak awal, hanya belum diagregasi terpisah.

    Return: {"complied": wait_stats(...), "default": wait_stats(...), "ratio": ...,
    "criterion_20pct_ok": bool|None}. ratio = mean(complied)/mean(default); None bila
    salah satu grup kosong. criterion_20pct_ok None bila ratio None, else ratio<=1.2."""
    complied_logs = [L for L in logs if L.get("complied")]
    default_logs = [L for L in logs if not L.get("complied")]
    complied = wait_stats(complied_logs)
    default = wait_stats(default_logs)
    ratio = None
    if complied["n"] > 0 and default["n"] > 0 and default["mean"] > 0:
        ratio = complied["mean"] / default["mean"]
    return {
        "complied": complied, "default": default, "ratio": ratio,
        "criterion_20pct_ok": (ratio <= 1.2) if ratio is not None else None,
    }


def served_stats(spklus) -> dict:
    """Cakupan & ketimpangan layanan akhir antar-SPKLU."""
    served = np.array([s.total_served for s in spklus.values()], dtype=float)
    n = served.size
    active = int(np.sum(served > 0))
    return {
        "total_served": int(served.sum()),
        "n_spklu": n,
        "n_active": active,
        "frac_inactive": float(1 - active / n) if n else 0.0,
        "gini_served": gini(served),
    }
