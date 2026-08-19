"""Observasi per-stasiun SETIA §3.1 MASTER (Zhang dkk., WWW 2021) -- dipakai HANYA oleh
`master_ddpg_policy.py`/`master_ddpg_trainer.py`. TIDAK dipakai lengan lain (mereka
sengaja tetap memakai observasi kaya-fitur `rollout.py`, lihat audit 2026-08-19 #15b).

MASTER §3.1: state stasiun i pada waktu t, s^i_t, memuat (paper Tabel 1):
    - indeks stasiun i
    - waktu t
    - jumlah slot tersedia
    - permintaan mendatang (perkiraan EV yang akan datang)
    - daya (kW)
    - ETA (waktu sampai slot berikutnya kosong)
    - CP (charging price)

SENGAJA TIDAK memuat: SoC pemohon, koordinat pemohon, kapasitas baterai pemohon --
identitas/atribut pemohon HANYA masuk secara implisit lewat ETA (station tak "melihat"
siapa yang bertanya, MASTER Pers. 11: a^i_t = b^i(o^i_t), o^i_t TIDAK memuat fitur
pemohon). Ini beda struktural dgn `RLRolloutAgent._build_obs` (rollout.py), yang memang
mendeskripsikan agen PERMINTAAN (Peran Agen: Permintaan EV, lihat memory audit), bukan
agen STASIUN.

CP (charging price) TIDAK DIMODELKAN di simulator ini (tak ada data harga per-SPKLU/
per-sesi) -- diisi KONSTANTA 0.0, didokumentasikan eksplisit sbg deviasi dari paper
(bukan disembunyikan/dihapus diam-diam dari vektor fitur, supaya dimensi & urutan fitur
tetap PERSIS mengikuti §3.1 walau satu suku selalu nol)."""
import math

import numpy as np

# Urutan & jumlah fitur PERSIS §3.1 (7 suku per stasiun).
STATION_FEAT_DIM_MASTER = 7
FEATURE_NAMES_MASTER = (
    "index_norm", "time_sin", "time_cos", "slots_avail_norm",
    "upcoming_demand_norm", "power_norm", "eta_norm",
)
# CP dilaporkan TERPISAH (lihat docstring modul) -- ditambahkan sbg suku ke-8 konstan.
CP_PLACEHOLDER = 0.0

# Skala normalisasi (agar nilai ~O(1), konsisten dgn konvensi rollout.py).
_ETA_SCALE = 60.0          # menit
_DEMAND_SCALE = 5.0        # jumlah EV pending/rec_activity
_POWER_SCALE = 150.0       # kW, batas atas realistis (Tier1: rentang 16,3-152,9 kW)


def build_station_obs(spklu, sid_idx: int, n_spklu: int, time_now_min: float,
                      sim) -> np.ndarray:
    """o^i_t (§3.1) utk SATU stasiun -- HANYA memakai info milik stasiun itu sendiri
    (kapasitas/antrean/charging/pending_arrivals -- lihat SPKLU.pending_arrivals,
    Validasi_Generik/LAPORAN_VALIDASI.md T2) + waktu global. TIDAK menerima `user`
    sama sekali (beda tanda tangan dgn `rollout.py::_station_feat`) -- desentralisasi
    yg SESUNGGUHNYA, bukan cuma broadcast konteks yg sama ke semua stasiun.

    `sim`: dipakai HANYA utk `compute_virtual_wait(None, spklu, time_now)` (ETA
    stasiun ybs -- formula murni lokal: antrean + EV charging + EV pending_arrivals
    milik stasiun ITU SENDIRI, TIDAK menengok stasiun lain atau pemohon manapun)."""
    cap_total = sum(spklu.capacities.values())
    charging_total = sum(len(c) for c in spklu.charging.values())
    slots_avail = max(0, cap_total - charging_total)
    slots_avail_norm = (slots_avail / cap_total) if cap_total > 0 else 0.0

    # Permintaan mendatang: EV yang SEDANG MENUJU stasiun ini (pending_arrivals, count
    # per tipe konektor) + aktivitas rekomendasi 24 jam (sim.recent_recs) -- keduanya
    # sinyal LOKAL milik stasiun (tak butuh melihat kondisi stasiun lain).
    n_pending = sum(spklu.pending_arrivals.values()) if spklu.pending_arrivals else 0
    n_recent_rec = sim.recent_recs.get(spklu.spklu_id, 0) if sim is not None else 0
    upcoming_demand_norm = (n_pending + n_recent_rec) / _DEMAND_SCALE

    power_kw = spklu.daya_efektif_dc if spklu.daya_efektif_dc > 0 else 7.0  # fallback AC nominal
    power_norm = min(power_kw / _POWER_SCALE, 1.0)

    eta = sim.compute_virtual_wait(None, spklu, time_now_min) if sim is not None else 0.0
    eta_norm = min(eta / _ETA_SCALE, 5.0)   # dibatasi (bukan dipotong keras) -- ekor jarang

    hour = (time_now_min / 60.0) % 24.0
    sin_h, cos_h = math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)
    index_norm = sid_idx / max(1, n_spklu - 1)

    return np.array([
        index_norm, sin_h, cos_h, slots_avail_norm, upcoming_demand_norm,
        power_norm, eta_norm,
    ], dtype=np.float32)


def build_joint_obs_master(sim, sids: list, time_now_min: float) -> np.ndarray:
    """(N, STATION_FEAT_DIM_MASTER) -- satu baris §3.1 per stasiun, dipanggil sekali
    per keputusan (dipakai aktor DESENTRALISASI baris-demi-baris, dan kritik terpusat
    lintas-baris via attention -- lihat master_ddpg_policy.py)."""
    n = len(sids)
    return np.stack([
        build_station_obs(sim.spklus[sid], i, n, time_now_min, sim)
        for i, sid in enumerate(sids)
    ]).astype(np.float32)


def snapshot_slots_avail(sim) -> dict:
    """{sid: slots_avail_norm} SAAT INI utk SELURUH stasiun -- dipakai trainer
    (`master_ddpg_trainer.py::_SlotAvailLog`) membangun fitur kritik-saja `future_avail`
    (Delayed Access Strategy, MASTER §4.3): nilai BENAR di t+d, hanya boleh diakses
    kritik SETELAH simulasi benar2 mencapai waktu t+d (bukan diprediksi/diperkirakan)."""
    out = {}
    for sid, s in sim.spklus.items():
        cap_total = sum(s.capacities.values())
        charging_total = sum(len(c) for c in s.charging.values())
        slots_avail = max(0, cap_total - charging_total)
        out[sid] = (slots_avail / cap_total) if cap_total > 0 else 0.0
    return out
