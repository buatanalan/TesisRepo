"""Wait-time forecaster (NON-RL) — memasok lookahead ŵ ke observasi EV Agent.
Selaras Spesifikasi_RL_EV_LoadBalancing.md §4 & §9 (mode A/B/C/D).

Prediksi wait = regresi supervised `state → ŵ`, BUKAN keputusan sekuensial → tidak ada
MDP/policy/reward. Dua implementasi:

  * FormulaForecaster  (Mode A) — bungkus SPKLU.estimate_wait_time (formula antrean
    tervalidasi). Tanpa pelatihan. Default paling sederhana.
  * LearnedForecaster  (Mode B/C/D) — regresor sklearn atas fitur station-wise + waktu.
    Mendukung fit() batch (offline, mode B/C) dan partial_fit() online (mode D).

Target training (`collect_forecast_dataset`): wait pada langkah BERIKUTNYA (t+1) —
menjadikan forecaster mempelajari lookahead temporal (pola jam/tren) yang MELAMPAUI
estimasi instan formula. Ground-truth diambil dari formula antrean simulator pada t+1
(tersedia penuh tanpa atribusi per-user); bila kelak realized-wait per-sesi di-log,
target dapat diganti ke sana tanpa mengubah antarmuka.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Urutan fitur gabungan: 5 kelompok fitur (19 dimensi total)
N_FEATURES = 19


def extract_features(spklu, time_now_min: float, user=None, soc: float = 50.0, recent_recs_count: int = 0, dist_override: float = None) -> np.ndarray:
    """Vektor fitur gabungan (user + SPKLU + temporal + aktivitas) pada waktu tertentu."""
    # Kelompok 1 — Karakteristik statis stasiun
    n_connectors = float(sum(spklu.capacities.values()))
    cap_AC = float(spklu.capacities.get("AC", 0))
    cap_DC = float(spklu.capacities.get("DC", 0))

    # Kelompok 2 — Fitur pengguna/kendaraan
    soc_norm = float(soc) / 100.0
    
    if dist_override is not None:
        dist = float(dist_override)
    elif user is not None and user.location is not None:
        dist = float(math.dist(user.location, spklu.location))
    else:
        dist = 0.0

    if user is not None and getattr(user, "connector_types", None) is not None:
        compat_AC = 1.0 if "AC" in user.connector_types else 0.0
        compat_DC = 1.0 if "DC" in user.connector_types else 0.0
    else:
        compat_AC = 1.0
        compat_DC = 1.0
        
    battery_capacity = 1.0  # Kapasitas baterai (karena tidak dimodelkan/tersedia)

    # Kelompok 3 — Fitur antrean saat ini
    queue_len = float(spklu.get_queue_length())
    charging_len = float(sum(len(c) for c in spklu.charging.values()))
    queue_AC = float(len(spklu.queues.get("AC", [])))
    queue_DC = float(len(spklu.queues.get("DC", [])))
    charging_AC = float(len(spklu.charging.get("AC", [])))
    charging_DC = float(len(spklu.charging.get("DC", [])))

    # Kelompok 4 — Fitur aktivitas sistem (jumlah rekomendasi dalam jendela terkini)
    recs_count = float(recent_recs_count)

    # Kelompok 5 — Fitur temporal
    hour = (time_now_min / 60.0) % 24.0
    dow = (time_now_min / (60.0 * 24.0)) % 7.0
    sin_hour = math.sin(2 * math.pi * hour / 24.0)
    cos_hour = math.cos(2 * math.pi * hour / 24.0)
    sin_dow = math.sin(2 * math.pi * dow / 7.0)
    cos_dow = math.cos(2 * math.pi * dow / 7.0)

    return np.array([
        n_connectors, cap_AC, cap_DC,
        soc_norm, dist, compat_AC, compat_DC, battery_capacity,
        queue_len, charging_len, queue_AC, queue_DC, charging_AC, charging_DC,
        recs_count,
        sin_hour, cos_hour, sin_dow, cos_dow
    ], dtype=float)


class ForecasterBase:
    """Antarmuka: predict(spklus, time_now) -> {sid: expected_wait_menit}."""

    def predict(self, spklus: dict, time_now_min: float = 0.0, user=None, soc: float = 50.0, sim=None) -> dict:
        raise NotImplementedError

    # Non-learning secara default (Mode A). Learned override di bawah.
    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        return self


class FormulaForecaster(ForecasterBase):
    """Mode A — ŵ = min estimate_wait_time antar tipe konektor. Tanpa pelatihan.

    KASAR SECARA SENGAJA: `SPKLU.estimate_wait_time` memakai rata-rata waktu-charge
    TETAP per tipe konektor (AC~136mnt/DC~49mnt), BUKAN `remaining_time` SESUNGGUHNYA
    EV yang sedang charging. Ini estimasi yang DITAMPILKAN ke pengguna & dipakai
    `User.update_trust` -- BERBEDA dari `eta_norm` yang dilihat AKTOR di observasi
    (`master_paper_obs.py::build_station_obs`, via `sim.compute_virtual_wait`, yang
    JAUH lebih rinci/akurat). Lihat `VirtualWaitForecaster` di bawah utk basis yang
    disamakan -- diagnosis event-trust menemukan bias sistematis besar (-70an menit)
    persis di celah dua estimator ini (agen "tahu" antrean akan cepat kosong via
    compute_virtual_wait, tapi forecaster kasar tetap menjanjikan wait rata-rata
    lama, membuat janji yang ditampilkan ke pengguna jauh meleset -> trust tererosi)."""

    def predict(self, spklus: dict, time_now_min: float = 0.0, user=None, soc: float = 50.0, sim=None) -> dict:
        est = {}
        for sid, s in spklus.items():
            waits = [s.estimate_wait_time(c) for c, cap in s.capacities.items() if cap > 0]
            est[sid] = float(min(waits)) if waits else 0.0
        return est


class VirtualWaitForecaster(ForecasterBase):
    """Basis estimasi DISAMAKAN dengan yang dilihat aktor (`eta_norm`, §3.1): memakai
    `sim.compute_virtual_wait` (simulasi FIFO dgn `remaining_time` SESUNGGUHNYA tiap EV
    yang sedang charging + EV yang sedang menuju stasiun) UNTUK KEDUANYA -- observasi
    aktor DAN estimasi yang ditampilkan/dinilai trust. Menghapus celah struktural yang
    memungkinkan agen "mengeksploitasi" info akurat yang tak terlihat forecaster kasar
    (`FormulaForecaster`) tanpa sengaja merusak akurasi janji ke pengguna.

    WAJIB `sim` diberikan (beda dari `FormulaForecaster` yang bisa jalan tanpanya) --
    `compute_virtual_wait` metode `Simulator`, bukan `SPKLU`."""

    def predict(self, spklus: dict, time_now_min: float = 0.0, user=None, soc: float = 50.0, sim=None) -> dict:
        assert sim is not None, "VirtualWaitForecaster butuh `sim` (compute_virtual_wait)"
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min))
               for sid, s in spklus.items()}


class MLPForecaster(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        # Softplus (bukan ReLU) di output: ReLU akhir bisa "mati" permanen (semua
        # pre-aktivasi < 0 -> output 0 & gradien 0 di SELURUH batch, tak terpulihkan
        # oleh Adam) pada inisialisasi acak tertentu -- ditemukan lewat rencana
        # pengujian B.3.3/C.3.1 (dead-neuron collapse, ~2/6 seed acak). Softplus tetap
        # menjamin output >= 0 tapi gradien tak pernah nol.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LearnedForecaster(ForecasterBase):
    """Mode B/C/D — regresor MLP PyTorch per-pasangan."""

    def __init__(self, hidden=64, lr=1e-3, clip=(0.0, 480.0)):
        self.hidden = hidden
        self.lr = lr
        self.clip = clip
        self.model = None
        self.opt = None
        self.loss_fn = nn.MSELoss()
        self._fitted = False
        self._fallback = FormulaForecaster()

    def _init_model_if_needed(self, input_dim):
        if self.model is None:
            self.model = MLPForecaster(input_dim, self.hidden)
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y, epochs=10, batch_size=64):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._init_model_if_needed(X.shape[1])
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                self.opt.zero_grad()
                pred = self.model(bx)
                loss = self.loss_fn(pred, by)
                loss.backward()
                self.opt.step()
        self.model.eval()
        self._fitted = True
        return self

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._init_model_if_needed(X.shape[1])
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        
        self.model.train()
        self.opt.zero_grad()
        pred = self.model(X_t)
        loss = self.loss_fn(pred, y_t)
        loss.backward()
        self.opt.step()
        self.model.eval()
        self._fitted = True
        return self

    def predict(self, spklus: dict, time_now_min: float = 0.0, user=None, soc: float = 50.0, sim=None) -> dict:
        if not self._fitted:
            return self._fallback.predict(spklus, time_now_min, user=user, soc=soc, sim=sim)
            
        sids = list(spklus.keys())
        recent_recs = sim.recent_recs if sim is not None else {}
        
        X_list = []
        for s in sids:
            recent_count = recent_recs.get(s, 0)
            X_list.append(extract_features(spklus[s], time_now_min, user=user, soc=soc, recent_recs_count=recent_count))
            
        X = np.vstack(X_list).astype(np.float32)
        self._init_model_if_needed(X.shape[1])
        
        with torch.no_grad():
            self.model.eval()
            pred = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
            
        lo, hi = self.clip
        return {s: float(np.clip(pred[i], lo, hi)) for i, s in enumerate(sids)}


def collect_forecast_dataset(sim, max_steps: int, agent=None):
    """Jalankan simulator step-demi-step, kumpulkan (X, y) supervised:
       X[t] = fitur pasangan (user, SPKLU) pada t ; y[t] = virtual wait SPKLU tsb (lookahead).
    Return (X, y) sebagai np.ndarray siap fit()."""
    sids = list(sim.spklus.keys())
    X_rows, y_rows = [], []
    for step in range(max_steps):
        time_now = step * sim.dt_minutes
        
        # 1. Dapatkan semua EV yang akan spawn di step ini
        spawns = sim.spawn_schedule.get(step, [])
        
        # 2. Ekstrak fitur & hitung virtual wait sebelum simulator maju ke step berikutnya
        if spawns:
            for spawn_tuple in spawns:
                # Unpack safely (in case old dataset without soc)
                if len(spawn_tuple) == 3:
                    user, spawn_loc, soc = spawn_tuple
                else:
                    user, spawn_loc = spawn_tuple
                    soc = 50.0
                
                for sid in sids:
                    recent = sim.recent_recs.get(sid, 0) if hasattr(sim, "recent_recs") else 0
                    dist = float(math.dist(spawn_loc, sim.spklus[sid].location))
                    feat = extract_features(
                        sim.spklus[sid], time_now, user=user, soc=soc,
                        recent_recs_count=recent, dist_override=dist
                    )
                    w = sim.compute_virtual_wait(user, sim.spklus[sid], time_now)
                    if np.isfinite(w):
                        X_rows.append(feat)
                        y_rows.append(w)
        else:
            # Fallback ke placeholder jika tidak ada spawn di step ini
            for sid in sids:
                recent = sim.recent_recs.get(sid, 0) if hasattr(sim, "recent_recs") else 0
                feat = extract_features(
                    sim.spklus[sid], time_now, user=None, soc=50.0,
                    recent_recs_count=recent
                )
                w = sim.compute_virtual_wait(None, sim.spklus[sid], time_now)
                if np.isfinite(w):
                    X_rows.append(feat)
                    y_rows.append(w)
            
        # 3. Jalankan satu step
        sim.step_once(step, agent=agent)
                
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)
