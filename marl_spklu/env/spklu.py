import json
import math
import os
import random

# --- Kalibrasi waktu layanan (menit) per tipe konektor ---
# Sumber: kolom DURASI data transaksi riil Jabodetabek 2024 (SPKLU Publik).
# AC (7/22 kW) jauh lebih lama daripada DC-fast (>=25 kW): median 105 vs 41 menit,
# mean 136 vs 49 menit. Lognormal dikalibrasi ke median & persentil-90 empiris,
# sehingga sekaligus mereproduksi median DAN mean durasi riil.
# Perbedaan ini menentukan okupansi/antrean: stasiun 1-konektor AC lambat bisa
# padat walau demand rendah ("hidden hotspot"), sedangkan DC-fast menyerap demand.
# AC TETAP pakai model ini (bukan model energi di bawah) -- data SoC utk sesi AC di
# data transaksi riil terlalu sedikit (~4 baris) utk diestimasi scr andal (Tier4 §3.6).
SERVICE_TIME_PARAMS = {
    "AC": {"mu_log": 4.654, "sigma_log": 0.757, "mean": 136.0},  # median 105, p90 277 mnt
    "DC": {"mu_log": 3.714, "sigma_log": 0.613, "mean": 49.0},   # median 41,  p90 90  mnt
}
SERVICE_TIME_CLIP = (5.0, 480.0)  # menit; batasi ekor tak realistis (>8 jam)


def _load_soc_delta_model():
    """Muat model ΔSoC(%)~f(FIRST_SOC%) (Tier4 §3.6, Panduan_Model_DeltaSoC.docx,
    model_config.json) -- dipakai HANYA utk konektor DC. Metode dipilih via
    perbandingan PIT KS (notebook §3.6.4): "beta_regression" (Metode 2, dipakai bila
    PIT KS-nya menang -- data ΔSoC di sini skewed, bukan simetris) atau
    "regression_truncnorm" (Metode 1, fallback bila Beta tak menang). Gagal-aman ke
    None (fallback ke SERVICE_TIME_PARAMS lama) bila config tak ada/skema tak dikenal.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "Synthetic Model", "model_config.json"),
        os.path.join(os.getcwd(), "Synthetic Model", "model_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            sd = cfg.get("tier4", {}).get("soc_delta")
            if not sd or "DC" not in sd.get("applies_to", []):
                continue
            # Catatan: config msh menyimpan kw_efektif_dc/sigma_kw_log (dari Tier4 §3.6.5,
            # dokumentasi/analisis) tapi TIDAK dipakai di sini -- durasi produksi pakai
            # daya_kw_dc riil per stasiun (Tier1, param sample_charge_time), bukan
            # kw_efektif hasil estimasi/sampling (lihat catatan _sample_charge_time_energy).
            base = {
                "method": sd["method"],
                "first_soc_range": tuple(sd["first_soc_range"]),
            }
            if sd["method"] == "beta_regression":
                base.update(a0=float(sd["a0"]), a1=float(sd["a1"]),
                            p0=float(sd["p0"]), p1=float(sd["p1"]))
                return base
            if sd["method"] == "regression_truncnorm":
                base.update(mu_alpha=float(sd["mu_alpha"]), mu_beta=float(sd["mu_beta"]),
                            sigma_s0=float(sd["sigma_s0"]), sigma_s1=float(sd["sigma_s1"]),
                            sigma_floor=float(sd["sigma_floor"]))
                return base
    return None


_SOC_DELTA_MODEL = _load_soc_delta_model()

_TRUNCNORM_MAX_TRIES = 50  # rejection sampling (fallback Metode 1); clip bila tak kunjung diterima


def _sample_delta_soc(first_soc: float) -> float:
    """Sampel ΔSoC(%) | FIRST_SOC=x, sesuai metode terpilih di model_config.json.

    "beta_regression" (Metode 2, Panduan §4.2): Ytilde=ΔSoC/(100-x) ~ Beta(mu*phi,
    (1-mu)*phi), mu=sigmoid(a0+a1*x), phi=exp(p0+p1*x) -- TAK PERNAH bocor keluar
    [0,100-x] scr konstruksi (domain Beta asli [0,1]). Dipakai stdlib `random.betavariate`.

    "regression_truncnorm" (Metode 1, fallback): TruncatedNormal(mu(x),sigma(x),0,100-x)
    via rejection sampling (hindari dependensi scipy di kode produksi).
    """
    m = _SOC_DELTA_MODEL
    lo_range, hi_range = m["first_soc_range"]
    x = min(max(first_soc, lo_range), hi_range)
    upper = 100.0 - first_soc

    if m["method"] == "beta_regression":
        mu = 1.0 / (1.0 + math.exp(-(m["a0"] + m["a1"] * x)))
        phi = math.exp(m["p0"] + m["p1"] * x)
        alpha, beta_ = mu * phi, (1.0 - mu) * phi
        y_tilde = random.betavariate(max(alpha, 1e-6), max(beta_, 1e-6))
        return min(max(y_tilde * upper, 0.0), upper)

    mu = m["mu_alpha"] + m["mu_beta"] * x
    sigma = max(m["sigma_s0"] + m["sigma_s1"] * x, m["sigma_floor"])
    for _ in range(_TRUNCNORM_MAX_TRIES):
        y = random.gauss(mu, sigma)
        if 0.0 <= y <= upper:
            return y
    return min(max(mu, 0.0), upper)  # fallback: clip mean ke batas fisik


def _sample_charge_time_energy(first_soc: float, capacity_kwh: float, daya_efektif_dc: float) -> float:
    """Durasi charging DC dari model energi (Tier4 §3.6): ΔSoC(%) tergantung FIRST_SOC
    -> energi (kWh, pakai kapasitas kendaraan AGEN ini, bukan rata-rata historis) ->
    durasi = energi / daya_efektif_dc.

    `daya_efektif_dc` = daya RIIL TERSERAP per stasiun (kW, median `implied_kw =
    KWH/(DURASI/60)` dari transaksi riil ID_SPKLU ybs, Tier1) -- BUKAN rating label
    charger (`daya_kw_dc`, mis. "200 kW" di spesifikasi alat). Efisiensi (rasio
    efektif/rating) ternyata bervariasi jauh antar-stasiun (21%-94%, bukan rasio
    tunggal), jadi dihitung PER STASIUN, bukan rating dikali faktor global. Statis
    per konektor -- tidak disampling (beda dari kw_efektif+LogNormal yg tadinya
    dipakai & terbukti memicu ledakan waktu tunggu di stasiun ber-konektor tunggal
    yg hampir jenuh -- lihat riwayat perubahan)."""
    delta_soc = _sample_delta_soc(first_soc)
    energy_kwh = capacity_kwh * (delta_soc / 100.0)
    duration_min = (energy_kwh / daya_efektif_dc) * 60.0
    lo, hi = SERVICE_TIME_CLIP
    return min(max(duration_min, lo), hi)


def sample_charge_time(connector_type: str, user=None, daya_efektif_dc: float = None) -> float:
    """Sampel durasi layanan (menit) untuk satu sesi.

    DC dgn `user` (punya `.soc` & `.battery_capacity_kwh`), model ΔSoC tersedia, DAN
    `daya_efektif_dc` (daya riil terserap stasiun ybs, >0): durasi dari kebutuhan
    energi (ΔSoC x kapasitas kendaraan agen ini / daya efektif stasiun).
    AC, atau DC tanpa salah satu data di atas: fallback ke lognormal terkalibrasi lama.
    """
    if (connector_type == "DC" and _SOC_DELTA_MODEL is not None
            and user is not None and getattr(user, "soc", None) is not None
            and daya_efektif_dc):
        t = _sample_charge_time_energy(user.soc, user.battery_capacity_kwh, daya_efektif_dc)
        if t is not None:
            return t

    p = SERVICE_TIME_PARAMS.get(connector_type, SERVICE_TIME_PARAMS["DC"])
    t = random.lognormvariate(p["mu_log"], p["sigma_log"])
    lo, hi = SERVICE_TIME_CLIP
    return min(max(t, lo), hi)


def mean_charge_time(connector_type: str) -> float:
    """Rata-rata durasi layanan (menit) per tipe konektor, untuk estimasi tunggu."""
    return SERVICE_TIME_PARAMS.get(connector_type, SERVICE_TIME_PARAMS["DC"])["mean"]


class SPKLU:
    def __init__(self, spklu_id: str, capacities: dict, location: tuple, daya_efektif_dc: float = 0.0):
        """
        spklu_id: unique identifier
        capacities: dict like {'AC': 2, 'DC': 1}
        location: (x, y) coordinates
        daya_efektif_dc: daya riil terserap konektor DC stasiun ini (kW, dari Tier1,
            median implied_kw per stasiun) -- dipakai model energi durasi charging
            (sample_charge_time); 0.0 bila tak ada konektor DC atau data tak tersedia
            (fallback ke model lognormal lama).
        """
        self.spklu_id = spklu_id
        self.location = location
        self.capacities = capacities
        self.daya_efektif_dc = daya_efektif_dc
        
        # Discrete queues and charging states
        self.queues = {c_type: [] for c_type in capacities if capacities[c_type] > 0}
        self.charging = {c_type: [] for c_type in capacities if capacities[c_type] > 0}
                
        # Metrics
        self.total_served = 0
        self.total_wait_time = 0.0

        # Perbaikan T2/T3 (Validasi_Generik/LAPORAN_VALIDASI.md): EV yang SEDANG
        # MENUJU stasiun ini (state TRAVELING, target_spklu == ini) akan merebut slot
        # bebas sebelum EV yang baru query estimate_wait_time tiba -- diisi Simulator
        # tiap step lewat _refresh_pending_arrivals(). {connector_type: jumlah EV
        # dalam perjalanan yang kompatibel}. Default kosong -> perilaku identik versi
        # lama utk pemakaian SPKLU di luar Simulator (mis. unit test langsung).
        self.pending_arrivals: dict = {}

    def get_queue_length(self, connector_type: str = None) -> int:
        if connector_type:
            if connector_type in self.queues:
                return len(self.queues[connector_type])
            return 0
        return sum(len(q) for q in self.queues.values())
        
    def get_utilization(self, connector_type: str = None) -> float:
        """Returns instantaneous utilization ratio (0 to 1)"""
        if connector_type:
            cap = self.capacities.get(connector_type, 0)
            if cap > 0:
                return len(self.charging.get(connector_type, [])) / cap
            return 0.0
            
        total_cap = sum(self.capacities.values())
        if total_cap == 0: return 0.0
        total_used = sum(len(c) for c in self.charging.values())
        return total_used / total_cap

    def estimate_wait_time(self, connector_type: str, avg_charge_time: float = None) -> float:
        """
        Deterministic estimation of wait time for the Discrete Time Step model.
        Bila avg_charge_time None, pakai rata-rata layanan terkalibrasi per tipe
        konektor (AC ~136 mnt, DC ~49 mnt) -> estimasi tunggu peka tipe konektor.

        Perbaikan T2/T3: konektor "bebas" HANYA benar-benar bebas bila jumlah slot
        kosong melebihi jumlah EV yang sedang menuju stasiun ini (`pending_arrivals`,
        lihat __init__). Kalau tidak, EV yang sedang di-query juga harus antre di
        BELAKANG kedatangan yang akan datang itu. `pending_arrivals` kosong (default)
        -> formula identik versi lama (0.0 saat ada slot bebas).
        """
        cap = self.capacities.get(connector_type, 0)
        if cap == 0:
            return float('inf')
        if avg_charge_time is None:
            avg_charge_time = mean_charge_time(connector_type)

        q_len = len(self.queues.get(connector_type, []))
        currently_charging = len(self.charging.get(connector_type, []))
        free_slots = max(0, cap - currently_charging)
        n_pending = self.pending_arrivals.get(connector_type, 0)

        if free_slots > n_pending:
            return 0.0
        ahead = q_len + max(0, n_pending - free_slots)
        return ((ahead + 1) * avg_charge_time) / cap
            
    def request_connector(self, user_id: str, connector_type: str):
        """Adds a user to the queue"""
        if connector_type in self.queues:
            self.queues[connector_type].append(user_id)

    def remove_from_queue(self, user_id: str):
        """Keluarkan user dari SEMUA antrean SPKLU ini (renege -- patience terlampaui,
        Model_Simulasi_Inti.md §4). Tidak berefek bila user tak ditemukan (mis. sudah
        keburu masuk konektor step yang sama)."""
        for q in self.queues.values():
            if user_id in q:
                q.remove(user_id)
            
    def step(self, dt_minutes: float = 15.0, user_lookup: dict = None) -> list:
        """
        Advances the SPKLU state by dt_minutes.
        Returns a tuple of (finished_users, newly_charging)
        where newly_charging is a list of tuples (user_id, carry_over_time)

        user_lookup: dict {user_id: User} opsional -- dipakai model energi durasi DC
        (Tier4 §3.6, butuh `.soc` & `.battery_capacity_kwh` milik user ybs). Tanpa ini,
        fallback ke model durasi lognormal lama (lihat sample_charge_time).
        """
        finished_users = []
        newly_charging = []
        
        for c_type, cap in self.capacities.items():
            if cap == 0: continue
            
            # 1. Update charging EVs
            remaining_chargers = []
            available_carry_overs = []
            
            for ev in self.charging[c_type]:
                ev["remaining_time"] -= dt_minutes
                if ev["remaining_time"] <= 0:
                    finished_users.append(ev["user_id"])
                    self.total_served += 1
                    # Connector became free early!
                    available_carry_overs.append(abs(ev["remaining_time"]))
                else:
                    remaining_chargers.append(ev)
            self.charging[c_type] = remaining_chargers
            
            # 2. Fill empty connectors from queue
            while len(self.charging[c_type]) < cap and len(self.queues[c_type]) > 0:
                next_user = self.queues[c_type].pop(0)
                
                # Remove this user from other queues in the same SPKLU
                for other_c_type in self.queues.keys():
                    if other_c_type != c_type and next_user in self.queues[other_c_type]:
                        self.queues[other_c_type].remove(next_user)
                        
                user = user_lookup.get(next_user) if user_lookup else None
                charge_time = sample_charge_time(c_type, user=user, daya_efektif_dc=self.daya_efektif_dc)

                carry = 0.0
                if available_carry_overs:
                    carry = available_carry_overs.pop(0)
                    charge_time -= carry
                    
                self.charging[c_type].append({
                    "user_id": next_user,
                    "remaining_time": max(0.0, charge_time)
                })
                newly_charging.append((next_user, carry))
                
        return finished_users, newly_charging
