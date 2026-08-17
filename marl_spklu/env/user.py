import numpy as np
import math

# Kapasitas baterai & konsumsi FALLBACK (dataset baru selalu menyertakan nilai per-user
# dari campuran 3 model kendaraan -- Pemodelan_Variasi_Distribusi.md §8). Konstanta ini
# hanya dipakai untuk dataset lama yang tidak menyertakan field tsb.
BATTERY_KWH = 50.0
CONSUMPTION_KWH_KM = 0.15

# Trust awal (Beta-count: T = alpha/(alpha+beta)). Model_Simulasi_Inti.md §3.3 & §7
# Pemodelan_Variasi_Distribusi.md. alpha0=beta0=1 -> T0=0.5, netral.
TRUST_ALPHA0 = 1.0
TRUST_BETA0 = 1.0

# Laju belajar trust (epsilon_alpha, epsilon_beta) -- SELESAI, Pemodelan_Variasi_Distribusi.md
# §7.4 & §7.5 item 4: eps_alpha = eps_beta (asimetri "penalti > reward" sudah terwakili lewat
# BENTUK fungsi -- beta tumbuh tanpa batas vs alpha dibatasi eps_alpha, §7.2 -- jadi rasio
# eps tak perlu asimetri ganda). alpha0=beta0=1 (prior lemah, §7.4) dipilih krn user rata-rata
# hanya beberapa kali charging per horizon (freq_i median rendah) -- trust perlu responsif
# sejak pengalaman awal, bukan "lengket" (butuh puluhan observasi baru bergeser).
TRUST_EPS_ALPHA = 0.3
TRUST_EPS_BETA = 0.3

# Pita toleransi DeltaW (menit) -- mekanisme RELATIF thd skala wait time skenario
# (k1*wait_mean, k2*wait_mean), bukan absolut mutlak. k1=0.169, k2=0.506 -- TIDAK berubah.
# BASELINE_WAIT_MEAN_MINUTES di-RECALIBRATE ULANG setelah TAU_DEFAULT dikalibrasi ke
# 0,55 (Tier 9, 09_Kalibrasi_Temperature_Tau.ipynb) -- softmax P_pref jadi lebih tajam
# drpd tau=1.0 (MNL standar/sampling murni), memunculkan herding parsial yg realistis
# (Gini~0,70, dekat riil 0,68) tapi jg menaikkan wait_mean dari 2,33 -> 30,43 menit
# (~13x) drpd kondisi tau=1.0 sebelumnya. Ambang di bawah TETAP dihitung k1/k2*mean
# (metodologi lama dipertahankan apa adanya). Utk skenario beban lain
# (ringan/sedang/berat, §4.5), ambang HARUS dihitung ulang via k1/k2 * wait_mean_skenario.
BASELINE_WAIT_MEAN_MINUTES = 12.399169787745699
DELTAW_K1 = 0.169
DELTAW_K2 = 0.506
DELTAW_TOL_LOW = DELTAW_K1 * BASELINE_WAIT_MEAN_MINUTES   # = 2.10
DELTAW_TOL_HIGH = DELTAW_K2 * BASELINE_WAIT_MEAN_MINUTES  # = 6.27

# Mode zona PENALTI trust (zona reward SELALU absolut & terbatas, tak terpengaruh flag ini):
#   "abs"    -- default, keputusan final Pemodelan_Variasi_Distribusi.md 7.2. Meleset ke
#               arah manapun mengikis trust; yang dinilai AKURASI prediksi.
#   "signed" -- hanya (actual - est) >= TOL_HIGH yang dihukum; over-estimasi (user tiba
#               dan menunggu jauh lebih singkat dari janji) NETRAL, tak menghukum & tak
#               memberi reward. Landasan: expectation-disconfirmation -- diskonfirmasi
#               POSITIF tidak merusak kepercayaan.
#   CATATAN: "signed" TIDAK boleh diterapkan ke zona reward. Bila delta_w bertanda dipakai
#   di cabang reward, est yang sangat besar (delta_w -> sangat negatif) lolos uji
#   `<= TOL_LOW` dan memberi alpha TANPA BATAS -- persis eksploit yang dulu diperbaiki
#   dengan memperkenalkan abs(). Implementasi di update_trust menjaga invarian ini.
TRUST_PENALTY_MODE = "abs"

# Patience default (menit) -- 1 hari, renege nonaktif efektif (Model_Simulasi_Inti.md
# §4 & Pemodelan_Variasi_Distribusi.md §9). Dataset per-user boleh menimpa nilai ini.
DEFAULT_PATIENCE_MINUTES = 24 * 60

# gamma (sensitivitas P_rec thd wait) -- Kandidat A (titik tengah dari skala wait riil)
# -> B (pusat sapuan ablasi). Titik tengah dipilih via "half-life = wait_mean baseline":
# exp(-gamma*wait_mean) = 0.5. RECALIBRATED ulang sejalan dgn BASELINE_WAIT_MEAN_MINUTES
# baru (30,43 mnt, sesudah TAU_DEFAULT=0,55 -- lihat catatan di atas) -- gamma =
# ln(2)/30,43 = 0,02278. gamma tetap GLOBAL (sama semua user, §6.2 -- bukan per-user).
GAMMA_DEFAULT = 0.05590271          # titik tengah sapuan (ln(2)/12.399169787745699)
GAMMA_SWEEP = (0.02795135, 0.05590271, 0.11180542)  # x0.5, x1, x2 -- utk ablasi sensitivitas (§6.3 Kandidat B)

# tau (temperature softmax P_pref, decide_spklu) -- DIKALIBRASI via Tier 9
# (09_Kalibrasi_Temperature_Tau.ipynb): grid search tau x seed (grid kasar 7x10, grid
# halus 7x15 di sekitar kandidat terbaik, verifikasi independen 20 seed), kriteria
# berurutan (1) tanpa collapse -- semua SPKLU dpt kunjungan, (2) CV Gini lintas seed
# kecil (konsisten), (3) mean Gini baseline paling dekat Gini riil Klaster 12 (0,680).
# tau=1.0 (MNL standar/sampling murni) -> Gini=0,106 (TERLALU merata). tau->0 (argmax)
# -> Gini=0,833 via monopoli 1 SPKLU 99,8% demand (TERLALU timpang). tau=0,55 (target
# Gini TRANSAKSI murni) -> Gini verifikasi 0,698+-0,023, selisih 0,018, TAPI msh 1/20
# seed collapse parsial.
#
# REVISI (Tahap 6): Gini UTILISASI riil (0,422) jauh lbh rendah dari Gini TRANSAKSI
# riil (0,680) krn beda definisi (utilisasi riil dinormalisasi per hari-aktif-per-SPKLU;
# di simulasi keduanya SELALU SAMA krn semua SPKLU py horizon identik). Krn simulasi tak
# bisa cocok ke KEDUA target sekaligus, dipilih tau SEIMBANG yg meminimalkan total
# error kuadrat ke keduanya: target = mean(0,680,0,422) = 0,551. Hasil grid search
# (8 tau x 15 seed + verifikasi 20 seed independen): **tau=0,68** -> Gini=0,581+-0,033
# (95% CI), selisih ke transaksi=0,099, ke utilisasi=0,159, **0/20 (0%) seed collapse**
# (lebih andal dari tau=0,55). Lihat Tier 9 §5b utk detail lengkap & trade-off.
TAU_DEFAULT = 0.68


def soc_physical_range_km(soc_percent: float, battery_kwh: float = BATTERY_KWH,
                           consumption_kwh_km: float = CONSUMPTION_KWH_KM) -> float:
    """Jangkauan fisik maksimum (km) yang bisa ditempuh dengan SoC saat ini."""
    return (soc_percent / 100.0) * (battery_kwh / consumption_kwh_km)


def feasible_candidates(location, soc_percent, spklu_features,
                        willingness_radius_km=None, willingness_ratio=None,
                        battery_kwh: float = BATTERY_KWH,
                        consumption_kwh_km: float = CONSUMPTION_KWH_KM):
    """Himpunan SPKLU yang TERCAPAI dari `location`. reach = min(jangkauan fisik SoC,
    radius absolut). Rasio membatasi relatif thd SPKLU terdekat. Keduanya None ->
    seluruh SPKLU. Selalu kembalikan >=1 (fallback ke terdekat -> limp mode)."""
    sids = list(spklu_features.keys())
    if willingness_radius_km is None and willingness_ratio is None:
        return sids
    dists = {sid: math.dist(location, spklu_features[sid].get('loc', (0.0, 0.0))) for sid in sids}
    reach = soc_physical_range_km(soc_percent, battery_kwh, consumption_kwh_km)
    if willingness_radius_km is not None:
        reach = min(reach, willingness_radius_km)
    d_near = min(dists.values()) if dists else 0.0
    ratio_cap = (willingness_ratio * d_near) if willingness_ratio is not None else float('inf')
    feasible = [sid for sid, dd in dists.items() if dd <= reach and dd <= ratio_cap]
    return feasible if feasible else [min(dists, key=dists.get)]


class UserState:
    IDLE = "idle"
    SPAWNED = "spawned"
    TRAVELING = "traveling"
    QUEUING = "queuing"
    CHARGING = "charging"
    DONE = "done"
    RENEGED = "reneged"


class User:
    """Model pilihan SPKLU: U(s) = w1*dist + w2*ln(1+pop) + w3*ln(1+conn) + w4*isPrev
    + w5*(1-soc/100)*dist  (Model_Simulasi_Inti.md §3.1). w1-w5 datang dari kelas
    Latent Class MNL (Analisis_Model_Pilihan_Klaster31.ipynb §5) -- dibaca langsung
    per-user dari dataset, BUKAN di-draw acak (beda dari MXL populasi lama).

    Pilihan gabungan dengan rekomendasi: P = (1-T)*P_pref + T*P_rec (§3.2), T dari
    model trust Beta-count (§3.3). Balk DIHAPUS (Model_Simulasi_Inti.md §4) -- user
    selalu masuk antrian; hanya renege (patience) yang tersisa sbg jalan keluar."""

    def __init__(self, user_id: str, w1: float, w2: float, w3: float, w4: float, w5: float,
                 lcmnl_class: str = None, freq_i: float = 1.0,
                 connector_types=None,
                 battery_capacity_kwh: float = BATTERY_KWH,
                 consumption_kwh_km: float = CONSUMPTION_KWH_KM,
                 trust_alpha0: float = TRUST_ALPHA0, trust_beta0: float = TRUST_BETA0,
                 patience_minutes: float = DEFAULT_PATIENCE_MINUTES,
                 segment: str = "unlabeled"):
        self.user_id = user_id
        self.segment = segment  # label pelaporan saja -- tidak menentukan perilaku

        # Bobot utilitas preferensi (LCMNL, per-user tetap sepanjang hidup instance)
        self.w1, self.w2, self.w3, self.w4, self.w5 = float(w1), float(w2), float(w3), float(w4), float(w5)
        self.lcmnl_class = lcmnl_class
        self.freq_i = float(freq_i)  # frekuensi kunjungan (LogNormal) -- pelaporan/bobot spawn

        # Kendaraan: kapasitas & konsumsi HETEROGEN (campuran 3 model EV,
        # Pemodelan_Variasi_Distribusi.md §8) -- dipakai jangkauan spasial.
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.consumption_kwh_km = float(consumption_kwh_km)

        self.connector_types = set(connector_types) if connector_types else {"AC", "DC"}

        # Trust: Beta-count (alpha, beta) -- T = alpha/(alpha+beta). §3.3/§7.
        self.trust_alpha = float(trust_alpha0)
        self.trust_beta = float(trust_beta0)
        self.compliance_history = []
        self.interaction_history = []

        # Patience -- renege bila wait_time (antrian) melampaui ambang ini (menit).
        # Default 1 hari = renege nonaktif efektif (§9 Pemodelan_Variasi_Distribusi.md).
        self.patience_minutes = float(patience_minutes)

        # State tracking
        self.state = UserState.IDLE
        self.location = (0.0, 0.0)
        self.target_spklu = None
        # SoC (%) saat kedatangan/spawn TERKINI -- diisi simulator saat spawn (Simulator
        # step_once), dipakai model energi durasi charging (ΔSoC(%), Tier4 §3.6) -- None
        # sebelum spawn pertama / dataset lama tanpa SoC (fallback ke model durasi lama).
        self.soc = None
        self.prev_spklu = None  # state dependence (isPrev)
        self.remaining_travel_time = 0.0
        self.wait_time = 0.0

        # Jejak trip terakhir untuk update trust saat sesi selesai/reneged.
        self.est_wait_presented = 0.0
        self.last_rec_complied = False

        # Diagnostik keputusan terakhir.
        self.last_candidate_ids = []
        self.last_final_probs = None
        self.last_u_pref = None

        # Mode kanal rekomendasi & aturan keputusan.
        self.rec_mode = "estwait"
        self.rec_weight = None
        # Default SAMPLING dari final_probs (bukan argmax) -- user memilih SPKLU scr
        # stokastik sesuai distribusi gabungan preferensi+rekomendasi (decide_spklu),
        # bukan selalu SPKLU probabilitas tertinggi. Set False utk perilaku deterministik
        # lama (mis. reproduksi eksperimen ablasi sebelumnya).
        self.rec_stochastic = True
        # Temperature softmax utk P_pref (bukan P_rec) -- lihat decide_spklu & TAU_DEFAULT.
        self.tau = TAU_DEFAULT

        # Trust IMAJINER/bayangan (opsional, diset ablation `constant_trust_shadow`):
        # bila terisi, decide_spklu MEMAKAI nilai ini utk mencampur P_pref/P_rec (variabel
        # kontrol eksperimen dibekukan), TAPI trust_alpha/trust_beta TETAP diperbarui
        # normal oleh update_trust() -- property `trust` (di bawah) jadi murni diagnostik:
        # "trust yg SEHARUSNYA terjadi bila tak dibekukan". TIDAK dibaca oleh _build_obs
        # manapun (H-PPO rollout.py / PDQN pdqn_agent.py) -- murni utk logging eksternal,
        # bukan bagian observasi aktor/kritik.
        self._trust_override = None

    @property
    def trust(self) -> float:
        return self.trust_alpha / (self.trust_alpha + self.trust_beta)

    @property
    def trust_effective(self) -> float:
        """Trust yg BENAR-BENAR dipakai mencampur keputusan (decide_spklu) -- sama dgn
        `trust` kecuali `_trust_override` aktif (ablasi constant_trust_shadow)."""
        return self.trust if self._trust_override is None else float(self._trust_override)

    def spawn(self, location: tuple):
        self.state = UserState.SPAWNED
        # Perbaikan sirkularitas kalibrasi (Validasi_Generik/
        # Diagnosis_Sirkularitas_Kalibrasi.md §4.1): `location` param di sini adalah
        # anchor GMM yg disampling SEKALI per user saat generate (generate_klaster12.py)
        # -- HANYA valid sbg referensi jarak utk permintaan PERTAMA user (blm py
        # prev_spklu), persis spt estimasi w1-w5 yg membuang occasion tanpa
        # prev_ID_SPKLU. Utk permintaan ke-2+, self.location SUDAH di-update ke lokasi
        # SPKLU terakhir dikunjungi (simulator.py, saat sesi selesai) -- JANGAN ditimpa
        # anchor awal itu lagi di sini.
        if self.prev_spklu is None:
            self.location = location
        self.target_spklu = None
        self.remaining_travel_time = 0.0
        self.wait_time = 0.0
        self.est_wait_presented = 0.0
        self.last_rec_complied = False

    def decide_spklu(self, recommendations: list, estimated_waits: dict, spklu_features: dict,
                      speed_kmh: float = 40.0, gamma: float = GAMMA_DEFAULT, soc_percent: float = 50.0,
                      willingness_radius_km: float = None, willingness_ratio: float = None,
                      queue_lengths: dict = None, tau: float = None) -> str:
        """spklu_features: {sid: {'loc': (x,y), 'pop': popularity, 'conn': connectors}}.
        `queue_lengths` diterima utk kompatibilitas signature tapi TIDAK dipakai untuk
        balking (dihapus dari mekanisme default, Model_Simulasi_Inti.md §4).
        `tau`: override sesaat suhu softmax P_pref; default None -> pakai `self.tau`."""
        tau = self.tau if tau is None else tau
        spklu_ids = feasible_candidates(self.location, soc_percent, spklu_features,
                                        willingness_radius_km, willingness_ratio,
                                        battery_kwh=self.battery_capacity_kwh,
                                        consumption_kwh_km=self.consumption_kwh_km)

        rec_set = set(recommendations)
        u_pref = np.zeros(len(spklu_ids))

        for i, sid in enumerate(spklu_ids):
            features = spklu_features[sid]
            target_loc = features.get('loc', (0.0, 0.0))
            dist_km = math.dist(self.location, target_loc)
            pop = features.get('pop', 1.0)
            conn = features.get('conn', 1.0)
            log_pop = math.log(1 + pop)
            log_conn = math.log(1 + conn)
            is_prev = 1.0 if sid == self.prev_spklu else 0.0
            soc_urgency = (1.0 - soc_percent / 100.0) * dist_km

            # U(s) = w1*dist + w2*ln(1+pop) + w3*ln(1+conn) + w4*isPrev + w5*soc_urgency
            u_pref[i] = (self.w1 * dist_km + self.w2 * log_pop + self.w3 * log_conn
                         + self.w4 * is_prev + self.w5 * soc_urgency)

        # P_pref = softmax(U/tau) -- temperature DITERAPKAN SEBELUM softmax (bukan pasca-
        # softmax pd P_pref jadi). tau kecil -> distribusi lebih tajam (mendekati argmax
        # saat tau->0); tau=1 -> MNL standar (perilaku default/lama).
        u_scaled = u_pref / max(tau, 1e-9)
        u_max = np.max(u_scaled) if len(u_scaled) > 0 else 0.0
        exp_u = np.exp(u_scaled - u_max)
        p_pref = exp_u / np.sum(exp_u)

        # P_rec = softmax(exp(-gamma*wait)) atas SPKLU yg direkomendasikan (§3.2)
        p_rec_raw = {}
        for sid in spklu_ids:
            p_rec_raw[sid] = np.exp(-gamma * estimated_waits.get(sid, 0.0)) if sid in rec_set else 0.0
        p_rec_sum = sum(p_rec_raw.values())
        p_rec = {k: (v / p_rec_sum if p_rec_sum > 0 else 0.0) for k, v in p_rec_raw.items()}

        # P = (1-T)*P_pref + T*P_rec
        T = self.trust_effective
        final_probs = np.array([(1.0 - T) * p_pref[i] + T * p_rec[sid] for i, sid in enumerate(spklu_ids)])
        s = final_probs.sum()
        final_probs = final_probs / s if s > 0 else np.ones(len(spklu_ids)) / len(spklu_ids)

        chosen_idx = (int(np.random.choice(len(spklu_ids), p=final_probs))
                      if self.rec_stochastic else int(np.argmax(final_probs)))

        self.last_candidate_ids = list(spklu_ids)
        self.last_final_probs = final_probs.copy()
        self.last_u_pref = u_pref.copy()

        chosen_spklu = str(spklu_ids[chosen_idx])
        complied = chosen_spklu in rec_set
        self.compliance_history.append(1 if complied else 0)
        if len(self.compliance_history) > 30:
            self.compliance_history.pop(0)

        self.last_rec_complied = complied
        self.est_wait_presented = float(estimated_waits.get(chosen_spklu, 0.0)) if complied else 0.0

        self.target_spklu = chosen_spklu
        self.prev_spklu = chosen_spklu

        target_loc = spklu_features[chosen_spklu].get('loc', (0.0, 0.0))
        dist_km = math.dist(self.location, target_loc)
        self.remaining_travel_time = (dist_km / speed_kmh) * 60.0
        self.state = UserState.TRAVELING

        return chosen_spklu

    def step(self, dt_minutes: float = 15.0):
        """Return True bila user RENEGE pada step ini (patience terlampaui)."""
        if self.state == UserState.TRAVELING:
            self.remaining_travel_time -= dt_minutes
            if self.remaining_travel_time <= 0:
                self.state = UserState.QUEUING
                self.wait_time = abs(self.remaining_travel_time)
        elif self.state == UserState.QUEUING:
            self.wait_time += dt_minutes
            if self.wait_time >= self.patience_minutes:
                self.state = UserState.RENEGED
                return True
        return False

    def update_trust(self, est_wait: float, actual_wait: float):
        """DeltaW ABSOLUT |actual - estimasi| (Pemodelan_Variasi_Distribusi.md §7.2, keputusan
        final): estimasi meleset jauh -- ke arah manapun -- sama-sama mengikis trust, karena
        yang dinilai adalah AKURASI prediksi wait, bukan cuma "untung karena lebih cepat dari
        janji". Ini simetris, beda dari versi bertanda sebelumnya yang memberi reward tanpa
        batas untuk estimasi yang jauh meleset ke arah lebih cepat.
          DeltaW <= 5   -> alpha += eps_alpha*(1 - DeltaW/5)   (zona reward, estimasi akurat)
          DeltaW >= 15  -> beta  += eps_beta*(DeltaW/15)        (zona penalti, tumbuh tanpa batas)
          5 < DeltaW < 15 -> netral, tak ada update (zona toleransi).

        Zona PENALTI dapat dialihkan ke mode bertanda via TRUST_PENALTY_MODE (lihat
        catatan di sana). Zona REWARD sengaja TETAP absolut di kedua mode -- itu yang
        mencegah eksploit "janjikan wait sangat besar utk memanen alpha tak terbatas"."""
        signed = actual_wait - est_wait
        delta_w = abs(signed)
        if delta_w <= DELTAW_TOL_LOW:
            self.trust_alpha += TRUST_EPS_ALPHA * (1.0 - delta_w / DELTAW_TOL_LOW)
        elif (signed if TRUST_PENALTY_MODE == "signed" else delta_w) >= DELTAW_TOL_HIGH:
            self.trust_beta += TRUST_EPS_BETA * (delta_w / DELTAW_TOL_HIGH)
        # else: zona netral, tak ada perubahan
