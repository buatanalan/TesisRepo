import inspect
import numpy as np
import math
from collections import defaultdict
from marl_spklu.env.user import UserState, feasible_candidates
from marl_spklu.env.spklu import SPKLU, sample_charge_time

# Sentinel wait time (menit) utk SPKLU yg TIDAK PUNYA konektor kompatibel sama sekali
# dgn EV user (mis. EV butuh DC, stasiun cuma AC atau n_conn_dc=0). BUKAN 0.0 (yg salah
# membuat stasiun tampak paling menarik/instan) & BUKAN float('inf') (akan merambat jadi
# NaN di observasi RL numerik -- lihat rollout.py yg memakai nilai ini langsung sbg array
# fitur tanpa filter isfinite). Nilai besar tapi BERHINGGA -- tetap valid scr aritmetik.
UNREACHABLE_WAIT_MINUTES = 1e5

class Simulator:
    def __init__(self, spklu_dict: dict, users: list, history_buffer,
                 log_actor_states: bool = False, log_active_only: bool = True,
                 log_every: int = 1, user_willingness_radius_km: float = None,
                 user_willingness_ratio: float = None,
                 rekam_deret: bool = False):
        self.recent_recs = defaultdict(int)
        self.spklus = spklu_dict
        self.users = users
        self.history = history_buffer
        self.spawn_schedule = {} # {step: [(User, (lat, lon))]}
        self.logs = []
        self.detailed_logs = []
        self.herding_events = 0
        self.current_step = 0
        self.dt_minutes = 15.0

        # Distribusi rekomendasi per-step (untuk Recommendation Entropy & Herding
        # Index ternormalisasi). Hanya terisi saat ada agen yang merekomendasikan.
        self.rec_distribution_log = []

        # Jejak PER-KEPUTUSAN (bukan per-trip-selesai spt `logs`). Diperlukan untuk
        # menjawab MENGAPA satu kebijakan mengungguli yang lain: `logs` hanya mencatat
        # hasil akhir (stasiun mana, tunggu berapa), tidak mencatat APA YANG DITAWARKAN,
        # seberapa jauh pengguna DIDORONG dari pilihan alaminya, dan berapa trust-nya
        # SAAT memutuskan. Hanya operasi append -- tidak menyentuh RNG, tidak mengubah
        # perilaku simulasi.
        self.decision_log = []

        # DERET WAKTU (opsional, `rekam_deret=True`) -- lubang terbesar sebelumnya: kondisi
        # jaringan hanya diketahui pada AKHIR simulasi (`total_served` kumulatif), sehingga
        # Gini akhir tak dapat dibedakan antara "merata sepanjang horizon" dan "timpang di
        # awal lalu terkoreksi". Dimatikan saat PELATIHAN (300 iterasi x banyak lengan akan
        # membengkak); dinyalakan saat EVALUASI.
        self.rekam_deret = bool(rekam_deret)
        self.station_log = []      # snapshot per JAM per stasiun
        self.daily_log = []        # snapshot per HARI tingkat jaringan
        self._served_hari_lalu = {}
        self._log_terakhir_jam = -1
        self._log_terakhir_hari = -1

        # Trace state per-aktor per-step (OPSIONAL; bisa besar untuk horizon
        # panjang). Nonaktif secara default -> nol overhead untuk run validasi.
        #   log_actor_states : hidupkan perekaman snapshot aktor.
        #   log_active_only  : lewati user IDLE/DONE (memangkas volume drastis).
        #   log_every        : rekam tiap-N step (downsample untuk horizon panjang).
        self.log_actor_states = log_actor_states
        self.log_active_only = log_active_only
        self.log_every = max(1, int(log_every))
        self.actor_state_log = []   # snapshot tiap User per step
        self.spklu_state_log = []   # snapshot tiap SPKLU per step

        # Batas jangkauan spasial user (radius kemauan, km). None = tanpa batas
        # (perilaku lama). Bila diisi, kandidat SPKLU di decide_spklu dibatasi ke
        # min(radius ini, jangkauan fisik SoC) -> menaikkan kesulitan & realisme.
        self.user_willingness_radius_km = user_willingness_radius_km
        # Batas jangkauan RELATIF (rasio thd SPKLU terdekat). None = tak dipakai.
        self.user_willingness_ratio = user_willingness_ratio

        # Balk DIHAPUS dari mekanisme default (Model_Simulasi_Inti.md §4) -- user selalu
        # masuk antrian; hanya renege (patience per-user) yang tersisa sbg jalan keluar.

        # Scheduling
        self.spawn_schedule = {} # step -> list of users
        self.spklu_locations = {sid: s.location for sid, s in self.spklus.items()}
        self.spklu_features = {
            sid: {
                'loc': s.location,
                'pop': getattr(s, "popularity", 1.0),
                'conn': sum(s.capacities.values()),
                'cluster_id': getattr(s, "cluster_id", None)
            }
            for sid, s in self.spklus.items()
        }
        
    def load_from_dataset(self, dataset_path: str):
        """Memuat scenario_dataset.json. Mendukung skema BARU (Klaster 31,
        generate_klaster31.py -- location_km/w1..w5/kendaraan/trust/patience) dengan
        fallback ke skema LAMA (location/beta_state/w_i/battery_kwh) untuk dataset lama."""
        import json
        from marl_spklu.env.spklu import SPKLU
        from marl_spklu.env.user import User, TRUST_ALPHA0, TRUST_BETA0, DEFAULT_PATIENCE_MINUTES

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        # Load SPKLUs -- 'location_km' (skema baru) diprioritaskan atas 'location' (lama).
        self.spklus = {}
        self.spklu_features = {}
        for s in dataset["spklus"]:
            loc = tuple(s["location_km"]) if "location_km" in s else tuple(s["location"])
            self.spklus[s["id"]] = SPKLU(s["id"], s["capacities"], loc,
                                         daya_efektif_dc=float(s.get("daya_efektif_dc", 0.0)))
            feat = {
                'loc': loc,
                'pop': float(s.get("popularity", 1.0)),
                'conn': sum(s["capacities"].values()),
                'cluster_id': s.get("cluster_id"),   # untuk disparitas dalam-klaster (reward RL)
            }
            if s.get("durasi_min") is not None:
                feat['durasi_min'] = float(s["durasi_min"])
            if s.get("operator") is not None:
                feat['operator'] = int(s["operator"])
            self.spklu_features[s["id"]] = feat
        self.spklu_locations = {sid: s.location for sid, s in self.spklus.items()}

        # Load Users
        self.users = []
        for u in dataset["users"]:
            if "w1_dist_km" in u:
                # Skema baru: w1-w5 langsung dari kelas LCMNL (bukan draw acak MXL).
                self.users.append(User(
                    u["user_id"],
                    w1=u["w1_dist_km"], w2=u["w2_log_pop"], w3=u["w3_log_conn"],
                    w4=u["w4_isPrev"], w5=u["w5_soc_urgency"],
                    lcmnl_class=u.get("lcmnl_class"), freq_i=u.get("freq_i", 1.0),
                    connector_types=u.get("connector_types"),
                    battery_capacity_kwh=u.get("battery_capacity_kwh", 50.0),
                    consumption_kwh_km=u.get("consumption_kwh_km", 0.15),
                    trust_alpha0=u.get("trust_alpha0", TRUST_ALPHA0),
                    trust_beta0=u.get("trust_beta0", TRUST_BETA0),
                    patience_minutes=u.get("patience_minutes", DEFAULT_PATIENCE_MINUTES),
                    segment=u.get("vehicle_model", "unlabeled"),
                ))
            else:
                # Skema lama: default_gravity/state->w4, w_i->tak terpakai lagi (tak ada
                # padanan langsung di model baru; T dibaca dari trust Beta-count seragam),
                # beta_pop/beta_conn/beta_dist lama -> w1/w2/w3 (mean populasi bila absen).
                self.users.append(User(
                    u["user_id"],
                    w1=u.get("beta_dist", -0.0585), w2=u.get("beta_pop", 0.8708),
                    w3=u.get("beta_conn", 0.0286), w4=u.get("beta_state", 4.04), w5=0.0,
                    connector_types=u.get("connector_types"),
                    battery_capacity_kwh=u.get("battery_kwh", 50.0),
                    consumption_kwh_km=0.15,
                    segment=u.get("segment", "unlabeled"),
                ))

        # Load Schedule -- skema baru pakai 'time_minutes' kontinu (dikonversi ke step
        # diskret dt_minutes=15 milik engine), skema lama pakai 'step' langsung.
        user_dict = {u.user_id: u for u in self.users}
        if dataset["schedule"] and "time_minutes" in dataset["schedule"][0]:
            max_steps = int(np.ceil(max(ev["time_minutes"] for ev in dataset["schedule"]) / self.dt_minutes)) + 1
            self.spawn_schedule = {s: [] for s in range(max_steps)}
            for ev in dataset["schedule"]:
                step = int(ev["time_minutes"] // self.dt_minutes)
                loc = tuple(ev.get("spawn_loc_km", ev.get("spawn_loc", (0.0, 0.0))))
                soc = ev.get("soc", 50.0)
                self.spawn_schedule.setdefault(step, []).append((user_dict[ev["user_id"]], loc, soc))
        else:
            max_steps = dataset["metadata"]["max_steps"]
            self.spawn_schedule = {s: [] for s in range(max_steps)}
            for ev in dataset["schedule"]:
                step = ev["step"]
                soc = ev.get("soc", 50.0)
                self.spawn_schedule.setdefault(step, []).append(
                    (user_dict[ev["user_id"]], tuple(ev["spawn_loc"]), soc))
        self.max_steps = max_steps  # horizon dataset, dipakai runner eksperimen
            
    @staticmethod
    def _call_predict_waits(agent, feasible_spklus, sim, user, time_now):
        """Perbaikan T2 (Validasi_Generik/LAPORAN_VALIDASI.md): teruskan `sim`/`user`/
        `time_now` ke `agent.predict_waits` HANYA bila metodenya benar2 menerima
        parameter itu (mis. GreedyAgent -> VirtualWaitPredictor) -- deteksi lewat
        signature, BUKAN try/except (supaya TypeError asli dari dalam predict_waits
        tak pernah tertelan tanpa sengaja). Agen lama/lain (RL rollout, wrapper skrip
        eksperimen) yang predict_waits-nya cuma menerima `spklus` tetap dipanggil
        persis spt sebelumnya -- tak ada perubahan perilaku utk mereka."""
        try:
            params = inspect.signature(agent.predict_waits).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_context = (
            any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            or {"sim", "user", "time_now"} <= set(params.keys())
        )
        if accepts_context:
            return agent.predict_waits(feasible_spklus, sim=sim, user=user, time_now=time_now)
        return agent.predict_waits(feasible_spklus)

    def _refresh_pending_arrivals(self):
        """Isi `spklu.pending_arrivals` dari EV yang sedang TRAVELING menuju tiap
        stasiun saat ini (lihat SPKLU.pending_arrivals, T2/T3). Dipanggil sekali per
        step, setelah arrival step ini ditangani (jadi tak menghitung ganda EV yang
        baru saja tiba) dan sebelum estimasi tunggu dipakai utk keputusan step ini."""
        counts = {sid: {} for sid in self.spklus}
        for u in self.users:
            if u.state != UserState.TRAVELING or u.target_spklu not in counts:
                continue
            spklu = self.spklus[u.target_spklu]
            conn_types = getattr(u, "connector_types", None) or list(spklu.capacities.keys())
            for ct in conn_types:
                if spklu.capacities.get(ct, 0) > 0:
                    counts[u.target_spklu][ct] = counts[u.target_spklu].get(ct, 0) + 1
        for sid, spklu in self.spklus.items():
            spklu.pending_arrivals = counts.get(sid, {})

    def get_spklu_wait_estimates(self):
        estimates = {}
        for sid, spklu in self.spklus.items():
            waits = []
            if spklu.capacities.get("AC", 0) > 0:
                waits.append(spklu.estimate_wait_time("AC"))
            if spklu.capacities.get("DC", 0) > 0:
                waits.append(spklu.estimate_wait_time("DC"))
            estimates[sid] = min(waits) if waits else float('inf')
        return estimates
            
    def _handle_arrival(self, user, step, time_now):
        """Alokasikan konektor untuk satu user yang baru tiba di SPKLU tujuannya."""
        spklu = self.spklus[user.target_spklu]
        overshoot = abs(user.remaining_travel_time)

        # `pop` dinamis (Pemodelan_Variasi_Distribusi.md §5.3, Kandidat D): tiap
        # kedatangan dihitung sbg "kunjungan hari ini" -- dipakai utk mengganti pop
        # esok hari (lihat step_once, blok pergantian hari).
        if not hasattr(self, "_pop_visit_counter"):
            self._pop_visit_counter = {sid: 0 for sid in self.spklu_features}
        self._pop_visit_counter[user.target_spklu] = self._pop_visit_counter.get(user.target_spklu, 0) + 1

        # Bypass antrian jika ADA konektor bebas & KOMPATIBEL saat ini.
        # Bug D (FIXED): hanya tipe konektor yang kompatibel dengan EV yang dipertimbangkan
        #   (dulu user dimasukkan ke SEMUA tipe tanpa cek kompatibilitas AC/DC).
        # Bug E (FIXED): prioritas konektor realistis DC (cepat) sebelum AC
        #   (dulu AC selalu diprioritaskan karena mengikuti urutan dict capacities).
        station_types = [c for c, cap in spklu.capacities.items() if cap > 0]
        compatible = [c for c in station_types if c in user.connector_types]
        if not compatible:
            compatible = station_types  # guard: EV tak cocok tipe apa pun -> tetap dilayani
        _pref = {"DC": 0, "AC": 1}
        compatible.sort(key=lambda c: _pref.get(c, 2))

        free_c_type = next(
            (c for c in compatible if len(spklu.charging.get(c, [])) < spklu.capacities[c]),
            None)

        if free_c_type:
            user.state = UserState.CHARGING
            user.wait_time = 0.0
            charge_time = sample_charge_time(free_c_type, user=user, daya_efektif_dc=spklu.daya_efektif_dc) - overshoot
            spklu.charging[free_c_type].append({
                "user_id": user.user_id,
                "remaining_time": max(0.0, charge_time)
            })
            self.detailed_logs.append({
                "step": step, "time": time_now,
                "actor_ev": user.user_id, "actor_spklu": user.target_spklu,
                "event": f"[EV] {user.user_id} tiba dan lgsg charging di {user.target_spklu} ({free_c_type}). (Sisa waktu {overshoot:.1f}m dikonversi jadi waktu charge)"
            })
        else:
            # Semua konektor kompatibel penuh -> antre di SATU tipe dgn estimasi tunggu
            # terpendek. Bug D (FIXED): dulu masuk ke SEMUA antrean -> panjang antrean &
            # EstWait menggelembung (dobel-hitung entitas yang sama).
            user.wait_time = overshoot
            queue_c_type = min(compatible, key=lambda c: spklu.estimate_wait_time(c))
            spklu.request_connector(user.user_id, queue_c_type)
            self.detailed_logs.append({
                "step": step, "time": time_now,
                "actor_ev": user.user_id, "actor_spklu": user.target_spklu,
                "event": f"[EV] {user.user_id} tiba di {user.target_spklu} dan masuk antrian {queue_c_type}. (Nunggu {overshoot:.1f}m)"
            })

    def run(self, max_steps: int = 672, agent=None):
        for step in range(max_steps):
            self.step_once(step, agent=agent)

    def step_once(self, step: int, agent=None):
        """Menjalankan satu langkah simulasi (isi loop `run`). Dipisah dari
        `run` supaya caller eksternal (mis. notebook visualisasi) bisa
        memajukan simulasi satu step demi satu step dan merekam state
        internal (charger/EV) di antaranya, tanpa mengubah perilaku `run`."""
        self.current_step = step
        time_now = step * self.dt_minutes

        # Reset per jendela (mis. setiap 96 step = 24 jam)
        if step % 96 == 0:
            self.recent_recs.clear()

            # `pop` dinamis, Kandidat D (Pemodelan_Variasi_Distribusi.md §5.3, SELESAI):
            # tiap pergantian hari, pop diganti PENUH dengan jumlah kunjungan simulasi
            # hari SEBELUMNYA (bukan blend/decay) -- valid krn pop_awal sudah dinormalkan
            # ke skala laju harian rata-rata (bukan total tahunan mentah), sepadan dgn
            # skala kunjungan simulasi harian. Hari ke-0 tetap memakai pop_awal dari dataset.
            if step > 0 and hasattr(self, "_pop_visit_counter"):
                for sid in self.spklu_features:
                    self.spklu_features[sid]["pop"] = float(self._pop_visit_counter.get(sid, 0))
                self._pop_visit_counter = {sid: 0 for sid in self.spklu_features}

        # Snapshot antrean SEBELUM window ini diproses -- basis "peningkatan beban
        # puncak" utk Flocking Index (definisi Bab IV.2 tesis, para. 686-698: rasio
        # antara peningkatan beban puncak stasiun target dan jumlah rekomendasi pada
        # stasiun itu dalam window 15 menit). dt_minutes=15 -> satu step == satu window.
        queue_before_window = {sid: s.get_queue_length() for sid, s in self.spklus.items()}

        # 1. Majukan waktu semua user; kumpulkan yang BARU tiba step ini.
        #    Patience 1 hari (default) membuat renege nyaris tak pernah terpicu --
        #    tetap ditangani agar varian ablasi patience-realistis (§9
        #    Pemodelan_Variasi_Distribusi.md) langsung berfungsi bila diaktifkan.
        arrivals = []
        for user in self.users:
            if user.state in [UserState.TRAVELING, UserState.QUEUING]:
                prev_state = user.state
                reneged = user.step(self.dt_minutes)
                if reneged:
                    if user.target_spklu in self.spklus:
                        self.spklus[user.target_spklu].remove_from_queue(user.user_id)
                    self.detailed_logs.append({
                        "step": step, "time": time_now,
                        "actor_ev": user.user_id, "actor_spklu": user.target_spklu,
                        "event": f"[EV] {user.user_id} RENEGE dari antrian {user.target_spklu} "
                                 f"setelah menunggu {user.wait_time:.1f}m (patience {user.patience_minutes:.0f}m)."
                    })
                elif prev_state == UserState.TRAVELING and user.state == UserState.QUEUING:
                    arrivals.append(user)

        # 2. Proses charging + isi konektor kosong dari ANTREAN LAMA lebih dulu.
        #    (Fix Bug F) Arrival ditangani SETELAH spklu.step, agar EV yang baru
        #    mulai charging tak ikut di-decrement pada step yang sama.
        #    Ini juga memberi pengantre lama prioritas FIFO yang benar atas arrival baru.
        user_lookup = {u.user_id: u for u in self.users}  # utk model energi durasi DC (Tier4 §3.6)
        for sid, spklu in self.spklus.items():
            finished_users, newly_charging = spklu.step(self.dt_minutes, user_lookup=user_lookup)

            for uid in finished_users:
                for u in self.users:
                    if u.user_id == uid and u.state == UserState.CHARGING:
                        u.state = UserState.DONE
                        # Perbaikan sirkularitas kalibrasi (Validasi_Generik/
                        # Diagnosis_Sirkularitas_Kalibrasi.md §4.1): dist_km di estimasi
                        # w1-w5 dihitung dari lokasi SESI SEBELUMNYA yg benar2 dikunjungi
                        # (prev_ID_SPKLU), bukan anchor sintetis beku. Update lokasi user
                        # ke SPKLU ybs stlh sesi selesai -> permintaan berikutnya (decide_spklu,
                        # feasible_candidates, travel time) konsisten pakai referensi yg SAMA
                        # dgn yg dipakai saat w1/w5 diestimasi. Permintaan PERTAMA (blm pernah
                        # charging) tetap pakai anchor spawn -- sama dgn estimasi yg membuang
                        # occasion tanpa prev_ID_SPKLU.
                        u.location = spklu.location
                        spklu.total_wait_time += u.wait_time  # (Fix Bug G)
                        # Loop trust: bila user mematuhi rekomendasi, nilai akurasi
                        # janji EstWait vs waktu tunggu aktual -> update trust.
                        # (Trip tanpa rekomendasi: last_rec_complied=False -> trust tetap.)
                        if u.last_rec_complied:
                            u.update_trust(u.est_wait_presented, u.wait_time)
                        # Tulis hasil AKTUAL balik ke entri keputusan -> satu berkas memuat
                        # janji DAN hasilnya, sehingga untung/rugi dapat dihitung.
                        self._backfill_keputusan(u)
                        self.logs.append({
                            # `step`/`time` DITAMBAHKAN 2026-08-18: tanpa penanda waktu,
                            # trip tak dapat diiris per hari -> analisis harian mustahil.
                            "step": step, "time": time_now,
                            "hari": int(time_now // 1440),
                            "user": u.user_id, "spklu": sid, "wait_time": u.wait_time,
                            "est_wait": u.est_wait_presented, "complied": u.last_rec_complied,
                            "trust_after": u.trust,
                        })
                        self.detailed_logs.append({
                            "step": step, "time": time_now,
                            "actor_ev": u.user_id, "actor_spklu": sid,
                            "event": f"[EV] {u.user_id} selesai charging di {sid}."
                        })
                        # Hook RL: sesi selesai -> emit reward wait (objektif primer) ke
                        # transisi keputusan user ini. Non-invasif (hanya bila agent punya).
                        if agent is not None and hasattr(agent, "on_charge_complete"):
                            agent.on_charge_complete(u)
                        break

            for uid, carry in newly_charging:
                for u in self.users:
                    if u.user_id == uid and u.state == UserState.QUEUING:
                        u.state = UserState.CHARGING
                        u.wait_time = max(0.0, u.wait_time - carry)
                        self.detailed_logs.append({
                            "step": step, "time": time_now,
                            "actor_ev": u.user_id, "actor_spklu": sid,
                            "event": f"[EV] {u.user_id} mulai charging di {sid} setelah nunggu {u.wait_time:.1f}m. (Masuk awal {carry:.1f}m)"
                        })
                        break

        # 3. Tangani arrival, diurutkan WAKTU TIBA AKTUAL dalam window.
        #    (Fix Bug A) overshoot lebih besar = melewati ambang lebih awal =
        #    tiba lebih dahulu => dilayani lebih dulu (FIFO yang benar).
        arrivals.sort(key=lambda u: abs(u.remaining_travel_time), reverse=True)
        for user in arrivals:
            self._handle_arrival(user, step, time_now)

        # 3b. Perbaikan T2/T3 (Validasi_Generik/LAPORAN_VALIDASI.md): sebelum EstWait
        # dihitung utk keputusan step ini, beri tiap SPKLU tahu berapa EV YANG SEDANG
        # MENUJU ke sana (termasuk yg baru saja direkomendasikan step2 sebelumnya) --
        # tanpa ini, estimate_wait_time() selalu 0 selama ada slot bebas SAAT INI,
        # padahal slot itu akan segera direbut EV yang sudah "commit" ke sana.
        self._refresh_pending_arrivals()

        # 4. Spawn EV baru (event charging_request).
        recs_this_step = []
        for spawn_tuple in self.spawn_schedule.get(step, []):
            # Unpack safely (in case old dataset without soc)
            if len(spawn_tuple) == 3:
                user, spawn_loc, soc = spawn_tuple
            else:
                user, spawn_loc = spawn_tuple
                soc = 50.0
                
            if user.state in [UserState.IDLE, UserState.DONE, UserState.RENEGED]:
                user.spawn(spawn_loc)
                user.soc = soc  # persisten sampai spawn berikutnya -- dipakai model energi durasi (Tier4 §3.6)

                # Range-aware: agen hanya MELIHAT & merekomendasikan SPKLU yang
                # terjangkau user ini (himpunan feasibel yang sama dengan kandidat
                # decide_spklu). Tanpa filter ini, agen global merekomendasikan
                # stasiun jauh yang diabaikan user -> agen jadi tak efektif di rezim lokal.
                feasible_ids = feasible_candidates(
                    user.location, soc, self.spklu_features,
                    self.user_willingness_radius_km, self.user_willingness_ratio,
                    battery_kwh=getattr(user, "battery_kwh", None) or 50.0)
                feasible_spklus = {sid: self.spklus[sid] for sid in feasible_ids}

                # Hook RL (non-invasif): ekspos user & SoC yang sedang diputuskan agar
                # agent RL bisa merakit observasi. Tak berpengaruh utk agent non-RL.
                self._current_spawn_user = user
                self._current_spawn_soc = soc

                recs = agent.get_recommendation(feasible_spklus) if agent else []
                if recs:
                    recs_this_step.append(recs[0])
                    # Hitung SEMUA stasiun yang direkomendasikan (bukan cuma recs[0]
                    # primer) -- ditemukan bug: rekomendasi sekunder (mis. slot ke-2
                    # pada top-K/PDQN k=2) sebelumnya tak pernah masuk recent_recs,
                    # membuat sinyal anti-herding (rec_activity, flock_reward_rolling)
                    # buta thd herding pada slot sekunder.
                    for sid in recs:
                        self.recent_recs[sid] += 1

                # EstWait yang dijanjikan kini TUGAS AGEN (akurasi = properti agen,
                # bisa dipelajari RL), diprediksi atas SPKLU feasibel. Fallback ke
                # estimasi deterministik Simulator bila agen tak punya prediktor.
                if agent is not None and hasattr(agent, "predict_waits"):
                    est_waits = self._call_predict_waits(agent, feasible_spklus, self, user, time_now)
                else:
                    est_waits = self.get_spklu_wait_estimates()
                queue_lengths = {sid: s.get_queue_length() for sid, s in self.spklus.items()}
                chosen_spklu_id = user.decide_spklu(recs, est_waits, self.spklu_features, soc_percent=soc,
                                                    willingness_radius_km=self.user_willingness_radius_km,
                                                    willingness_ratio=self.user_willingness_ratio,
                                                    queue_lengths=queue_lengths)

                # Hook RL: umpan-balik keputusan (kepatuhan) utk atribusi reward R_accept.
                if agent is not None and hasattr(agent, "on_decision"):
                    agent.on_decision(user, chosen_spklu_id, recs, feasible_spklus)

                self.detailed_logs.append({
                    "step": step, "time": time_now,
                    "user_id": user.user_id,
                    "event": "spawn",
                    "location": spawn_loc,
                    "soc": soc,
                    "target_spklu": chosen_spklu_id,
                })

                # --- Jejak per-keputusan (append murni; tidak menyentuh RNG) ---
                # `trust_effective` yang dicatat, BUKAN `trust` mentah: itulah yang
                # benar-benar menggerakkan keputusan (di rezim beku keduanya berbeda).
                def _jarak(sid):
                    loc = self.spklu_features.get(sid, {}).get("loc")
                    if loc is None or spawn_loc is None:
                        return None
                    return math.dist(spawn_loc, loc)

                jarak_feasible = {sid: _jarak(sid) for sid in feasible_ids}
                terdekat = min((v, k) for k, v in jarak_feasible.items() if v is not None) \
                    if any(v is not None for v in jarak_feasible.values()) else (None, None)
                d_terdekat, sid_terdekat = terdekat
                d_pilih = _jarak(chosen_spklu_id) if chosen_spklu_id else None
                d_rec = _jarak(recs[0]) if recs else None
                self.decision_log.append({
                    "step": step, "time": time_now, "user_id": user.user_id,
                    "trust": float(getattr(user, "trust_effective", user.trust)),
                    "recs": list(recs), "chosen": chosen_spklu_id,
                    "patuh": bool(recs) and chosen_spklu_id in recs,
                    "n_feasible": len(feasible_ids),
                    "sid_terdekat": sid_terdekat,
                    "d_terdekat": d_terdekat, "d_pilih": d_pilih, "d_rec": d_rec,
                    # Seberapa jauh pengguna BERGESER dari stasiun terdekatnya, dan
                    # seberapa jauh sistem MENAWARKAN pergeseran itu. Selisih keduanya
                    # memisahkan "didorong jauh" dari "mau bergerak jauh".
                    "dorong_pilih": (None if None in (d_pilih, d_terdekat)
                                     else d_pilih - d_terdekat),
                    "dorong_rec": (None if None in (d_rec, d_terdekat)
                                   else d_rec - d_terdekat),
                    "est_pilih": float(est_waits.get(chosen_spklu_id, float("nan"))),
                    "est_terdekat": float(est_waits.get(sid_terdekat, float("nan")))
                    if sid_terdekat else float("nan"),
                    "hari": int(time_now // 1440), "jam_hari": int(time_now // 60) % 24,
                    # Peringkat stasiun terpilih di dalam daftar rekomendasi: 0 = primer,
                    # 1 = sekunder, -1 = menolak. Menjawab apakah slot sekunder terpakai.
                    "peringkat_pilih": (recs.index(chosen_spklu_id)
                                        if chosen_spklu_id in recs else -1),
                    # --- PENGARUH JANJI TERHADAP KEPUTUSAN (kontrafaktual) ---
                    # `sid_pref` = stasiun yang AKAN dipilih pengguna tanpa rekomendasi
                    # (argmax utilitas pribadi). Tersedia utk SEMUA agen krn
                    # `decide_spklu` selalu menghitung `last_u_pref`.
                    **self._jejak_pengaruh_janji(user, chosen_spklu_id, est_waits),
                    # Kondisi SELURUH stasiun feasible saat keputusan -- tanpa ini tak
                    # dapat dinilai apakah rekomendasi memang lebih buruk bagi individu,
                    # atau justru pilihan mandiri pengguna yang lebih buruk.
                    "est_feasible": {sid: float(est_waits.get(sid, float("nan")))
                                     for sid in feasible_ids},
                    "queue_feasible": {sid: int(self.spklus[sid].get_queue_length())
                                       for sid in feasible_ids},
                })
            else:
                # (Fix Bug C) User masih sibuk -> TUNDA event ke step berikutnya,
                # jangan dibuang (mencegah demand undercount, terutama saat beban tinggi).
                self.spawn_schedule.setdefault(step + 1, []).append((user, spawn_loc, soc))

        # 5. Deteksi herding (Fix Bug H): >=3 agen ke SPKLU yang sama, sesuai
        #    definisi Herding Index di Rancangan Tahap 2.2.
        if recs_this_step:
            from collections import Counter
            rec_counts = Counter(recs_this_step)
            # Peningkatan beban puncak per stasiun target dlm window ini (dipakai
            # flocking_index() -- lihat marl_spklu/experiments/metrics.py). Diukur
            # sesudah arrival+spawn+charging window ini selesai diproses (di atas),
            # jadi queue_after mencerminkan efek window penuh, bukan snapshot parsial.
            queue_after_window = {sid: s.get_queue_length() for sid, s in self.spklus.items()}
            queue_delta = {sid: max(0.0, queue_after_window[sid] - queue_before_window.get(sid, 0.0))
                          for sid in rec_counts}
            # Rekam distribusi rekomendasi step ini -> basis Recommendation Entropy
            # dan Herding Index ternormalisasi (rasio window herding / total window rec).
            self.rec_distribution_log.append({
                "step": step, "time": time_now,
                "counts": dict(rec_counts), "n_recs": len(recs_this_step),
                "queue_delta": queue_delta,
            })
            for sid, count in rec_counts.items():
                if count >= 3:
                    self.herding_events += 1
                    self.detailed_logs.append({
                        "step": step, "time": time_now,
                        "actor_ev": "SYSTEM", "actor_spklu": sid,
                        "event": f"[HERDING] {count} EV direkomendasikan serempak ke {sid}"
                    })

        # 6. Rekam utilisasi instan ke history buffer tiap step (G1a).
        #    Menyuplai komponen historis state agen (util_7d, slope_Gini).
        if self.history is not None:
            self.history.record_utilization(
                {sid: s.get_utilization() for sid, s in self.spklus.items()}
            )

        # 7. (Opsional) Rekam snapshot state SETIAP aktor (User + SPKLU) pada step
        #    ini. Dipanggil di akhir step agar semua transisi state sudah selesai.
        if self.log_actor_states and (step % self.log_every == 0):
            self._record_actor_states(step, time_now)

        # 8. (Opsional) Deret waktu: snapshot per JAM per stasiun + per HARI tingkat
        #    jaringan. Dipanggil paling akhir supaya seluruh transisi step ini selesai.
        if self.rekam_deret:
            self._rekam_deret_waktu(step, time_now)

    # -------------------------------------------------- pengaruh janji thd keputusan
    def _jejak_pengaruh_janji(self, user, chosen_spklu_id, est_waits):
        """Seberapa jauh JANJI waktu tunggu menggeser keputusan, dan apakah pengguna
        UNTUNG atau RUGI karenanya.

        `decide_spklu` mencampur dua kanal: P = (1-T)*P_pref + T*P_rec. Kontrafaktual yang
        relevan adalah keputusan pada kanal preferensi SAJA (T=0) -- yaitu apa yang akan
        dipilih pengguna seandainya tak ada rekomendasi. Itu tersedia dari `last_u_pref`
        yang selalu dihitung, sehingga ukuran ini berlaku utk greedy dan S0 juga, bukan
        hanya lengan RL.

        Yang direkam:
          sid_pref      stasiun kontrafaktual (argmax utilitas pribadi)
          est_pref      janji waktu tunggu DI stasiun kontrafaktual itu
          est_pilih     janji di stasiun yang benar-benar dipilih
          untung_janji  est_pref - est_pilih; POSITIF = janji mengarahkan ke stasiun yang
                        DIJANJIKAN lebih cepat drpd pilihan alaminya
          p_pilih_pref  peluang memilih stasiun itu tanpa rekomendasi
          p_pilih_akhir peluang setelah dicampur rekomendasi
          geser_p       selisihnya -- besar pengaruh janji pada keputusan INI

        CATATAN: `untung_janji` bersifat EX-ANTE (janji vs janji). Untung/rugi EX-POST
        memerlukan waktu tunggu AKTUAL, yang baru diketahui saat sesi selesai -- di-backfill
        ke `wait_aktual` oleh `_backfill_keputusan`.
        """
        ids = getattr(user, "last_candidate_ids", None)
        up = getattr(user, "last_u_pref", None)
        fp = getattr(user, "last_final_probs", None)
        if not ids or up is None or len(up) != len(ids):
            return {}
        up = np.asarray(up, dtype=float)
        tau = max(float(getattr(user, "tau", 1.0)), 1e-9)
        e = np.exp(up / tau - np.max(up / tau))
        p_pref = e / e.sum()
        i_pref = int(np.argmax(up))
        sid_pref = str(ids[i_pref])
        try:
            i_pilih = ids.index(chosen_spklu_id)
        except (ValueError, AttributeError):
            i_pilih = None
        est_pref = float(est_waits.get(sid_pref, float("nan")))
        est_pilih = float(est_waits.get(chosen_spklu_id, float("nan")))
        jejak = {
            "sid_pref": sid_pref, "est_pref": est_pref,
            "untung_janji": est_pref - est_pilih,
            "pilih_sama_pref": bool(chosen_spklu_id == sid_pref),
            "p_pilih_pref": float(p_pref[i_pilih]) if i_pilih is not None else float("nan"),
            "p_pilih_akhir": (float(np.asarray(fp)[i_pilih])
                              if (fp is not None and i_pilih is not None) else float("nan")),
            "wait_aktual": None,     # di-backfill saat sesi selesai
        }
        jejak["geser_p"] = jejak["p_pilih_akhir"] - jejak["p_pilih_pref"]
        # Indeks entri ini supaya hasil aktualnya bisa ditulis balik nanti.
        user._idx_keputusan = len(self.decision_log)
        return jejak

    def _backfill_keputusan(self, user):
        """Tulis waktu tunggu AKTUAL ke entri keputusan pengguna ini.

        Tanpa ini, `decision_log` hanya memuat janji (ex-ante) dan tak pernah tahu apakah
        janji itu ditepati -- sehingga pertanyaan "apakah pengguna untung atau rugi karena
        mengikuti janji" tak dapat dijawab dari satu berkas.
        """
        i = getattr(user, "_idx_keputusan", None)
        if i is None or not (0 <= i < len(self.decision_log)):
            return
        e = self.decision_log[i]
        if e.get("user_id") != user.user_id:
            return
        e["wait_aktual"] = float(user.wait_time)
        ep = e.get("est_pref")
        # EX-POST: janji di stasiun kontrafaktual vs yang BENAR-BENAR dialami.
        # Membandingkan prediksi (alternatif) dgn realisasi (pilihan) -- satu-satunya cara
        # tanpa menjalankan dunia tandingan; harus dilaporkan sbg perkiraan, bukan ukuran
        # eksak.
        e["untung_expost"] = (None if ep is None or not np.isfinite(ep)
                              else float(ep - user.wait_time))
        user._idx_keputusan = None

    # ------------------------------------------------------------------ deret waktu
    def _rekam_deret_waktu(self, step, time_now):
        """Snapshot berkala. Per JAM utk stasiun (bukan per step: 15 mnt x 6 stasiun x 90
        hari = 51.840 baris, terlalu besar utk manfaat yg sama)."""
        jam = int(time_now // 60)
        hari = int(time_now // 1440)

        if jam != self._log_terakhir_jam:
            self._log_terakhir_jam = jam
            for sid, s in self.spklus.items():
                self.station_log.append({
                    "step": step, "time": time_now, "hari": hari, "jam_hari": jam % 24,
                    "spklu": sid,
                    "queue": s.get_queue_length(),
                    "n_charging": sum(len(v) for v in s.charging.values())
                    if isinstance(s.charging, dict) else len(s.charging),
                    "utilisasi": float(s.get_utilization()),
                    "served_kum": int(s.total_served),
                    "wait_kum": float(s.total_wait_time),
                })

        if hari != self._log_terakhir_hari:
            self._log_terakhir_hari = hari
            self.daily_log.append(self._ringkas_hari(hari, time_now))

    def _ringkas_hari(self, hari, time_now):
        """Ringkasan tingkat jaringan utk hari yang BARU SAJA selesai (hari-1).

        Menyediakan lintasan Gini/wait/acceptance/trust tanpa perlu menjalankan simulasi
        berulang -- sebelumnya lintasan trust dihasilkan skrip terpisah yang menjalankan
        ulang seluruh simulasi utk tiap titik waktu.
        """
        h = hari - 1
        served = np.array([s.total_served for s in self.spklus.values()], dtype=float)
        # Served KUMULATIF -> Gini kumulatif; served HARIAN -> Gini hari itu saja.
        harian = served - np.array([self._served_hari_lalu.get(sid, 0.0)
                                    for sid in self.spklus], dtype=float)
        self._served_hari_lalu = {sid: float(s.total_served)
                                  for sid, s in self.spklus.items()}

        trip = [l for l in self.logs if l.get("hari") == h]
        w = np.array([l["wait_time"] for l in trip], dtype=float)
        c = np.array([bool(l["complied"]) for l in trip], dtype=bool)
        tr = np.array([u.trust for u in self.users], dtype=float)

        def _gini(x):
            x = np.sort(np.asarray(x, dtype=float))
            n = x.size
            if n == 0 or x.sum() <= 0:
                return 0.0
            return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

        return {
            "hari": h, "time": time_now,
            "gini_kumulatif": _gini(served),
            "gini_harian": _gini(harian),
            "served_harian": float(harian.sum()),
            "n_trip": len(trip),
            "wait_mean": float(w.mean()) if w.size else float("nan"),
            "wait_p50": float(np.percentile(w, 50)) if w.size else float("nan"),
            "wait_p90": float(np.percentile(w, 90)) if w.size else float("nan"),
            "acceptance": float(c.mean()) if c.size else float("nan"),
            "trust_mean": float(tr.mean()), "trust_p10": float(np.percentile(tr, 10)),
            "trust_min": float(tr.min()),
        }

    def ringkas_pengguna(self):
        """Ringkasan PER PENGGUNA, dihitung di akhir run (bukan direkam tiap step).

        Menjawab pertanyaan yang tak dapat dijawab metrik agregat: apakah beban pemerataan
        ditanggung merata, atau ditimpakan ke sebagian kecil pengguna. Gini stasiun yang
        membaik sementara ketimpangan antar-PENGGUNA memburuk adalah hasil yang berbeda
        maknanya -- dan sebelumnya tak terdeteksi sama sekali.
        """
        per_user = defaultdict(lambda: {"wait": [], "spklu": [], "patuh": 0, "n": 0,
                                        "hari_pertama": None})
        for l in self.logs:
            d = per_user[l["user"]]
            d["n"] += 1
            d["wait"].append(l["wait_time"])
            d["spklu"].append(l["spklu"])
            d["patuh"] += 1 if l["complied"] else 0
            hh = l.get("hari")
            if hh is not None and (d["hari_pertama"] is None or hh < d["hari_pertama"]):
                d["hari_pertama"] = hh

        dorong = defaultdict(list)
        for e in self.decision_log:
            if e.get("dorong_pilih") is not None:
                dorong[e["user_id"]].append(e["dorong_pilih"])

        out = []
        for u in self.users:
            d = per_user.get(u.user_id)
            n = d["n"] if d else 0
            wait = np.array(d["wait"], dtype=float) if n else np.array([])
            sids = d["spklu"] if n else []
            unik, cacah = np.unique(sids, return_counts=True) if sids else ([], [])
            p = cacah / cacah.sum() if len(cacah) else np.array([])
            entropi = float(-(p * np.log2(p + 1e-12)).sum()) if p.size else 0.0
            dd = np.array(dorong.get(u.user_id, []), dtype=float)
            out.append({
                "user_id": u.user_id, "n_trip": n,
                "n_patuh": d["patuh"] if n else 0,
                "rasio_patuh": (d["patuh"] / n) if n else float("nan"),
                "trust_akhir": float(u.trust),
                "wait_mean": float(wait.mean()) if wait.size else float("nan"),
                "wait_maks": float(wait.max()) if wait.size else float("nan"),
                "n_spklu_unik": int(len(unik)),
                "entropi_spklu": entropi,
                "dorong_mean": float(dd.mean()) if dd.size else float("nan"),
                "hari_pertama": d["hari_pertama"] if n else None,
            })
        return out

        # Hook RL: akhir step -> agen RL hitung Phi & bagikan shaping ΔPhi ke transisi
        # yang direkam pada step ini. Non-invasif (hanya jika agent punya on_step_end).
        if agent is not None and hasattr(agent, "on_step_end"):
            agent.on_step_end(self, step)

    def _record_actor_states(self, step: int, time_now: float):
        """Rekam snapshot state tiap aktor pada satu step ke actor_state_log
        (User) dan spklu_state_log (SPKLU). Aktif hanya bila log_actor_states=True."""
        # --- SPKLU ---
        for sid, s in self.spklus.items():
            self.spklu_state_log.append({
                "step": step, "time": time_now, "spklu_id": sid,
                "utilization": s.get_utilization(),
                "queue_total": s.get_queue_length(),
                "charging_total": sum(len(c) for c in s.charging.values()),
                "queues": {ct: len(q) for ct, q in s.queues.items()},
                "charging": {ct: len(c) for ct, c in s.charging.items()},
                "total_served": s.total_served,
                "total_wait_time": s.total_wait_time,
            })
        # --- User ---
        for u in self.users:
            if self.log_active_only and u.state in (UserState.IDLE, UserState.DONE):
                continue
            loc = u.location if u.location is not None else (None, None)
            self.actor_state_log.append({
                "step": step, "time": time_now, "user_id": u.user_id,
                "segment": u.segment, "state": u.state,
                "x": loc[0], "y": loc[1],
                "target_spklu": u.target_spklu,
                "wait_time": u.wait_time,
                "trust": u.trust, "lcmnl_class": u.lcmnl_class,
            })

    def export_actor_logs(self, path_prefix: str):
        """Tulis trace per-aktor ke dua berkas JSON: <prefix>_users.json &
        <prefix>_spklus.json. Berguna untuk visualisasi/analisis pasca-run."""
        import json
        with open(f"{path_prefix}_users.json", "w") as f:
            json.dump(self.actor_state_log, f)
        with open(f"{path_prefix}_spklus.json", "w") as f:
            json.dump(self.spklu_state_log, f)
        return f"{path_prefix}_users.json", f"{path_prefix}_spklus.json"

    def compute_virtual_wait(self, user, spklu, current_time):
        """Hitung waktu tunggu hipotetis tanpa mengubah state."""
        # 1. Tentukan konektor yang kompatibel
        if user is not None and getattr(user, "connector_types", None) is not None:
            compatible_types = [ct for ct in spklu.capacities if spklu.capacities[ct] > 0 and ct in user.connector_types]
        else:
            compatible_types = [ct for ct in spklu.capacities if spklu.capacities[ct] > 0]
            
        if not compatible_types:
            # Tak ada konektor kompatibel sama sekali (mis. EV DC-only, stasiun cuma AC
            # atau n_conn_dc=0) -- sentinel besar, BUKAN 0.0 (dulu bug: membuat stasiun
            # yg sama sekali tak bisa dipakai tampak paling menarik/wait instan).
            return UNREACHABLE_WAIT_MINUTES

        # Tentukan travel time user ke SPKLU
        if user is not None and user.location is not None:
            dist_km = math.dist(user.location, spklu.location)
        else:
            dist_km = 0.0
        t_travel = (dist_km / 40.0) * 60.0
        user_arrival_time = current_time + t_travel

        # Temukan semua traveling EVs yang sedang menuju ke SPKLU ini
        traveling_evs = [
            u for u in self.users
            if u.state == UserState.TRAVELING and u.target_spklu == spklu.spklu_id and (user is None or u.user_id != user.user_id)
        ]

        from marl_spklu.env.spklu import mean_charge_time
        
        c_type_waits = []
        for c_type in compatible_types:
            cap = spklu.capacities[c_type]
            # Salin ketersediaan slot awal berdasarkan EV yang sedang charging
            slot_availability = []
            for ev in spklu.charging.get(c_type, []):
                slot_availability.append(current_time + ev["remaining_time"])
            # Sisanya adalah slot kosong yang langsung tersedia pada current_time
            while len(slot_availability) < cap:
                slot_availability.append(current_time)
                
            # Kumpulkan semua EV lain yang mengantre/menuju tipe konektor ini
            # a. Antrean saat ini (sudah sampai di stasiun)
            other_evs = []
            for q_uid in spklu.queues.get(c_type, []):
                other_evs.append({
                    "arrival_time": current_time,
                    "charge_dur": mean_charge_time(c_type)
                })
            # b. EV yang sedang menuju stasiun dan kompatibel dengan c_type
            for t_ev in traveling_evs:
                if c_type in t_ev.connector_types:
                    other_evs.append({
                        "arrival_time": current_time + t_ev.remaining_travel_time,
                        "charge_dur": mean_charge_time(c_type)
                    })
                    
            # Urutkan secara FIFO berdasarkan arrival_time
            other_evs.sort(key=lambda x: x["arrival_time"])
            
            # Simulasikan komitmen antrean/layanan untuk other_evs
            for ev in other_evs:
                earliest_slot_idx = min(range(len(slot_availability)), key=lambda j: slot_availability[j])
                start_time = max(ev["arrival_time"], slot_availability[earliest_slot_idx])
                slot_availability[earliest_slot_idx] = start_time + ev["charge_dur"]
                
            # Terakhir, sisipkan user kita
            earliest_slot_idx = min(range(len(slot_availability)), key=lambda j: slot_availability[j])
            start_time = max(user_arrival_time, slot_availability[earliest_slot_idx])
            virtual_wait = max(0.0, start_time - user_arrival_time)
            c_type_waits.append(virtual_wait)
            
        return min(c_type_waits) if c_type_waits else 0.0
