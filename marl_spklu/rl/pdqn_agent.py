"""Agent adapter PDQN -- menjembatani PDQNQNetwork dgn Simulator lewat hook yang sama
persis dgn RLRolloutAgent (get_recommendation/predict_waits/on_decision/on_step_end/
on_charge_complete), sehingga plug-compatible dgn harness.compare_scenarios.

Mengikuti Lin et al. 2024 (IEEE TCE 70(3):5238-5250), lihat pdqn_policy.py:
  * Riwayat yang diumpankan ke preference extractor adalah PASANGAN AKSI (a_hat, a)
    milik PENGGUNA YANG SAMA -- a_hat = SPKLU direkomendasikan, a = SPKLU dipilih.
    Disimpan per-pengguna di agent (`self._pref_hist`), BUKAN di User.interaction_history
    (yang formatnya skalar patuh/estwait milik jalur H-PPO, tanpa identitas stasiun).
  * Aksi diskrit tunggal a_hat -> `get_recommendation` mengembalikan list SATU elemen,
    shg User.decide_spklu membentuk rec_set={a_hat} (setara f_rec_binary, adapter M-B3).

Reward memakai RewardCalculator yang sudah berjalan (rewards.py), BUKAN r=1/(tau_tr+
tau_qu+tau_ch) milik paper -- sesuai instruksi: tujuan Tahap 0 adalah pemerataan
utilisasi, bukan minimasi waktu layanan seperti objektif asli paper.

REDESAIN v2 (2026-08-13): agen merekomendasikan N_REC=2 SPKLU/keputusan, Q-network
menilai stasiun secara INDEPENDEN (bukan kombinasi gabungan -- v1 kombinasi 15-aksi
terbukti memburuk, lihat pdqn_policy.py).

REDESAIN v3 (2026-08-13, gaya dekomposisi-nilai/VDN): percobaan v2 "sibling transition"
(dua transisi terpisah per keputusan, reward IDENTIK disalin ke keduanya) TERBUKTI GAGAL
juga (Gini memburuk monoton sepanjang training) -- didiagnosis: dua TD-error terpisah
dgn target SAMA `y` memaksa Q(s,i) DAN Q(s,j) menuju nilai ABSOLUT yg sama, menghapus
diferensiasi yg seharusnya dipelajari (lihat memori/diskusi root-cause).

v3 kembali ke SATU `PDQNTransition` per keputusan (`action_idx` = PASANGAN (i,j), bukan
skalar). Q gabungan Q_tot(s) = Q(s,i) + Q(s,j) (dijumlah, bukan diduplikasi) -> SATU
TD-error target y = r + gamma*max_{a,b} [Q_target(s',a)+Q_target(s',b)]. Gradiennya
sama besar utk Q(s,i) & Q(s,j) (turunan Q_tot thd tiap komponen = 1), TAPI yg dipaksa
benar hanya JUMLAHNYA -- Q(s,i) & Q(s,j) bebas berbeda selama totalnya cocok, tak
dipaksa collapse ke nilai sama spt v2. (Analog Value Decomposition Networks di MARL
kooperatif -- tak sepenuhnya selesaikan atribusi kredit individual, tapi menghindari
patologi v2 yg lebih parah.)
"""
import math
from collections import deque

import numpy as np

from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.pdqn_policy import (PDQN_HIST_K, N_REC, hist_feat_dim,
                                       hist_feat_dim_feature, PREF_STATION_FEAT_DIM)

# Riwayat LSTM menyimpan PDQN_HIST_K*N_REC sub-langkah (2 pasangan per keputusan:
# (rekomendasi_1,pilihan) & (rekomendasi_2,pilihan)) -- TAK berubah dari redesain v1.
HIST_LEN = PDQN_HIST_K * N_REC


class PDQNTransition:
    """Transisi DQN untuk SATU keputusan (v3: `action_idx` = PASANGAN (i,j), Q_tot =
    Q(s,i)+Q(s,j) -- lihat docstring modul). SATU transisi per keputusan (bukan dua).

    reward terkumpul bertahap: R_global tiap langkah (on_step_end) + wait/honesty saat
    sesi charging selesai (on_charge_complete) -> `resolved=True`. `pushed` menandai
    transisi sudah dimasukkan ke replay buffer sbg pasangan (s,a,r,s') oleh trainer."""
    __slots__ = ("obs", "mask", "hist", "pref_type", "action_idx", "step", "reward", "done",
                 "complied", "disp_estwait", "wait_default", "resolved", "pushed",
                 "flock_penalty")

    def __init__(self, obs, mask, hist, action_idx, step, pref_type=0):
        self.obs = obs; self.mask = mask; self.hist = hist
        self.pref_type = int(pref_type)
        self.action_idx = action_idx; self.step = step   # action_idx: tuple (i, j)
        self.reward = 0.0; self.done = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.resolved = False; self.pushed = False; self.flock_penalty = 0.0


class PDQNRolloutAgent:
    """Agent yang dipasang ke Simulator selama training PDQN: merekam transisi diskrit,
    reward, dan riwayat pasangan aksi (a_hat, a) per pengguna."""

    def __init__(self, q_net, sim, reward_calc, forecaster=None, epsilon: float = 0.1,
                 record: bool = True, pref_feature_mode: bool = False):
        self.q_net = q_net
        self.sim = sim
        self.rc = reward_calc
        self.forecaster = forecaster or FormulaForecaster()
        self.epsilon = float(epsilon)
        # record=False (inferensi): transisi/reward tak dipakai -> jangan diakumulasi
        # (riwayat preferensi _pref_hist TETAP dicatat, LSTM membutuhkannya).
        self.record = bool(record)
        # pref_feature_mode=True: pasangan (a_hat,a) sbg VEKTOR FITUR stasiun, bukan
        # one-hot identitas -- lihat pdqn_policy.py::hist_feat_dim_feature.
        self.pref_feature_mode = bool(pref_feature_mode)
        # Mode reward PEMERATAAN (EquityRewardCalculator): seluruh reward diketahui SEGERA
        # di on_decision -> transisi langsung resolved, tak ada suku global per-langkah
        # maupun suku wait yang tertunda sampai sesi charging selesai.
        self._equity_mode = hasattr(reward_calc, "decision_reward_equity")

        self.sids = list(sim.spklus.keys())
        self.sid_to_idx = {s: i for i, s in enumerate(self.sids)}
        self.N = len(self.sids)
        self.hist_dim = (hist_feat_dim_feature() if self.pref_feature_mode
                         else hist_feat_dim(self.N))

        xs = np.array([sim.spklus[s].location[0] for s in self.sids], dtype=float)
        ys = np.array([sim.spklus[s].location[1] for s in self.sids], dtype=float)
        self.pos_scale = max(1e-6, float(np.std(np.concatenate([xs, ys]))))
        self.dist_scale = 10.0
        self.wait_scale = 60.0
        self.q_scale = 5.0
        self.conn_scale = 5.0
        self.battery_scale = 80.0
        self.rec_activity_scale = 10.0   # sama dgn rollout.py (H-PPO), skala jendela 24 jam

        self.transitions = []          # URUT keputusan (dipakai trainer utk pasangan s->s')
        self._pending = None           # (transisi, action_idx, user) keputusan berjalan
        self._user_trip_tr = {}        # user_id -> transisi trip berjalan (reward delayed)
        # Riwayat preferensi per pengguna: barisan pasangan (a_hat_idx, a_idx).
        self._pref_hist = {}

    # ---------------- Preference extraction input (phi: (a_hat,a) -> p) ----------------
    def _pref_station_feat(self, sid, user, wait_hat):
        """Vektor fitur (PREF_STATION_FEAT_DIM=5) 1 stasiun utk pasangan preferensi mode
        FITUR: [jarak, est_wait, antrean, konektor, utilisasi] -- identik semantik
        rollout.py::RLRolloutAgent._pref_station_feat, disamakan agar PDQN diskrit dan
        kontinu bisa dibandingkan apple-to-apple bila pref_feature_mode diaktifkan."""
        feat = self.sim.spklu_features.get(sid, {})
        loc = feat.get('loc', (0.0, 0.0))
        dist = math.dist(user.location, loc) / self.dist_scale
        wait = wait_hat.get(sid, 0.0) / self.wait_scale
        q = self.sim.spklus[sid].get_queue_length() / self.q_scale
        conn = feat.get('conn', 1.0) / self.conn_scale
        util = self.sim.spklus[sid].get_utilization()
        return np.array([dist, wait, q, conn, util], dtype=np.float32)

    def _build_hist(self, user):
        """Riwayat HIST_LEN=PDQN_HIST_K*N_REC sub-langkah pasangan (a_hat_i, a) terakhir
        pengguna ini -> (HIST_LEN, dim) left-padded nol -- one-hot ganda (default, paper
        §IV.A, diperluas N_REC pasangan/keputusan) atau vektor fitur (pref_feature_mode).
        REDESAIN: tiap keputusan menyumbang N_REC=2 sub-langkah berurutan (satu per
        anggota kombinasi yg direkomendasikan, dibandingkan thd pilihan SAMA pengguna)."""
        arr = np.zeros((HIST_LEN, self.hist_dim), dtype=np.float32)
        h = self._pref_hist.get(user.user_id)
        if h:
            recent = list(h)[-HIST_LEN:]
            offset = HIST_LEN - len(recent)
            for t, pair in enumerate(recent):
                if self.pref_feature_mode:
                    arr[offset + t] = pair
                else:
                    a_hat_idx, a_idx = pair
                    arr[offset + t, a_hat_idx] = 1.0            # one-hot(a_hat)
                    arr[offset + t, self.N + a_idx] = 1.0       # one-hot(a)
        return arr

    def _record_pref(self, user, a_hat_idx, a_idx, wait_hat=None):
        h = self._pref_hist.get(user.user_id)
        if h is None:
            h = deque(maxlen=HIST_LEN)
            self._pref_hist[user.user_id] = h
        if self.pref_feature_mode:
            feat_hat = self._pref_station_feat(self.sids[a_hat_idx], user, wait_hat or {})
            feat_a = self._pref_station_feat(self.sids[a_idx], user, wait_hat or {})
            h.append(np.concatenate([feat_hat, feat_a]))
        else:
            h.append((int(a_hat_idx), int(a_idx)))

    # ---------------- Observasi state ----------------
    def _build_obs(self, user, soc, feasible_ids, time_now):
        """[scalars(6), onehot_prev(N), dist(N), wait_hat(N), queue(N), conn(N), util(N),
        rec_activity(N)]. Blok util = D_hat (utilisasi ternormalisasi per SPKLU).

        Blok rec_activity (BARU, opsi #1 "bagaimana agar PDQN menang"): jumlah rekomendasi
        ke SPKLU ini dlm jendela 24 jam berjalan (`sim.recent_recs`, sama persis dipakai
        critic_obs H-PPO di rollout.py -- SEBELUMNYA tak pernah diumpankan ke PDQN).
        Ini sinyal yg Greedy TAK MUNGKIN dimiliki krn ia stateless antar-rekomendasi:
        bila beberapa permintaan muncul dlm jendela waktu yg sama, Greedy mengirim SEMUA
        ke SPKLU yg tampak paling sepi (blind spot overshoot -- lihat contoh motivasi
        Fig.1 paper Lin et al.). PDQN yg melihat blok ini BISA belajar menghindari
        menumpuk ke SPKLU yg baru saja ramai direkomendasikan, sesuatu yg secara
        struktural tak bisa ditutup aturan argmin-utilisasi-saat-ini."""
        loc = user.location
        wait_hat = self.forecaster.predict(self.sim.spklus, time_now, user=user, soc=soc, sim=self.sim)
        onehot = np.zeros(self.N, dtype=float)
        if user.prev_spklu in self.sid_to_idx:
            onehot[self.sid_to_idx[user.prev_spklu]] = 1.0
        d = np.array([math.dist(loc, self.sim.spklus[s].location) for s in self.sids]) / self.dist_scale
        wv = np.array([wait_hat.get(s, 0.0) for s in self.sids]) / self.wait_scale
        q = np.array([self.sim.spklus[s].get_queue_length() for s in self.sids]) / self.q_scale
        conn = np.array([self.sim.spklu_features.get(s, {}).get('conn', 1.0) for s in self.sids]) / self.conn_scale
        util = np.array([self.sim.spklus[s].get_utilization() for s in self.sids])
        rec_act = np.array([self.sim.recent_recs.get(s, 0) for s in self.sids]) / self.rec_activity_scale

        hour = (time_now / 60.0) % 24.0
        sin_hour, cos_hour = math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)
        battery_norm = user.battery_capacity_kwh / self.battery_scale
        scalars = np.array([soc / 100.0, loc[0] / self.pos_scale, loc[1] / self.pos_scale,
                            sin_hour, cos_hour, battery_norm])

        # "Stasiun default" pengguna (P_pref murni, tanpa suku wait) -- pembanding wait
        # pada reward. U(s) = w1*dist+w2*log_pop+w3*log_conn+w4*isPrev+w5*soc_urgency
        # (Model_Simulasi_Inti.md §3.1), w1-w5 dari kelas Latent Class MNL per-user.
        u_def = np.full(self.N, -np.inf)
        for j, sid in enumerate(self.sids):
            if sid not in feasible_ids:
                continue
            feat = self.sim.spklu_features.get(sid, {})
            tgt = feat.get('loc', (0.0, 0.0))
            dk = math.dist(loc, tgt)
            pop = feat.get('pop', 1.0); conn_j = feat.get('conn', 1.0)
            is_prev = 1.0 if sid == user.prev_spklu else 0.0
            soc_urgency = (1.0 - soc / 100.0) * dk
            u_def[j] = (user.w1 * dk + user.w2 * math.log(1 + pop) +
                        user.w3 * math.log(1 + conn_j) + user.w4 * is_prev +
                        user.w5 * soc_urgency)
        default_idx = int(np.argmax(u_def)) if np.isfinite(u_def).any() else 0
        obs = np.concatenate([scalars, onehot, d, wv, q, conn, util, rec_act]).astype(np.float32)
        return obs, default_idx, wait_hat

    def _feasible_mask(self, feasible_ids):
        m = np.zeros(self.N, dtype=bool)
        for s in feasible_ids:
            if s in self.sid_to_idx:
                m[self.sid_to_idx[s]] = True
        return m

    # ---------------- Hook Simulator ----------------
    def get_recommendation(self, feasible_spklus: dict):
        """v3: Q INDEPENDEN per stasiun, `act()` kembalikan PASANGAN (i,j) top-2 --
        SATU keputusan menghasilkan SATU `PDQNTransition` dgn `action_idx=(i,j)` (Q_tot
        = Q(s,i)+Q(s,j), lihat docstring modul)."""
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        obs, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        mask = self._feasible_mask(feasible_ids)
        hist = self._build_hist(user)
        # Tipe preferensi memilih LSTM mana yang dipakai (paper: satu LSTM per tipe).
        # None utk dataset tanpa tipe -> jaringan memakai LSTM tunggal.
        ptype = getattr(user, "pref_type", None)
        act = self.q_net.act(obs, mask, hist, epsilon=self.epsilon, pref_type=ptype)
        i, j = act["combo"]                        # pasangan indeks stasiun (top-2 Q)
        sids_chosen = [self.sids[i], self.sids[j]]
        # Dibaca SEBELUM Simulator menaikkan recent_recs (step_once memanggil
        # get_recommendation lalu BARU menambah hitungan) -> "seberapa ramai PASANGAN
        # ini direkomendasikan SEBELUM keputusan ini" (rata2 keduanya), dipakai suku
        # anti-overshoot/anti-herding.
        recent_rec_count = float(np.mean(
            [self.sim.recent_recs.get(sid, 0) for sid in sids_chosen]))

        estimated_waits = {sid: float(wait_hat.get(sid, 0.0)) for sid in feasible_ids}
        # "Primer" = anggota PERTAMA (nilai-Q tertinggi) -- dipakai sbg acuan tunggal utk
        # bookkeeping wait_default (disp_estwait sendiri sudah tak memengaruhi reward
        # sejak alpha_honesty dihapus, murni diagnostik).
        disp_estwait = estimated_waits.get(sids_chosen[0], 0.0)

        if self.record:
            tr = PDQNTransition(obs, mask, hist, (i, j), self.sim.current_step,
                                pref_type=(ptype if ptype is not None else 0))
            default_in_combo = default_idx in (i, j)
            wait_default_val = float(self.sim.compute_virtual_wait(
                user, self.sim.spklus[self.sids[default_idx]], time_now)
            ) if not default_in_combo else disp_estwait
            tr.disp_estwait = disp_estwait
            tr.wait_default = wait_default_val
            self.transitions.append(tr)
            self._user_trip_tr[user.user_id] = tr
        else:
            tr = None
        self._pending = (tr, (i, j), user, default_idx, mask, recent_rec_count, wait_hat)
        return sids_chosen

    def _station_feat(self, sid, wait_hat):
        """Vektor fitur fisik SPKLU (lokasi, konektor, estimasi tunggu) utk Prox --
        IDENTIK `rollout.py::RLRolloutAgent._station_feat` (H-PPO), supaya beta_prox
        genap sebanding lintas metode. Sebelumnya PDQN tak pernah memanggil suku ini
        sama sekali -- bukan ketidakcocokan arsitektur, murni belum diimplementasikan."""
        feat = self.sim.spklu_features.get(sid, {})
        loc = feat.get('loc', (0.0, 0.0))
        conn = feat.get('conn', 1.0)
        w = wait_hat.get(sid, 0.0)
        return np.array([loc[0] / self.pos_scale, loc[1] / self.pos_scale,
                         conn, w / self.wait_scale])

    def predict_waits(self, feasible_spklus: dict):
        time_now = self.sim.current_step * self.sim.dt_minutes
        user = getattr(self.sim, "_current_spawn_user", None)
        soc = getattr(self.sim, "_current_spawn_soc", 50.0)
        wait_hat = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        return {s: float(wait_hat.get(s, 0.0)) for s in feasible_spklus}

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        """Pasangan (a_hat, a) terbentuk di sini -- masukan phi milik paper. `combo`=(i,j)
        dua anggota -> DUA pasangan (a_hat_i,a) dicatat ke riwayat preferensi (LSTM,
        tak terkait Q-learning). v3: SATU transisi/reward per keputusan (Q_tot,
        bukan dua transisi terduplikasi -- lihat docstring modul)."""
        if self._pending is None:
            return
        tr, combo, _, default_idx, mask, recent_rec_count, wait_hat = self._pending
        i, j = combo
        chosen_idx = self.sid_to_idx.get(chosen_spklu_id)
        if tr is not None:
            tr.complied = bool(chosen_spklu_id in set(recs))
            # Suku Prox (segera, SAMA persis mekanisme H-PPO rollout.py::on_decision) --
            # kedekatan fitur fisik SPKLU rekomendasi PRIMER (i, anggota Q tertinggi) vs
            # SPKLU yang benar-benar dipilih. Terdefinisi baik saat diterima maupun
            # ditolak (tak dikalikan indikator kepatuhan) -- beta_prox=0 (default semua
            # preset lama) membuat suku ini no-op, jadi aman ditambah tanpa mengubah
            # hasil eksperimen manapun sebelumnya.
            feat_rec = self._station_feat(self.sids[i], wait_hat)
            feat_chosen = self._station_feat(chosen_spklu_id, wait_hat)
            tr.reward += self.rc.decision_reward(self.rc.prox(feat_rec, feat_chosen))
            if self._equity_mode and chosen_idx is not None:
                # Reward pemerataan diketahui SEPENUHNYA di titik ini. `decision_reward_equity`
                # dirancang utk SATU a_hat -- dgn Q_tot=Q(i)+Q(j), reward transisi TUNGGAL
                # dipakai rata2 dari kontribusi kedua anggota (jalur ini TAK dieksekusi di
                # pipeline PDQN kita saat ini, yg pakai RewardCalculator biasa -- lihat
                # _equity_mode di __init__).
                utils = np.array([self.sim.spklus[s].get_utilization() for s in self.sids])
                feasible_idx = np.nonzero(mask)[0]
                r_i = self.rc.decision_reward_equity(
                    utils, feasible_idx, i, default_idx, chosen_idx,
                    recent_rec_count=recent_rec_count)
                r_j = self.rc.decision_reward_equity(
                    utils, feasible_idx, j, default_idx, chosen_idx,
                    recent_rec_count=recent_rec_count)
                tr.reward = 0.5 * (r_i + r_j)
                tr.resolved = True
            elif not self._equity_mode:
                # PERBAIKAN (ditemukan saat migrasi PDQN, 2026-08-12): anti-herding
                # jendela bergulir (Tahap 0.1, H-PPO) TIDAK PERNAH diterapkan ke jalur
                # RewardCalculator biasa di sini -- on_step_end (di bawah) masih
                # memakai flocking_penalty/global_reward LAMA (per-langkah, sama-
                # langkah-saja, aktivasi ~10%), bukan flock_reward_rolling (jendela 24
                # jam, aktivasi ~86%) yang dipakai rollout.py::on_decision. Prinsip
                # "perbaikan substrat diterapkan IDENTIK ke semua lengan" (Tahap 0.1)
                # dilanggar diam-diam -- alpha_flock RewardCalculator.seimbang()
                # dikalibrasi utk statistik jendela bergulir, BUKAN utk mekanisme lama
                # yg nyaris selalu nol ini. `recent_rec_count` SUDAH dihitung di
                # get_recommendation, cuma belum pernah dipakai di jalur ini.
                flock_term = self.rc.flock_reward_rolling(recent_rec_count)
                flock_pen = self.rc.flocking_penalty_rolling(recent_rec_count)
                tr.reward += flock_term
                tr.flock_penalty = flock_pen
        if chosen_idx is not None:
            # DUA pasangan (bukan satu) -- satu per anggota kombinasi yg direkomendasikan,
            # dibandingkan thd pilihan SAMA -- lihat HIST_LEN = PDQN_HIST_K*N_REC. Ini
            # riwayat preferensi (LSTM), TERPISAH dari transisi Q-learning di atas.
            self._record_pref(user, i, chosen_idx, wait_hat)
            self._record_pref(user, j, chosen_idx, wait_hat)
        self._pending = None

    def on_step_end(self, sim, step):
        if self._equity_mode:
            return   # reward pemerataan sudah lunas di on_decision (tak ada suku per-langkah)
        # Suku Gini SAJA di sini (per-langkah, CTDE) -- flocking DIPINDAH ke on_decision
        # (jendela bergulir, lihat perbaikan di atas), TIDAK LAGI dihitung di sini
        # (dulu digabung dlm global_reward -- itu yg memakai flocking_penalty lama).
        transitions_this_step = [tr for tr in self.transitions if tr.step == step]
        if not transitions_this_step:
            return
        utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
        gini_term = self.rc.gini_reward(utils)
        for tr in transitions_this_step:
            if not tr.resolved:
                tr.reward += gini_term

    def on_charge_complete(self, user):
        if self._equity_mode:
            self._user_trip_tr.pop(user.user_id, None)
            return   # tak ada suku wait tertunda di mode pemerataan
        tr = self._user_trip_tr.pop(user.user_id, None)
        if tr is not None:
            # PERBAIKAN (2026-08-16): digerbang `tr.complied` -- IDENTIK rollout.py
            # (H-PPO). Saat rekomendasi (i,j) DITOLAK, hasil wait di stasiun pilihan
            # pengguna bukan konsekuensi kausal dari (i,j) yg direkomendasikan -- lihat
            # Kajian_Literatur_Solusi_Domain_Serupa.md §2.5.
            if tr.complied:
                wait_term = self.rc.wait_reward(tr.wait_default, user.wait_time, tr.disp_estwait)
                tr.reward += wait_term
            tr.resolved = True


class PDQNInferenceAgent:
    """Bungkus PDQNQNetwork terlatih untuk INFERENSI (epsilon=0). Kontrak sama dgn
    GreedyAgent (get_recommendation/predict_waits) + bind_to_sim utk harness."""

    def __init__(self, q_net, forecaster=None, pref_feature_mode: bool = False):
        self.q_net = q_net
        self.forecaster = forecaster or FormulaForecaster()
        self._roll = None
        # HARUS sama dgn mode yg dipakai saat q_net dilatih (menentukan dim pref_hist) --
        # tak bisa dideteksi otomatis dari bobot tersimpan, jadi wajib eksplisit sesuai
        # trainer sumbernya.
        self.pref_feature_mode = bool(pref_feature_mode)

    def bind_to_sim(self, sim):
        from marl_spklu.rl.rewards import RewardCalculator
        self._roll = PDQNRolloutAgent(self.q_net, sim, RewardCalculator(),
                                      self.forecaster, epsilon=0.0, record=False,
                                      pref_feature_mode=self.pref_feature_mode)

    def get_recommendation(self, feasible_spklus):
        return self._roll.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._roll.predict_waits(feasible_spklus)

    # Riwayat preferensi HARUS tetap terisi saat inferensi (LSTM butuh (a_hat,a) lampau).
    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
