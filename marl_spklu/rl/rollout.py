"""Pengumpul rollout RL — menjembatani policy PPO dengan Simulator via hook
(get_recommendation / predict_waits / on_decision / on_step_end / on_charge_complete).

Setiap keputusan spawn = satu transisi RL (parameter sharing: semua EV pakai policy
yang sama). Reward v2 (Spesifikasi_Teknis_RL.md — suku kejujuran DIHAPUS bersama a2):
    R_individual = alpha_wait*relu(wait_default-wait_aktual)/scale     (delayed, on_charge_complete)
                 + beta_prox*Prox(rekomendasi, terpilih)               (segera, on_decision)
    R_global     = -alpha_gini*Gini(utilisasi) - alpha_flock*flocking  (per-step, on_step_end)

Ruang aksi v2: agen HANYA memilih SPKLU mana direkomendasikan (top-K/threshold
deterministik, lihat policy.py::act()). EstWait yang ditampilkan SELALU jujur (murni
`forecaster.predict`, tak ada modifikasi/delta dari kebijakan) — SPKLU feasible di luar
himpunan rekomendasi diberi EstWait = inf (lihat get_recommendation).
"""
import math
from collections import deque

import numpy as np

from marl_spklu.rl.forecaster import FormulaForecaster
from marl_spklu.rl.rewards import _gini as _gini_calc
from marl_spklu.rl.policy import HIST_FEAT_DIM
from marl_spklu.rl.pdqn_policy import (PDQN_HIST_K, hist_feat_dim as _pref_hist_feat_dim,
                                       hist_feat_dim_feature as _pref_hist_feat_dim_feature,
                                       hist_feat_dim_feature_outcome as _pref_hist_feat_dim_outcome,
                                       PREF_STATION_FEAT_DIM)

HIST_K = 5   # panjang jendela riwayat interaksi yang dilihat encoder rekuren (c_t)

# --- Aliran reward terpisah (adopsi kritik-ganda MASTER, lihat Rencana_Kritik_Ganda_MASTER.md) ---
# MASTER (Zhang dkk. WWW 2021) memakai DUA kritik terpisah (Q^cwt, Q^cp) + bobot gradien
# adaptif, dgn alasan eksplisit "the optimal solution of different objectives may be
# divergent" -- terbukti berlaku di sini (korelasi Gini<->waktu tunggu -0,665 pada 9
# kebijakan sehat). Reward TIDAK LAGI dijumlah jadi satu skalar; tiap suku masuk ke aliran
# sesuai SIAPA yang menanggung konsekuensinya:
N_REWARD_STREAMS = 2
STREAM_INDIVIDUAL = 0   # Prox (kecocokan preferensi) + wait -- konsekuensi bagi PENGGUNA
STREAM_GLOBAL = 1       # Gini (pemerataan) + flock (anti-herding) -- konsekuensi bagi JARINGAN

# --- MODE ALIRAN-MURNI (2026-08-30, opt-in `pure_streams=True`) -------------------
# SATU suku per aliran, TANPA penggabungan di dalam aliran:
#     aliran 0 = wait (termasuk circuit-breaker CFR)  -- tertunda, individual
#     aliran 1 = gini                                  -- agregat, populasi
#     aliran 2 = acceptance                            -- segera, individual
# `prox` & `flock` SENGAJA DIBUANG di mode ini (lihat catatan di `on_decision`).
#
# ALASAN: bila satu aliran hanya berisi satu suku, `alpha` suku itu merosot jadi
# penskala SERAGAM aliran -- dan penskalaan seragam lenyap DI DUA TEMPAT sekaligus:
#   (i)  normalisasi advantage per-aliran : (c*adv - mean)/std == (adv - mean)/std
#   (ii) gap-ratio DGR                    : (c*r* - c*ret)/|c*r*| == (r* - ret)/|r*|
# sehingga `alpha` menjadi BENAR-BENAR tanpa efek, bukan sekadar kecil. Tidak ada
# lagi proporsi antar-suku yang perlu dikalibrasi; SELURUH penyeimbangan objektif
# ditangani `beta` (gap-ratio) yang dinamis. Ini membalik sifat scale-invariance yang
# dulu membuat `alpha_gini`/CMDP mati menjadi properti desain yang menguntungkan.
#
# TIDAK mengubah `N_REWARD_STREAMS` (=2) yang dipakai 6 lengan lain -- mode ini
# hanya aktif bila diminta eksplisit.
N_REWARD_STREAMS_PURE = 3
STREAM_PURE_WAIT = 0
STREAM_PURE_GINI = 1
STREAM_PURE_ACCEPT = 2


# Manfaat struktural tambahan: krn advantage dinormalisasi PER ALIRAN di PPOTrainer,
# ketimpangan skala antar-suku (terukur 13,7x, Rumusan_Masalah_Teknis_RL.md §3.2) hilang
# dgn sendirinya -- tak perlu lagi dikalibrasi manual lewat bobot reward.


class Transition:
    """chosen_indices: array PANJANG TETAP = k (dipakai agent), berisi n_rec slot valid
    diikuti padding (indeks 0) -- evaluate() PPO mengabaikan slot >= n_rec via n_rec_b,
    padding di sini murni supaya array bisa di-stack jadi tensor batch seragam.
    chosen_indices[0] = rekomendasi PRIMER (dipakai bookkeeping trust/reward wait_default
    & disp_estwait). v2: TIDAK ADA lagi `deltas` -- ruang aksi murni pemilihan SPKLU,
    EstWait selalu jujur (lihat Spesifikasi_Teknis_RL.md).

    pref_hist: (K,2N) riwayat pasangan (a_hat,a) one-hot, HANYA diisi bila policy punya
    modul preferensi PDQN (lihat RLRolloutAgent._use_pref) -- None untuk H-PPO biasa.

    bids: (N,) float32 -- HANYA untuk lengan stasiun-sebagai-agen (MasterBiddingPolicy).
    Pada lengan itu AKSI SESUNGGUHNYA adalah vektor *bid*, bukan himpunan terpilih:
    seleksi k-terendah bersifat deterministik begitu bid diketahui, sehingga rasio PPO
    harus dihitung atas bid. None untuk seluruh lengan lain."""
    __slots__ = ("obs", "critic_obs", "hist", "mask", "chosen_indices", "n_rec",
                 "logp", "value", "step", "reward_streams", "done",
                 "complied", "disp_estwait", "wait_default", "resolved", "flock_penalty",
                 "pref_hist", "bids", "trust_before")

    def __init__(self, obs, critic_obs, hist, mask, chosen_indices, n_rec, logp, value, step,
                pref_hist=None):
        self.obs = obs; self.critic_obs = critic_obs; self.hist = hist; self.mask = mask
        self.chosen_indices = chosen_indices; self.n_rec = n_rec
        self.logp = logp
        # value: array (K,) -- satu estimasi nilai per aliran reward (K=1 utk kritik tunggal,
        # perilaku lama). Disimpan sbg array supaya compute_gae bisa seragam utk K berapa pun.
        # float64 DISENGAJA (bukan float32): kode sebelum refaktor menyimpan value sbg Python
        # float (float64) dan menjalankan rekursi GAE di float64. Memakai float32 di sini
        # menggeser hasil ~1e-7 per langkah yang TERAMPLIFIKASI kaotik lintas 300 update
        # (terukur: Gini 0,0375 -> 0,0519 pada gerbang regresi). Presisi harus dipertahankan.
        self.value = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self.step = step
        self.reward_streams = np.zeros(N_REWARD_STREAMS, dtype=np.float64)
        self.done = False
        self.flock_penalty = 0.0
        self.pref_hist = pref_hist
        self.bids = None
        # resolved=True bila reward wait (delayed) sudah di-emit saat sesi user selesai.
        # Trainer continuing hanya memakai transisi resolved (§ppo).
        self.resolved = False
        # Diagnostik perilaku (diisi di on_decision):
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        # `trust` (MENTAH, bukan trust_effective) milik pengguna SAAT keputusan diambil --
        # diisi on_decision, dibandingkan dgn nilai SETELAH sesi selesai (on_charge_complete)
        # utk suku shaping trust opsional (RewardCalculator.alpha_trust, baku 0.0 = mati).
        self.trust_before = 0.5

    @property
    def reward(self) -> float:
        """Reward SKALAR = jumlah seluruh aliran. Dipertahankan supaya seluruh pembacaan
        lama (`t.reward` utk logging/diagnostik, dan jalur kritik-tunggal) tetap bekerja
        TANPA perubahan -- dan supaya kritik-tunggal (K=1) identik numerik dgn sebelum
        refaktor: menjumlah aliran = persis reward skalar lama."""
        return float(self.reward_streams.sum())

    def reward_vec(self, n_critics: int) -> np.ndarray:
        """Vektor reward sepanjang `n_critics`. Reward SELALU dicatat per-aliran, tapi
        jumlah KRITIK bisa lebih sedikit: dgn n_critics=1 (kritik tunggal, perilaku lama)
        seluruh aliran DICIUTKAN jadi satu skalar — persis reward terskalarisasi sebelum
        refaktor, sehingga jalur lama identik numerik."""
        if n_critics == 1:
            return np.array([self.reward_streams.sum()], dtype=np.float64)
        if n_critics != N_REWARD_STREAMS:
            raise ValueError(
                f"n_critics={n_critics} tak cocok dgn N_REWARD_STREAMS={N_REWARD_STREAMS}; "
                "hanya 1 (diciutkan) atau K=N_REWARD_STREAMS yang didukung.")
        return self.reward_streams

    def add_reward(self, value: float, stream: int = STREAM_INDIVIDUAL) -> None:
        """Tambahkan suku reward ke ALIRAN tertentu (lihat STREAM_* di atas). Menggantikan
        `tr.reward += value` supaya tiap suku bisa dilacak ke kritiknya sendiri."""
        self.reward_streams[stream] += float(value)


class RLRolloutAgent:
    """Agent yang dipasang ke Simulator selama rollout. Merekam transisi + reward."""

    def __init__(self, policy, sim, reward_calc, forecaster=None, k: int = 3,
                 equity_calc=None, pref_feature_mode: bool = False,
                 epsilon: float = 0.0, threshold: float = 0.20,
                 pref_pair_outcome: bool = False, pref_pad_right: bool = False,
                 pref_hist_k: int = None, accept_stream: int = STREAM_INDIVIDUAL,
                 pure_streams: bool = False):
        self.policy = policy
        self.sim = sim
        self.rc = reward_calc
        self.forecaster = forecaster or FormulaForecaster()
        self.k = int(k)   # BATAS ATAS (langit-langit) jumlah rekomendasi -- v2, bukan target tetap
        # v2: EstWait SELALU jujur (murni forecaster.predict, tak ada a2/delta) -- lihat
        # get_recommendation. `epsilon`/`threshold` diteruskan ke policy.act() tiap keputusan
        # (Spesifikasi_Teknis_RL.md §2.2-2.3): threshold = ambang probabilitas gabungan
        # (lantai 1, langit-langit k); epsilon = peluang eksplorasi acak (gaya PDQN diskrit).
        self.epsilon = float(epsilon)
        self.threshold = float(threshold)
        # equity_calc (opsional, EquityRewardCalculator dari rewards.py): suku reward
        # PEMERATAAN ala PDQN diskrit (r_rec + r_shift + r_flock, SEGERA di on_decision),
        # ditambahkan ke tr.reward SEBAGAI TAMBAHAN (bukan pengganti) R_individual/R_global
        # biasa -- lihat diskusi ablasi "reward seperti PDQN diskrit utk PPO/DDPG kontinu".
        self.equity_calc = equity_calc

        self.sids = list(sim.spklus.keys())
        self.sid_to_idx = {s: i for i, s in enumerate(self.sids)}
        self.N = len(self.sids)

        # Skala normalisasi observasi (agar nilai ~O(1)).
        xs = np.array([sim.spklus[s].location[0] for s in self.sids], dtype=float)
        ys = np.array([sim.spklus[s].location[1] for s in self.sids], dtype=float)
        self.pos_scale = max(1e-6, float(np.std(np.concatenate([xs, ys]))))
        self.dist_scale = 10.0
        self.wait_scale = 60.0
        self.q_scale = 5.0
        self.conn_scale = 5.0
        self.rec_activity_scale = 10.0
        self.beta_state_scale = 5.0
        self.battery_scale = 80.0   # kWh, batas atas rentang realistis (Rancangan Sec 6.1)
        self.freq_scale = 10.0      # freq_i (LogNormal) -- skala normalisasi kasar

        self.transitions = []
        self._pending = None
        self._user_trip_tr = {}   # user_id -> transisi keputusan trip berjalan (utk wait reward)
        self._prev_gini = None    # utk delta_gini (kritik temporal-aware)

        # Modul preferensi PDQN (opsional): aktif hanya bila `policy` punya `pref_lstm`
        # (lihat pdqn_continuous_policy.py::PDQNContinuousPolicy) -- H-PPO biasa (HPPOPolicy
        # tanpa modul ini) tidak terpengaruh sama sekali, jalur ini murni tak dieksekusi.
        self._use_pref = hasattr(policy, "pref_lstm")
        # pref_feature_mode=True: pasangan (a_hat,a) dikodekan sbg PASANGAN VEKTOR FITUR
        # stasiun (jarak/wait/antrean/konektor/util, lihat _pref_station_feat), BUKAN
        # one-hot identitas paper asli -- supaya LSTM belajar dari KARAKTERISTIK stasiun,
        # bukan cuma memorisasi indeks (berguna terutama saat N besar/heterogen).
        self.pref_feature_mode = bool(pref_feature_mode)
        # `pref_pair_outcome` (2026-08-29): tempelkan blok HASIL [complied, realized_gap_norm]
        # di belakang pasangan fitur -- lihat pdqn_policy.py::PREF_OUTCOME_DIM utk alasan
        # lengkap. WAJIB `pref_feature_mode=True` (blok hasil tak bermakna di mode one-hot).
        # BAKU MATI -> dimensi & perilaku lengan lama TAK BERUBAH sama sekali.
        self.pref_pair_outcome = bool(pref_pair_outcome) and self.pref_feature_mode
        if self.pref_pair_outcome:
            self._pref_hist_feat_dim = _pref_hist_feat_dim_outcome()
        elif self.pref_feature_mode:
            self._pref_hist_feat_dim = _pref_hist_feat_dim_feature()
        else:
            self._pref_hist_feat_dim = _pref_hist_feat_dim(self.N)
        # Arah padding `_build_pref_hist`. BAKU depan (kiri) -- perilaku historis, dipakai
        # PDQN & lengan lama yg encoder-nya memproses seluruh K langkah apa adanya.
        # `pref_pad_right=True` WAJIB bila encoder memakai `pack_padded_sequence` (yang
        # mensyaratkan padding di BELAKANG) -- lih. `master_ev_ppo_policy.py::
        # MasterEVPPORolloutAgent._build_pref_hist` yg meng-override justru krn ini.
        # BUG DITEMUKAN 2026-08-29: keluarga Master-Hybrid memakai `_encode_pref` ber-
        # `pack_padded_sequence` TAPI mewarisi padding-kiri basis ini -- akibatnya
        # `lengths` baris pertama yg diambil justru PADDING NOL, sehingga `pref_lstm`
        # TAK PERNAH melihat riwayat sungguhan. Flag ini perbaikannya.
        self._pref_pad_right = bool(pref_pad_right)
        # `pref_hist_k` (2026-08-30): override panjang jendela riwayat P per-instance,
        # TERISOLASI dari konstanta global `PDQN_HIST_K` -- pola SAMA `master_ev_ppo_
        # policy.py::MasterEVPPORolloutAgent`. Baku None -> ikut PDQN_HIST_K (perilaku
        # lama, TAK berubah). Dipakai utk ablasi panjang jendela (mis. K=5 vs K=10) tanpa
        # menyentuh lengan lain yg berbagi modul ini (PDQN, Kandidat A/B, Hybrid).
        self._pref_hist_k = int(pref_hist_k) if pref_hist_k is not None else PDQN_HIST_K
        # `accept_stream` (2026-08-30): aliran tujuan suku `acceptance_reward`. BAKU
        # STREAM_INDIVIDUAL = perilaku lama, TAK mengubah lengan manapun yg sudah ada.
        # ALASAN opsi ini ada: diagnosis 2026-08-21 (lih. `master_ev_ppo_policy.py`
        # ::on_decision) menemukan acceptance ber-magnitudo +-1 SEGERA tiap keputusan,
        # bila menumpuk di aliran yg sama dgn `wait_reward` (kecil & TERTUNDA) membuat
        # r_bar aliran itu meledak 10-30x -- diduga penyebab wait/gini liar pada
        # Kandidat A. Kandidat A menyelesaikannya dgn memindahkan acceptance ke aliran
        # GLOBAL. Keluarga Hybrid hanya punya 2 aliran (INDIVIDUAL=prox+wait,
        # GLOBAL=gini+flock), jadi risiko tumpukan itu SAMA -- opsi ini memungkinkan
        # menaruhnya di GLOBAL, sekaligus lebih selaras konseptual bila acceptance
        # diperlakukan sbg PRASYARAT tujuan pemerataan (bukan kenyamanan individual).
        self.accept_stream = int(accept_stream)
        # Mode aliran-murni (lihat catatan STREAM_PURE_* di kepala modul). Indeks tujuan
        # tiap suku disimpan sbg atribut supaya hook `on_*` di bawah tak perlu bercabang
        # di setiap pemanggilan -- cukup `self.s_wait`/`self.s_gini`/`self.s_accept`.
        self.pure_streams = bool(pure_streams)
        self.n_streams = N_REWARD_STREAMS_PURE if self.pure_streams else N_REWARD_STREAMS
        if self.pure_streams:
            self.s_wait, self.s_gini = STREAM_PURE_WAIT, STREAM_PURE_GINI
            self.s_accept = STREAM_PURE_ACCEPT
        else:
            self.s_wait, self.s_gini = STREAM_INDIVIDUAL, STREAM_GLOBAL
            self.s_accept = int(accept_stream)
        self._pref_hist = {}   # user_id -> deque[pair], onehot-idx ATAU vektor fitur tergantung mode

    def _pref_station_feat(self, sid, user, wait_hat):
        """Vektor fitur (PREF_STATION_FEAT_DIM=5) 1 stasiun utk pasangan preferensi mode
        FITUR: [jarak, est_wait, antrean, konektor, utilisasi] -- deskriptif, bukan
        identitas, sehingga preferensi bisa digeneralisasi lintas stasiun BERKARAKTER
        SERUPA (bukan cuma dihafal per-indeks seperti one-hot paper asli)."""
        feat = self.sim.spklu_features.get(sid, {})
        loc = feat.get('loc', (0.0, 0.0))
        dist = math.dist(user.location, loc) / self.dist_scale
        wait = wait_hat.get(sid, 0.0) / self.wait_scale
        q = self.sim.spklus[sid].get_queue_length() / self.q_scale
        conn = feat.get('conn', 1.0) / self.conn_scale
        util = self.sim.spklus[sid].get_utilization()
        return np.array([dist, wait, q, conn, util], dtype=np.float32)

    def _build_pref_hist(self, user):
        """Riwayat T=PDQN_HIST_K pasangan (a_hat,a) terakhir pengguna ini -> (K, dim)
        left-padded nol -- one-hot ganda (K,2N, default, paper §IV.A) atau pasangan
        vektor fitur (K,2xPREF_STATION_FEAT_DIM, pref_feature_mode=True)."""
        arr = np.zeros((self._pref_hist_k, self._pref_hist_feat_dim), dtype=np.float32)
        h = self._pref_hist.get(user.user_id)
        if h:
            recent = list(h)[-self._pref_hist_k:]
            # padding di BELAKANG (offset=0) bila encoder memakai pack_padded_sequence,
            # di DEPAN (offset=K-len) spt semula bila tidak -- lihat `_pref_pad_right`.
            offset = 0 if self._pref_pad_right else self._pref_hist_k - len(recent)
            for t, pair in enumerate(recent):
                if self.pref_feature_mode:
                    arr[offset + t] = pair
                else:
                    a_hat_idx, a_idx = pair
                    arr[offset + t, a_hat_idx] = 1.0
                    arr[offset + t, self.N + a_idx] = 1.0
        return arr

    def _record_pref(self, user, a_hat_idx, a_idx, wait_hat=None, complied=None):
        """`complied` hanya dipakai mode `pref_pair_outcome` -- blok hasil ditulis
        [complied, 0.0]; elemen kedua (`realized_gap_norm`) di-BACKFILL belakangan di
        `on_charge_complete`, krn baru diketahui setelah sesi pengisian selesai. Pola
        backfill identik `user.interaction_history[-1]` di hook yang sama."""
        h = self._pref_hist.get(user.user_id)
        if h is None:
            h = deque(maxlen=self._pref_hist_k)
            self._pref_hist[user.user_id] = h
        if self.pref_feature_mode:
            feat_hat = self._pref_station_feat(self.sids[a_hat_idx], user, wait_hat or {})
            feat_a = self._pref_station_feat(self.sids[a_idx], user, wait_hat or {})
            row = np.concatenate([feat_hat, feat_a])
            if self.pref_pair_outcome:
                row = np.concatenate([row, np.array([1.0 if complied else 0.0, 0.0],
                                                    dtype=np.float32)])
            h.append(row)
        else:
            h.append((int(a_hat_idx), int(a_idx)))

    def _backfill_pref_outcome(self, user, realized_gap_norm: float):
        """Isi elemen terakhir (`realized_gap_norm`) blok hasil pd entri riwayat
        preferensi TERAKHIR pengguna ini. Aman krn `_user_trip_tr` menjamin satu trip
        aktif per pengguna, sehingga entri terakhir memang milik trip yang baru selesai."""
        if not self.pref_pair_outcome:
            return
        h = self._pref_hist.get(user.user_id)
        if h:
            h[-1][-1] = np.float32(realized_gap_norm)

    def _build_hist(self, user):
        """Riwayat interaksi K-langkah terakhir -> (K, HIST_FEAT_DIM), left-padded nol.
        Konsumsi encoder rekuren (§POMDP, IV.1.7): agen tidak melihat trust T_i, hanya
        jejak (complied, disp_estwait_norm, wait_default_norm) dari mana c_t diinfer."""
        h = user.interaction_history[-HIST_K:]
        arr = np.zeros((HIST_K, HIST_FEAT_DIM), dtype=np.float32)
        if h:
            arr[HIST_K - len(h):] = np.asarray(h, dtype=np.float32)
        return arr

    # ---------------- Observasi (§3.3.1): 3 + 5N ----------------
    def _build_obs(self, user, soc, feasible_ids, time_now):
        loc = user.location
        wait_hat = self.forecaster.predict(self.sim.spklus, time_now, user=user, soc=soc, sim=self.sim)
        onehot = np.zeros(self.N, dtype=float)
        if user.prev_spklu in self.sid_to_idx:
            onehot[self.sid_to_idx[user.prev_spklu]] = 1.0
        d = np.array([math.dist(loc, self.sim.spklus[s].location) for s in self.sids]) / self.dist_scale
        wv = np.array([wait_hat.get(s, 0.0) for s in self.sids]) / self.wait_scale
        q = np.array([self.sim.spklus[s].get_queue_length() for s in self.sids]) / self.q_scale
        # Karakteristik statis (jumlah konektor) -- SEBELUMNYA hanya dipakai internal
        # (Prox/u_def), tak pernah diobservasi aktor -> aktor tak bisa membedakan
        # "antrean panjang krn kapasitas kecil" vs "krn memang padat" tanpa ini.
        conn = np.array([self.sim.spklu_features.get(s, {}).get('conn', 1.0) for s in self.sids]) / self.conn_scale
        # Waktu-dalam-hari eksplisit (sin/cos, kontinu tanpa loncatan di batas 24 jam) --
        # simulasi memodelkan pola 2-puncak harian (generator dataset), TAPI sebelumnya
        # hanya tersirat lewat wait_hat forecaster, tak pernah jadi observasi langsung.
        hour = (time_now / 60.0) % 24.0
        sin_hour, cos_hour = math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)
        # Kapasitas baterai KENDARAAN INI (heterogen -- campuran 3 model EV, lihat
        # User.battery_capacity_kwh, Pemodelan_Variasi_Distribusi.md §8).
        battery_norm = user.battery_capacity_kwh / self.battery_scale
        scalars = np.array([soc / 100.0, loc[0] / self.pos_scale, loc[1] / self.pos_scale,
                            sin_hour, cos_hour, battery_norm])

        # Preferensi bawaan pengguna (P_pref murni) -- dipakai untuk menentukan "stasiun
        # default" sebagai pembanding waktu-tunggu pada reward. Konsisten dengan model
        # pilihan pengguna (user.decide_spklu): U(s) = w1*dist + w2*log_pop + w3*log_conn
        # + w4*isPrev + w5*soc_urgency (Model_Simulasi_Inti.md §3.1). w1-w5 dari kelas
        # Latent Class MNL, dibaca langsung per-user (bukan draw acak MXL).
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
        # rec_activity (jendela 24 jam, sim.recent_recs) -- WAJIB di obs aktor v2 (bukan
        # cuma critic) supaya jaringan bisa belajar menurunkan skor SPKLU yg baru sering
        # direkomendasikan SEBELUM memilih, krn seleksi kini deterministik (top-K/threshold,
        # bukan sampling stokastik) -- tanpa sinyal ini herding tak bisa dihindari proaktif
        # (lihat Spesifikasi_Teknis_RL.md §2.3).
        rec_act = np.array([self.sim.recent_recs.get(s, 0) for s in self.sids]) / self.rec_activity_scale
        obs = np.concatenate([scalars, onehot, d, wv, q, conn, rec_act]).astype(np.float32)
        return obs, default_idx, wait_hat

    def _feasible_mask(self, feasible_ids):
        m = np.zeros(self.N, dtype=bool)
        for s in feasible_ids:
            if s in self.sid_to_idx:
                m[self.sid_to_idx[s]] = True
        return m

    def _build_critic_obs(self, user, time_now):
        """CTDE: kritik mengakses variabel PRIVILEGED (tak pernah masuk obs aktor):
        trust ASLI T_i (agregat populasi + trust PENGGUNA YANG SEDANG DIPUTUSKAN),
        freq_i & w4 (isPrev) pengguna itu (frekuensi kunjungan & loyalitas -- sifat
        bawaan tak teramati di deployment), dan WAIT ORACLE per-stasiun dihitung
        LANGSUNG dari state antrean/charging sungguhan simulator (`compute_virtual_wait`
        -- FIFO atas remaining_time EV yang sedang charging + EV traveling menuju
        stasiun), BUKAN estimasi forecaster yang dipakai aktor (`wait_hat`, bisa
        noisy/belum terlatih). Kritik cuma aktif saat training (CTDE) jadi boleh
        "curang" melihat kondisi sebenarnya -- ini bukan kebocoran ke aktor.

        Fitur per-stasiun (5): utilisasi, antrean, aktivitas rekomendasi (phi_activity,
        sinyal sama yg dipakai prediktor sadar-koordinasi), jumlah konektor, WAIT
        ORACLE (ground-truth, bukan wait_hat forecaster).

        Skalar global (8): gini, trust_mean/std/min (populasi), delta_gini (temporal),
        trust/freq_i/w4 PENGGUNA AKTIF (transisi ini dievaluasi utk pengguna spesifik
        -- baseline value seharusnya beda utk pengguna trust-tinggi vs rendah, bukan
        cuma tahu rata-rata populasi)."""
        utilisasi_per_spklu = np.array([self.sim.spklus[s].get_utilization() for s in self.sids])
        antrean_per_spklu = np.array([self.sim.spklus[s].get_queue_length() for s in self.sids]) / self.q_scale
        rec_activity_per_spklu = np.array(
            [self.sim.recent_recs.get(s, 0) for s in self.sids]) / self.rec_activity_scale
        conn_per_spklu = np.array(
            [self.sim.spklu_features.get(s, {}).get('conn', 1.0) for s in self.sids]) / self.conn_scale
        oracle_wait_per_spklu = np.array([
            self.sim.compute_virtual_wait(user, self.sim.spklus[s], time_now) for s in self.sids
        ]) / self.wait_scale
        gini_utilisasi = _gini_calc(utilisasi_per_spklu)

        if self._prev_gini is None:
            delta_gini = 0.0
        else:
            delta_gini = gini_utilisasi - self._prev_gini
        self._prev_gini = gini_utilisasi

        # trust_effective (BUKAN u.trust mentah) -- variabel yg SECARA KAUSAL memengaruhi
        # decide_spklu (lihat User.trust_effective). Kalau ablasi constant_trust_shadow
        # aktif, u.trust jadi trust IMAJINER (diagnostik saja, tak berpengaruh ke dinamika
        # nyata) -- kritik CTDE harus tetap melihat variabel yg SUNGGUH relevan, bukan
        # bayangan itu, supaya training kritik tak dikacaukan varians semu.
        trusts = np.array([u.trust_effective for u in self.sim.users], dtype=float)
        if trusts.size:
            trust_mean, trust_std, trust_min = float(trusts.mean()), float(trusts.std()), float(trusts.min())
        else:
            trust_mean, trust_std, trust_min = 0.5, 0.0, 0.5

        user_trust = float(user.trust_effective)
        user_freq_norm = float(user.freq_i) / self.freq_scale
        user_w4_norm = float(user.w4) / self.beta_state_scale
        hour = (time_now / 60.0) % 24.0
        sin_hour, cos_hour = math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)

        return np.concatenate([
            utilisasi_per_spklu,
            antrean_per_spklu,
            rec_activity_per_spklu,
            conn_per_spklu,
            oracle_wait_per_spklu,
            [gini_utilisasi, trust_mean, trust_std, trust_min, delta_gini,
             user_trust, user_freq_norm, user_w4_norm, sin_hour, cos_hour],
        ]).astype(np.float32)

    def _station_feat(self, sid, wait_hat):
        """Vektor fitur fisik SPKLU teramati (lokasi, konektor, estimasi tunggu) utk Prox."""
        feat = self.sim.spklu_features.get(sid, {})
        loc = feat.get('loc', (0.0, 0.0))
        conn = feat.get('conn', 1.0)
        w = wait_hat.get(sid, 0.0)
        return np.array([loc[0] / self.pos_scale, loc[1] / self.pos_scale,
                         conn, w / self.wait_scale])

    # ---------------- Hook Simulator ----------------
    def get_recommendation(self, feasible_spklus: dict):
        """a1 kini k SPKLU sekaligus (bukan 1) -- lihat policy.py act(). Rekomendasi
        PRIMER (chosen_indices[0]) tetap jadi dasar bookkeeping trust/reward wait
        (disp_estwait/wait_default), TIDAK berubah dari perilaku k=1 sebelumnya."""
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        obs, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)
        mask = self._feasible_mask(feasible_ids)
        critic_obs = self._build_critic_obs(user, time_now)
        hist = self._build_hist(user)
        if self._use_pref:
            pref_hist = self._build_pref_hist(user)
            act = self.policy.act(obs, mask, hist, k=self.k, critic_obs_np=critic_obs,
                                  epsilon=self.epsilon, threshold=self.threshold,
                                  pref_hist_np=pref_hist)
        else:
            pref_hist = None
            act = self.policy.act(obs, mask, hist, k=self.k, critic_obs_np=critic_obs,
                                  epsilon=self.epsilon, threshold=self.threshold)

        chosen_indices = act["chosen_indices"]   # list panjang n_rec (<=k)
        n_rec = act["n_rec"]
        recs = [self.sids[i] for i in chosen_indices]

        # v2: EstWait SELALU jujur (murni forecaster.predict, tak ada a2/delta) -- SPKLU
        # feasible di LUAR himpunan rekomendasi diberi inf (bukan proksi 2x-max lama),
        # supaya P_rec (softmax(exp(-gamma*wait))) memberi insentif TEGAS memilih di
        # antara yg direkomendasikan (exp(-gamma*inf)=0 eksak, aman scr numerik).
        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_indices}
        estimated_waits = {
            sid: rec_disps[sid] if sid in rec_disps else float("inf")
            for sid in feasible_ids
        }

        primary_idx = chosen_indices[0]
        primary_disp = rec_disps[self.sids[primary_idx]]

        # Array PANJANG TETAP=k utk konsistensi batching PPO (slot >= n_rec = padding).
        idx_arr = np.zeros(self.k, dtype=np.int64)
        idx_arr[:n_rec] = chosen_indices

        tr = Transition(obs, critic_obs, hist, mask, idx_arr, n_rec,
                        act["logp"], act["value"], self.sim.current_step, pref_hist=pref_hist)
        # Lengan stasiun-sebagai-agen: simpan vektor bid sbg AKSI sesungguhnya (lihat
        # Transition.__doc__). None untuk seluruh lengan lain -- `act` tak memuat kunci ini.
        tr.bids = act.get("bids")
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        # recent_rec_count DIBACA SEBELUM Simulator menambah hitungan (recs[0]) di luar
        # sini -- "seberapa ramai a_hat ini direkomendasikan SEBELUM keputusan ini",
        # sama semantik pdqn_agent.py (suku anti-overshoot EquityRewardCalculator).
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr   # utk emit wait reward saat sesi selesai
        return recs

    def predict_waits(self, feasible_spklus: dict):
        if self._pending is None:
            return {s: 0.0 for s in feasible_spklus}
        return self._pending[1]

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        if self._pending is None:
            return
        tr, _, primary_idx, wait_hat, default_idx, recent_rec_count = self._pending
        complied = chosen_spklu_id in set(recs)
        tr.complied = bool(complied)
        tr.trust_before = float(user.trust)   # snapshot SEBELUM sesi ini bisa mengubahnya
        # Suku shaping acceptance (opsional, BAKU MATI -- rc.alpha_accept=0.0 tak
        # mengubah reward sama sekali). SIMETRIS (+patuh/-tolak), beda dari Prox/wait
        # yg tak pernah menghukum penolakan -- lihat RewardCalculator.acceptance_reward.
        if self.rc.alpha_accept != 0.0:
            tr.add_reward(self.rc.acceptance_reward(complied), self.s_accept)
        # Suku Prox (segera): kedekatan fitur fisik SPKLU rekomendasi PRIMER vs SPKLU
        # terpilih. Terdefinisi baik saat diterima maupun ditolak (tidak dikalikan
        # indikator kepatuhan).
        # DIBUANG di mode aliran-murni: menaruhnya bersama `wait` akan mengembalikan
        # kebutuhan `alpha` (proporsi wait:prox) yang justru ingin dihapus mode ini.
        # Kerugiannya kecil -- prox nyaris konstan scr struktural (rasio mean/std ~6,5),
        # sehingga kontribusinya ke advantage TERNORMALISASI memang hampir nol.
        if not self.pure_streams:
            feat_rec = self._station_feat(self.sids[primary_idx], wait_hat)
            feat_chosen = self._station_feat(chosen_spklu_id, wait_hat)
            prox_value = self.rc.prox(feat_rec, feat_chosen)
            tr.add_reward(self.rc.decision_reward(prox_value), STREAM_INDIVIDUAL)
        # Anti-herding JENDELA BERGULIR (Tahap 0.1) -- dihitung di sini (per KEPUTUSAN,
        # bukan per langkah) krn `recent_rec_count` (jendela 24 jam berjalan) sudah
        # tersedia dari `get_recommendation`. Menggantikan penalti per-langkah lama
        # (dulu di on_step_end) yg cuma aktif 10,1% transisi -- herding di sistem ini
        # bersifat TEMPORAL, bukan simultan-per-langkah.
        # DIBUANG di mode aliran-murni (alasan sama spt prox): menaruh flock bersama gini
        # mengembalikan kebutuhan `alpha` (proporsi gini:flock). Perannya sbg objektif
        # JARINGAN tumpang tindih dgn gini -- ini perubahan desain yang WAJIB dinyatakan
        # eksplisit saat melaporkan mode ini (anti-herding Tahap 0.1 tak lagi aktif).
        flock_term = self.rc.flock_reward_rolling(recent_rec_count)
        if not self.pure_streams:
            tr.add_reward(flock_term, STREAM_GLOBAL)   # anti-herding melayani tujuan JARINGAN
        tr.flock_penalty = self.rc.flocking_penalty_rolling(recent_rec_count)   # diagnostik, tetap
        # Suku reward PEMERATAAN ala PDQN diskrit (opsional, TAMBAHAN) -- segera,
        # bebas derau kepatuhan, langsung proporsional thd utilisasi saat keputusan.
        if self.equity_calc is not None:
            chosen_idx_eq = self.sid_to_idx.get(chosen_spklu_id)
            if chosen_idx_eq is not None:
                utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
                feasible_idx = np.nonzero(tr.mask)[0]
                tr.add_reward(self.equity_calc.decision_reward_equity(
                    utils, feasible_idx, primary_idx, default_idx, chosen_idx_eq,
                    recent_rec_count=recent_rec_count), self.s_gini)
        if self._use_pref:
            chosen_idx = self.sid_to_idx.get(chosen_spklu_id)
            if chosen_idx is not None:
                self._record_pref(user, primary_idx, chosen_idx, wait_hat, complied=complied)
        # Catat interaksi ini ke riwayat pengguna -> dibaca encoder LSTM pada keputusan
        # berikutnya (proksi belief trust dari jejak, bukan akses T_i langsung). Elemen
        # ke-4 (realized_gap_norm) BELUM diketahui saat ini (hasil aktual baru ada di
        # on_charge_complete, sesi belum selesai) -> placeholder netral 0.0, di-backfill
        # nanti. Aman krn user TAK BISA membuat keputusan baru sebelum trip ini selesai
        # (state machine) -> entry ini selalu jadi elemen TERAKHIR saat backfill.
        user.interaction_history.append((
            1.0 if complied else 0.0,
            tr.disp_estwait / self.wait_scale,
            tr.wait_default / self.wait_scale,
            0.0,
        ))
        if len(user.interaction_history) > 30:
            user.interaction_history.pop(0)
        self._pending = None

    def on_step_end(self, sim, step):
        # Suku Gini (CTDE), hanya utk transisi yang DIBUAT step ini. Flocking (anti-
        # herding) TIDAK lagi dihitung di sini -- lihat on_decision::flock_reward_rolling
        # (Tahap 0.1, jendela bergulir 24 jam via sim.recent_recs, bukan per-langkah).
        # PENTING: `gini_reward` HANYA dipanggil bila ADA transisi step ini (persis
        # semantik lama) -- delta-Gini jadi "perubahan sejak keputusan TERAKHIR
        # dievaluasi" (interval tak beraturan), bukan "perubahan per-langkah simulasi
        # tetap" (~34,5% langkah punya keputusan) -- keduanya statistik BERBEDA, yg
        # kedua akan diam-diam membatalkan kalibrasi std Tahap 0.1.
        transitions_this_step = [tr for tr in self.transitions if tr.step == step]
        if not transitions_this_step:
            return
        utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
        gini_term = self.rc.gini_reward(utils)
        for tr in transitions_this_step:
            tr.add_reward(gini_term, self.s_gini)   # pemerataan = konsekuensi JARINGAN

    def on_charge_complete(self, user):
        """Sesi user selesai -> emit suku wait-improvement & honesty (delayed) ke transisi
        keputusannya, lalu tandai resolved (siap dipakai update). Dipanggil Simulator.
        Sekaligus BACKFILL elemen ke-4 interaction_history (realized_gap_norm) dgn
        akurasi janji SESUNGGUHNYA -- SEBELUMNYA encoder LSTM tak pernah melihat apakah
        janji lampau meleset, cuma tahu (kepatuhan, apa yg dijanjikan). Hanya diisi bila
        user MEMATUHI rekomendasi (janji itu relevan utknya -- sama spt syarat
        User.update_trust); tanda: gap>0 = janji meleset/actual lbh lama (over-promise,
        SAMA arah dgn asimetri trust alpha>rho), gap<0 = under-promise."""
        tr = self._user_trip_tr.pop(user.user_id, None)
        if tr is not None:
            # PERBAIKAN (2026-08-16): digerbang `tr.complied` -- saat rekomendasi DITOLAK
            # (pengguna pergi ke stasiun lain), hasil wait DI STASIUN LAIN ITU bukan
            # konsekuensi kausal dari rekomendasi yg diberikan, memberi reward/penalti di
            # sini adalah kesalahan atribusi kredit (lihat Kajian_Literatur_Solusi_
            # Domain_Serupa.md §2.5). Beda dgn Prox/flock (on_decision) yg TETAP valid saat
            # ditolak -- keduanya mengukur properti rekomendasi itu sendiri, bukan hasil di
            # stasiun pilihan pengguna.
            if tr.complied:
                tr.add_reward(
                    self.rc.wait_reward(tr.wait_default, user.wait_time, tr.disp_estwait),
                    self.s_wait)   # perbaikan wait = konsekuensi bagi PENGGUNA
                # Suku shaping trust (opsional, BAKU MATI -- rc.alpha_trust=0.0 tak
                # mengubah reward sama sekali, murni penjumlahan +0.0). `User.update_trust`
                # SUDAH dipanggil simulator SEBELUM hook ini (simulator.py) bila
                # `last_rec_complied` -- `user.trust` di sini sudah nilai TERBARU.
                if self.rc.alpha_trust != 0.0:
                    delta_trust = float(user.trust) - tr.trust_before
                    tr.add_reward(self.rc.trust_shaping_reward(delta_trust),
                                 self.s_wait)
            tr.resolved = True
            if tr.complied and user.interaction_history:
                complied_v, disp_v, wdef_v, _ = user.interaction_history[-1]
                realized_gap_norm = (user.wait_time - tr.disp_estwait) / self.wait_scale
                user.interaction_history[-1] = (complied_v, disp_v, wdef_v, realized_gap_norm)
                # Backfill blok HASIL riwayat preferensi (mode pref_pair_outcome, BAKU MATI
                # -> no-op utk lengan lama). Digerbang `tr.complied` SAMA spt di atas:
                # bila rekomendasi DITOLAK, `realized_gap` mengukur akurasi janji di stasiun
                # yg TIDAK kita rekomendasikan -- bukan konsekuensi kausal aksi agen, jadi
                # dibiarkan 0.0 (netral), konsisten `wait_reward`/`interaction_history`.
                self._backfill_pref_outcome(user, realized_gap_norm)


class InferenceAgent:
    """Bungkus policy untuk INFERENSI (tanpa memakai reward) — dipakai evaluasi bersih dan
    mode C pengumpulan dataset forecaster.

    PERBAIKAN (2026-08-17) — ketidakcocokan latih-uji yang membuat encoder riwayat MATI:
    versi sebelumnya hanya mengekspos `get_recommendation` dan `predict_waits`. Simulator
    memanggil `on_decision`/`on_charge_complete` hanya bila agen memilikinya (hasattr), jadi
    keduanya TAK PERNAH terpanggil saat evaluasi. Akibatnya:

      * `user.interaction_history` tak pernah terisi -> `_build_hist` selalu nol
        (encoder riwayat H-PPO mati; terukur: 0 dari 2.636 pengguna terisi)
      * `_pref_hist` tak pernah terisi -> `_build_pref_hist` selalu nol
        (modul preferensi P-PPO tak pernah menerima masukan sama sekali)

    Kebijakan DILATIH dengan riwayat nyata lalu DIEVALUASI dengan riwayat kosong — yakni
    dievaluasi di luar distribusi pelatihannya. Meneruskan kedua kait ini menyamakan jalur
    observasi evaluasi dengan pelatihan. Reward tetap tak dipakai (RewardCalculatorStub
    mengembalikan nol), sehingga tak ada pembelajaran yang terjadi di sini.
    """

    def __init__(self, policy, sim, forecaster=None, k: int = 3, epsilon: float = 0.0,
                threshold: float = 0.20):
        self._roll = RLRolloutAgent(policy, sim, RewardCalculatorStub(), forecaster, k,
                                    epsilon=epsilon, threshold=threshold)

    def get_recommendation(self, feasible_spklus):
        return self._roll.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._roll.predict_waits(feasible_spklus)

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        # Mengisi `user.interaction_history` dan `_pref_hist` -- syarat agar encoder
        # riwayat & modul preferensi menerima masukan yang sama seperti saat pelatihan.
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        # Transisi tak dipakai saat inferensi; buang supaya memori tak tumbuh sepanjang
        # horizon evaluasi (90 hari bisa puluhan ribu transisi).
        self._roll.transitions.clear()

    def on_charge_complete(self, user):
        # Mem-backfill elemen ke-4 riwayat interaksi (realized_gap_norm) dgn akurasi janji
        # yang SESUNGGUHNYA -- tanpa ini encoder hanya melihat placeholder 0,0.
        self._roll.on_charge_complete(user)
        self._roll.transitions.clear()


class RewardCalculatorStub:
    """Reward tak dipakai saat inferensi; sediakan API minimal yang disentuh RLRolloutAgent.

    Sejak `InferenceAgent` meneruskan `on_decision`/`on_charge_complete` (perbaikan
    ketidakcocokan latih-uji), stub ini ikut menyentuh suku flocking dan gini -- keduanya
    WAJIB ada di sini, kalau tidak evaluasi akan gagal dgn AttributeError. Semuanya
    mengembalikan nol: tak ada reward, tak ada pembelajaran saat inferensi.
    """
    alpha_wait = 0.0
    beta_prox = 0.0
    alpha_gini = 0.0
    alpha_flock = 0.0
    alpha_trust = 0.0
    alpha_accept = 0.0
    alpha_equity = 0.0

    def acceptance_reward(self, complied):
        return 0.0

    def local_equity_reward(self, chosen_util, mean_util, recent_rec_count=0.0,
                            overshoot_scale=10.0):
        return 0.0

    def gini_reward_terminal(self, gini_level):
        return 0.0

    def prox(self, feat_rec, feat_chosen):
        return 0.0

    def decision_reward(self, prox_value):
        return 0.0

    def wait_reward(self, wait_default, wait_actual, disp_estwait):
        return 0.0

    def flock_reward_rolling(self, recent_rec_count):
        return 0.0

    def flocking_penalty_rolling(self, recent_rec_count):
        return 0.0

    def gini_reward(self, *args, **kwargs):
        return 0.0

    def reset_episode_state(self):
        return None

    def global_reward(self, utilizations, n_same=0, n_window=0):
        return 0.0

    @staticmethod
    def flocking_penalty(n_same, n_window):
        return 0.0


def collect_rollout(sim, policy, reward_calc, forecaster=None, k: int = 3, max_steps=None,
                    epsilon: float = 0.0, threshold: float = 0.20):
    """Jalankan satu episode penuh, kembalikan (transitions, info). `sim` sudah dimuat."""
    agent = RLRolloutAgent(policy, sim, reward_calc, forecaster, k,
                           epsilon=epsilon, threshold=threshold)
    if max_steps is None:
        max_steps = sim.max_steps
    for step in range(max_steps):
        sim.step_once(step, agent=agent)
    if agent.transitions:
        agent.transitions[-1].done = True
    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    trs = agent.transitions
    if trs:
        rewards = np.array([t.reward for t in trs])
        info = {
            "n_transitions": len(trs),
            "mean_reward": float(rewards.mean()),
            # Sinyal perilaku (tautkan RL ke KPI tesis + deteksi reward-hacking).
            "acceptance_rate": float(np.mean([t.complied for t in trs])),
            "mean_disp_estwait": float(np.mean([t.disp_estwait for t in trs])),
            "mean_flock_penalty": float(np.mean([t.flock_penalty for t in trs])),
            "herding_events": sim.herding_events,
            "gini_served": _gini(served),
            "total_served": int(served.sum()),
        }
    else:
        info = {"n_transitions": 0, "mean_reward": 0.0, "gini_served": _gini(served),
                "total_served": int(served.sum())}
    return trs, info


def evaluate_policy(sim, policy, forecaster=None, k: int = 3, max_steps=None,
                    epsilon: float = 0.0, threshold: float = 0.20):
    """Jalankan satu episode penuh dengan policy TERLATIH (inferensi) sebagai agent.
    Kembalikan dict metrik evaluasi (served, Gini, herding, waktu tunggu)."""
    agent = InferenceAgent(policy, sim, forecaster, k, epsilon=epsilon, threshold=threshold)
    if max_steps is None:
        max_steps = sim.max_steps
    sim.run(max_steps=max_steps, agent=agent)
    served = np.array([s.total_served for s in sim.spklus.values()], dtype=float)
    waits = [L["wait_time"] for L in sim.logs]
    return {
        "gini_served": _gini(served),
        "total_served": int(served.sum()),
        "n_active": int((served > 0).sum()),
        "herding_events": sim.herding_events,
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
        "p90_wait": float(np.quantile(waits, 0.9)) if waits else 0.0,
    }


def _gini(a):
    a = np.clip(np.asarray(a, float), 0, None)
    if a.sum() == 0:
        return 0.0
    a = np.sort(a); n = a.shape[0]; idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * np.sum(a)))
