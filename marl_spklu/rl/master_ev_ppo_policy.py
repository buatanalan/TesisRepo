"""MASTER perspektif-EV, DILATIH via `ppo.py::PPOTrainer` LANGSUNG (tanpa modifikasi)
-- pengganti trainer custom `master_ev_trainer.py::MasterEVTrainer` yang kritiknya
terbukti TAK PERNAH stabil sepanjang 300 chunk (critic_loss tetap orde ribuan-puluhan
ribu hampir sepanjang training, gini_util per-chunk melompat 0,04-0,57 tanpa tren --
lihat riwayat `master_ev_training_results.json`).

SATU PERUBAHAN UTAMA (agar dapat diatribusikan): kritik "kolektif per-timestep"
(`MasterEVJointCritic`, self-attention antar-EV yang berkeputusan sama saat itu)
DIGANTI kritik V(s) TUNGGAL ber-atensi HANYA atas kandidat stasiun EV itu SENDIRI
(`AttentionPooling`, `policy.py`) -- pola PERSIS `MasterStationPPOPolicy`, satu-satunya
lengan keluarga MASTER-PPO yang terbukti stabil (giniSD 0,005-0,010 lintas 3 seed,
`[[tahap2-gae-delayed-reward-bug]]` & diskusi perbandingan MASTER-EV). Trainer custom
(_value_pass/_critic_loss/_group_by_step per-timestep) DIHAPUS SELURUHNYA -- memakai
`ppo.py::PPOTrainer.update()` apa adanya (KL early-stop, minibatch epoch, gap-ratio DGR,
normalisasi advantage per-aliran -- SEMUA mesin yang sudah terbukti stabil di H-PPO/
P-PPO/MASTER-bid/MASTER-stasiun-PPO, TAK diduplikasi/ditulis ulang).

DIPERTAHANKAN dari `master_ev_policy.py` (unit agen = permintaan EV, BUKAN stasiun)
------------------------------------------------------------------------------------
  - Observasi §3.1+state-EV (`build_joint_obs_master_ev`, 10 fitur/stasiun) -- EV native
    di observasi aktor, bukan tempelan.
  - Aksi = rekomendasi kategorikal LANGSUNG (top-K/threshold+epsilon-greedy, IDENTIK
    `HPPOPolicy.act()`/`evaluate()` -- disalin apa adanya, bukan re-implementasi, supaya
    perbedaan hasil tak bisa dijelaskan oleh perbedaan logika seleksi).

DIHILANGKAN scr EKSPLISIT (sama pola & alasan `MasterStationPPOPolicy`)
-------------------------------------------------------------------------
  - Kritik kolektif per-timestep (evaluasi "dikumpulkan per timestep") -- diganti V(s)
    tunggal, krn diagnosis kuat (`critic_loss` liar 300 chunk penuh) menunjuk ke SINI
    sbg biang ketidakstabilan, bukan ke tulang-punggung DDPG (sudah PPO dari awal) atau
    reward (preset `seimbang4x` sudah diuji & DITOLAK di kelas lama).
  - Delayed Access Strategy (privileged future_avail) -- kritik V(s) di sini HANYA
    melihat state SAAT INI, sama seperti H-PPO/P-PPO/MASTER-bid/MASTER-stasiun-PPO.

BILA hasil membaik tajam & stabil, itu ISYARAT KUAT (bukan bukti definitif -- kritik
kolektif-per-timestep tetap gagasan yang beralasan dari sisi teori CTDE, hanya
implementasinya di sini yang terbukti tak stabil) bahwa arsitektur kritiknya-lah yang
perlu didesain ulang lebih hati-hati, BUKAN unit-agen-EV atau bentuk-aksi-kategorikalnya."""
import random
import warnings

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.policy import StationEncoder, AttentionPooling, NEG_INF, HIST_FEAT_DIM, HIST_HIDDEN
from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER_EV, build_joint_obs_master_ev
from marl_spklu.rl.rollout import (RLRolloutAgent, Transition, RewardCalculatorStub, _gini,
                                   STREAM_INDIVIDUAL, STREAM_GLOBAL)
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.ppo import PPOTrainer, _make_logger
from marl_spklu.rl.pdqn_policy import (PreferenceAttention, hist_feat_dim,
                                       hist_feat_dim_feature, PREF_STATION_FEAT_DIM,
                                       PDQN_HIST_K)
from marl_spklu.rl.p_ppo_policy import PREF_D_LSTM, PREF_D_ATTN

# --- Aliran reward 3-arah, KHUSUS keluarga MasterEVPPO (n_critics=3) ---------------
# TIDAK mengubah `rollout.py::STREAM_INDIVIDUAL/STREAM_GLOBAL/N_REWARD_STREAMS` yang
# dipakai BERSAMA oleh hampir semua lengan lain (H-PPO/P-PPO/MASTER-bid/dst) -- itu
# akan mengubah bentuk kritik SEMUA lengan itu sekaligus & merusak checkpoint yang
# sudah ada. Sebagai gantinya: 3 aliran LOKAL, hanya aktif bila `policy.n_critics==3`
# (lihat `MasterEVPPORolloutAgent._split_prox`), memisahkan Prox dari perbaikan-wait
# (SEBELUMNYA tergabung satu kolom di STREAM_INDIVIDUAL -- diagnosis "kenapa wait
# anjlok saat +P+K2" menunjuk ke sini: begitu P membuat Prox lebih mudah dioptimalkan,
# kontribusinya mendominasi kolom gabungan, wait ikut terseret tanpa disadari terpisah).
STREAM_WAIT = 0     # perbaikan-wait SAJA (dulu tergabung STREAM_INDIVIDUAL)
STREAM_PROX = 1     # Prox SAJA (dulu tergabung STREAM_INDIVIDUAL)
STREAM_GLOBAL3 = 2  # Gini + anti-herding (identik STREAM_GLOBAL, indeks beda krn 3 aliran)


class MasterEVPPOPolicy(nn.Module):
    """Aktor per-PERMINTAAN (`station_encoder`+`disc_head`) + kritik V(s) "buta-aksi"
    ber-atensi (`AttentionPooling`, sama kelas dipakai `MasterStationPPOPolicy`).

    `hist_lstm` (2026-08-21, PERBAIKAN): encoder riwayat interaksi (`HPPOPolicy.
    hist_lstm`, IDENTIK -- disalin apa adanya) yang SEBELUMNYA TAK ADA di kelas ini
    ("hist diabaikan"). Ini kekurangan nyata, bukan desain sengaja: `alpha_trust`
    (RewardCalculator, lihat rewards.py) memberi reward atas perubahan `user.trust`,
    tapi `user.trust` sendiri TAK PERNAH masuk observasi manapun (sesuai §3.1 MASTER
    -- desentralisasi, agen buta thd atribut pemohon) -- membuat reward itu BERISIK bagi
    agen yg tak punya cara membedakan "pengguna trust tinggi" dari "pengguna trust
    rendah" pada fitur stasiun yg identik. `hist_lstm` memberi PROXY trust yg TERAMATI
    (riwayat patuh/estwait/wait_default/realized_gap K langkah terakhir -> c_t, belief
    state ala POMDP Kaelbling dkk. 1998 -- BUKAN akses langsung ke `trust` mentah,
    tetap konsisten §3.1) -- prasyarat supaya `alpha_trust` punya dasar yg bisa
    dipelajari, bukan lagi noise laten murni."""

    def __init__(self, n_spklu: int, hidden: int = 64, critic_hidden: int = 128,
                n_critics: int = 1, hist_hidden: int = HIST_HIDDEN, use_hist: bool = True):
        super().__init__()
        self.n_spklu = int(n_spklu)
        self.n_critics = int(n_critics)
        self.scalar_dim = 0   # state EV sudah masuk tiap baris kandidat, tak perlu blok skalar
        self.hist_hidden = int(hist_hidden)
        # `use_hist=False` (2026-08-21, ABLASI): matikan kontribusi c_t (`hist_lstm`)
        # TANPA mengubah bentuk jaringan (module tetap ada, dim tak berubah, checkpoint
        # tetap kompatibel) -- utk mengisolasi dugaan bahwa `hist_lstm` & `pref_lstm`
        # (MasterEVPPOPrefPolicy) saling mengganggu krn menggali sinyal riwayat yg
        # tumpang tindih. Satu-satunya hasil P yg pernah POSITIF di seluruh sesi
        # terjadi SEBELUM hist_lstm ada -- belum pernah diuji ulang P TANPA hist_lstm
        # setelah keduanya digabung. Pola sama `use_preference=False` (ablasi P).
        self.use_hist = bool(use_hist)

        self.hist_lstm = nn.LSTM(HIST_FEAT_DIM, self.hist_hidden, batch_first=True)

        self.station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV, self.hist_hidden, hidden)
        self.disc_head = nn.Linear(hidden, 1)

        self.critic_station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV, self.hist_hidden,
                                                      critic_hidden)
        self.critic_pool = AttentionPooling(critic_hidden)
        self.critic_head = nn.Sequential(
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, self.n_critics),
        )

    def _encode_hist(self, hist):
        """hist: (B,K,HIST_FEAT_DIM) -> c_t: (B,hist_hidden). IDENTIK
        `HPPOPolicy._encode_hist`, TAMBAH gerbang ablasi `use_hist` (lihat __init__)."""
        if not self.use_hist:
            return torch.zeros(hist.shape[0], self.hist_hidden, device=hist.device, dtype=hist.dtype)
        _, (h_n, _) = self.hist_lstm(hist)
        return h_n[-1]

    def _split_station_block(self, flat, feat_dim, scalar_dim):
        """flat: (B, feat_dim*N) FEATURE-MAJOR -- lihat `MasterEVPPORolloutAgent` utk
        konversi dari (N,feat_dim) station-major (sama konvensi `MasterStationPPOPolicy`)."""
        scalars = flat[:, :scalar_dim]
        block = flat[:, scalar_dim:]
        n = block.shape[1] // feat_dim
        station_feats = block.view(-1, feat_dim, n).transpose(1, 2)
        return scalars, station_feats

    def forward(self, obs, hist=None, critic_obs=None):
        """`hist`: (B,K,HIST_FEAT_DIM) WAJIB (lihat `MasterEVPPORolloutAgent.
        get_recommendation` -- riwayat SUNGGUHAN via `_build_hist`, bukan lagi dummy).
        `critic_obs` diabaikan bila None -- kritik memakai `obs` YANG SAMA dgn aktor
        (V(s) sederhana, bukan CTDE penuh), TAPI TETAP melihat `c_t` (bukan privileged,
        sudah terlihat aktor juga -- sama pola `HPPOPolicy`)."""
        c_t = self._encode_hist(hist)
        _, station_feats = self._split_station_block(obs, STATION_FEAT_DIM_MASTER_EV, 0)

        emb = self.station_encoder(station_feats, c_t)
        logits = self.disc_head(emb).squeeze(-1)             # (B,N)

        c_emb = self.critic_station_encoder(station_feats, c_t)
        pooled = self.critic_pool(c_emb)
        value = self.critic_head(pooled)                     # (B,n_critics)
        return logits, value

    def _fwd(self, obs, hist, critic_obs, pref_hist):
        """Panggil `forward()` polimorfik -- teruskan `pref_hist` HANYA bila subclass
        (`MasterEVPPOPrefPolicy`) benar2 punya modul preferensi. Pola SAMA PERSIS
        `_BiddingMixin._fwd` (master_bidding_policy.py) -- disalin, bukan diwarisi,
        krn kelas ini basisnya nn.Module langsung (bukan _BiddingMixin)."""
        if pref_hist is not None and hasattr(self, "pref_lstm"):
            return self.forward(obs, hist, critic_obs, pref_hist=pref_hist)
        return self.forward(obs, hist, critic_obs)

    # ---------------- act()/evaluate(): IDENTIK `HPPOPolicy` (policy.py), disalin ----
    # apa adanya (bukan re-implementasi) -- top-K/threshold+epsilon-greedy, log-prob
    # Categorical berurutan-tanpa-pengembalian. Lihat `policy.py::HPPOPolicy.act` utk
    # penjelasan lengkap tiap cabang; komentar di sini dipangkas (tak diduplikasi).
    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 3, critic_obs_np=None,
           epsilon: float = 0.0, threshold: float = 0.20, pref_hist_np=None, **_abaikan):
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                     if critic_obs_np is not None else None)

        pref_hist = (torch.as_tensor(pref_hist_np, dtype=torch.float32).unsqueeze(0)
                    if pref_hist_np is not None else None)
        logits, value = self._fwd(obs, hist, critic_obs, pref_hist)
        if not torch.isfinite(logits).all():
            warnings.warn("MasterEVPPOPolicy.act: logits non-finite (NaN/Inf).", RuntimeWarning)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=NEG_INF)

        feasible_idx = torch.nonzero(mask[0], as_tuple=True)[0]
        n_feasible = int(feasible_idx.numel())
        k_eff = max(1 if n_feasible > 0 else 0, min(int(k), n_feasible))

        import random as _random
        if n_feasible > 0 and _random.random() < epsilon:
            explore_size = _random.randint(1, k_eff)
            perm = feasible_idx[torch.randperm(n_feasible)][:explore_size]
            chosen_order = perm.tolist()
        else:
            masked_logits = logits[0].masked_fill(~mask[0], NEG_INF)
            probs = torch.softmax(masked_logits, dim=-1)
            above = feasible_idx[probs[feasible_idx] > threshold]
            if above.numel() == 0:
                above = feasible_idx[probs[feasible_idx].argmax().unsqueeze(0)]
            if above.numel() > k_eff:
                top = torch.topk(probs[above], k_eff).indices
                above = above[top]
            order = torch.argsort(probs[above], descending=True)
            chosen_order = above[order].tolist()

        remaining_mask = mask[0].clone()
        logp_total = 0.0
        for idx in chosen_order:
            masked_logits_j = logits[0].masked_fill(~remaining_mask, NEG_INF)
            dist_j = torch.distributions.Categorical(logits=masked_logits_j)
            idx_t = torch.as_tensor(idx)
            logp_total = logp_total + dist_j.log_prob(idx_t)
            remaining_mask[idx] = False

        return {
            "chosen_indices": [int(i) for i in chosen_order],
            "n_rec": len(chosen_order),
            "logp": float(logp_total) if isinstance(logp_total, float) else float(logp_total.item()),
            "value": value[0].detach().cpu().numpy().astype("float64"),
        }

    def evaluate(self, obs_b, mask_b, chosen_indices_b, n_rec_b, hist_b, critic_obs_b=None,
                pref_hist_b=None, **_abaikan):
        logits, value = self._fwd(obs_b, hist_b, critic_obs_b, pref_hist_b)
        logits = logits.masked_fill(~mask_b, NEG_INF)

        B, k = chosen_indices_b.shape
        remaining_mask = mask_b.clone()
        logp = torch.zeros(B, device=logits.device)
        entropy = torch.zeros(B, device=logits.device)
        for j in range(k):
            valid_j = j < n_rec_b
            masked_logits = logits.masked_fill(~remaining_mask, NEG_INF)
            dist_disc = torch.distributions.Categorical(logits=masked_logits)
            idx_j = chosen_indices_b[:, j]
            logp_disc_j = dist_disc.log_prob(idx_j)
            ent_disc_j = dist_disc.entropy()

            logp = logp + torch.where(valid_j, logp_disc_j, torch.zeros_like(logp_disc_j))
            entropy = entropy + torch.where(valid_j, ent_disc_j, torch.zeros_like(ent_disc_j))
            remove = torch.zeros_like(remaining_mask)
            remove.scatter_(1, idx_j.unsqueeze(1), True)
            remaining_mask = remaining_mask & ~(remove & valid_j.unsqueeze(1))

        return logp, entropy, value


class MasterEVPPOPrefPolicy(MasterEVPPOPolicy):
    """`MasterEVPPOPolicy` + modul ekstraksi preferensi PDQN (`pref_lstm` +
    `PreferenceAttention` + `pref_gate`) -- pola SAMA PERSIS `MasterStationPPOPrefPolicy`/
    `PPPOPolicy`, disuntikkan sbg konteks tambahan yg di-broadcast ke seluruh kandidat
    stasiun saat encoding.

    BEDA PENTING dgn `MasterStationPPOPrefPolicy`: di sana P dianggap "melanggar" §3.1
    murni krn stasiun SEHARUSNYA buta terhadap pemohon (Pers. 11 MASTER). Di sini TIDAK
    ADA tegangan itu -- unit agen MEMANG permintaan EV itu sendiri, jadi riwayat
    (a_hat,a) MILIK SATU EV YANG SAMA yang sedang membuat keputusan adalah identitas
    yang alami, bukan tempelan. Ini pengujian ULANG hipotesis "P gagal krn identitas-
    ambigu di perspektif stasiun" (5x gagal sebelumnya, SEMUA di arsitektur stasiun-
    sbg-agen atau kritik-per-timestep yg belum stabil) pada kondisi yg sudah dibersihkan
    dari KEDUA confound itu sekaligus."""

    def __init__(self, n_spklu: int, hidden: int = 64, critic_hidden: int = 128,
                n_critics: int = 1, pref_d_lstm: int = PREF_D_LSTM, pref_d_attn: int = PREF_D_ATTN,
                pref_feature_mode: bool = False, use_preference: bool = True,
                use_hist: bool = True):
        super().__init__(n_spklu, hidden=hidden, critic_hidden=critic_hidden, n_critics=n_critics,
                         use_hist=use_hist)
        self.pref_feature_mode = bool(pref_feature_mode)
        self.use_preference = bool(use_preference)
        self.pref_d_attn = int(pref_d_attn)
        self.pref_hist_feat_dim = (hist_feat_dim_feature(PREF_STATION_FEAT_DIM)
                                   if self.pref_feature_mode else hist_feat_dim(n_spklu))
        self.pref_lstm = nn.LSTM(self.pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(STATION_FEAT_DIM_MASTER_EV, pref_d_lstm, pref_d_attn)
        # Gerbang nol-awal (GTrXL/Parisotto dkk. 2019) -- modul baru diinisialisasi acak,
        # disuntik penuh langsung akan jadi derau besar bagi station_encoder yg sedianya
        # belajar lancar tanpa itu (sama alasan seluruh kelas +P lain di repo ini).
        self.pref_gate = nn.Parameter(torch.tensor(0.0))
        # GANTI station_encoder: context_dim = hist_hidden (c_t, riwayat compliance --
        # SUDAH ADA di kelas dasar sejak perbaikan 2026-08-21) + pref_d_attn (preferensi
        # eksplisit) -- pola SAMA `PPPOPolicy` (H-PPO+P).
        self.station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV,
                                              self.hist_hidden + pref_d_attn, hidden)
        # PERBAIKAN (2026-08-21): kritik SEBELUMNYA hanya diberi `c_t`, buta terhadap
        # `attended_pref` -- V(s) tak bisa "menjelaskan" bagian nilai yg didorong modul P,
        # membuat estimasi advantage utk aksi yg dipengaruhi P jadi bias/kurang akurat.
        # `attended_pref` BUKAN privileged (aktor sendiri sudah melihatnya) -- aman
        # diteruskan ke kritik, sama prinsip CTDE penuh `HPPOPolicy`.
        self.critic_station_encoder = StationEncoder(STATION_FEAT_DIM_MASTER_EV,
                                                      self.hist_hidden + pref_d_attn, critic_hidden)

    def _encode_pref(self, pref_hist):
        """PERBAIKAN (2026-08-21): pakai `pack_padded_sequence` -- `pref_lstm` MELEWATI
        langkah padding-nol sepenuhnya (bukan memprosesnya sbg input nol yg tetap
        mengubah cell-state), menghilangkan dilusi-padding TANPA harus memperpendek
        jendela `pref_hist_k` (lihat `MasterEVPPORolloutAgent._build_pref_hist` --
        padding kini di BELAKANG, syarat API ini). Panjang riwayat SUNGGUHAN tiap
        baris dihitung dari baris bukan-nol (aman: fitur stasiun asli hampir tak pernah
        PERSIS nol di semua dimensi sekaligus -- jarak/wait/antrean/konektor/utilisasi)."""
        if not self.use_preference:
            return torch.zeros(pref_hist.shape[0], self.pref_lstm.hidden_size,
                               device=pref_hist.device, dtype=pref_hist.dtype)
        lengths = (pref_hist.abs().sum(dim=-1) > 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(pref_hist, lengths, batch_first=True,
                                                    enforce_sorted=False)
        _, (h_n, _) = self.pref_lstm(packed)
        return h_n[-1]

    def forward(self, obs, hist=None, critic_obs=None, pref_hist=None):
        c_t = self._encode_hist(hist)
        _, station_feats = self._split_station_block(obs, STATION_FEAT_DIM_MASTER_EV, 0)

        if pref_hist is not None and self.use_preference:
            c_pref = self._encode_pref(pref_hist)
            attended_pref, _ = self.pref_attn(station_feats, c_pref)
            attended_pref = self.pref_gate * attended_pref
        else:
            attended_pref = torch.zeros(obs.shape[0], self.pref_d_attn, device=obs.device)

        context = torch.cat([c_t, attended_pref], dim=-1)
        emb = self.station_encoder(station_feats, context)
        logits = self.disc_head(emb).squeeze(-1)

        c_emb = self.critic_station_encoder(station_feats, context)
        pooled = self.critic_pool(c_emb)
        value = self.critic_head(pooled)
        return logits, value


class MasterEV3Transition:
    """Duplikat `rollout.py::Transition`, HANYA beda `reward_streams` berukuran 3
    (bukan `N_REWARD_STREAMS`=2 bersama) -- lihat catatan `STREAM_WAIT/PROX/GLOBAL3`
    di atas kenapa ini TERISOLASI di modul ini, bukan mengubah `Transition` bersama."""
    __slots__ = ("obs", "critic_obs", "hist", "mask", "chosen_indices", "n_rec",
                 "logp", "value", "step", "reward_streams", "done",
                 "complied", "disp_estwait", "wait_default", "resolved", "flock_penalty",
                 "pref_hist", "bids", "trust_before")

    def __init__(self, obs, critic_obs, hist, mask, chosen_indices, n_rec, logp, value, step,
                pref_hist=None):
        self.obs = obs; self.critic_obs = critic_obs; self.hist = hist; self.mask = mask
        self.chosen_indices = chosen_indices; self.n_rec = n_rec
        self.logp = logp
        self.value = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self.step = step
        self.reward_streams = np.zeros(3, dtype=np.float64)
        self.done = False
        self.flock_penalty = 0.0
        self.pref_hist = pref_hist
        self.bids = None
        self.resolved = False
        self.complied = False; self.disp_estwait = 0.0; self.wait_default = 0.0
        self.trust_before = 0.5

    @property
    def reward(self) -> float:
        return float(self.reward_streams.sum())

    def reward_vec(self, n_critics: int) -> np.ndarray:
        if n_critics == 1:
            return np.array([self.reward_streams.sum()], dtype=np.float64)
        if n_critics != 3:
            raise ValueError(f"n_critics={n_critics} != 3 (MasterEV3Transition)")
        return self.reward_streams

    def add_reward(self, value: float, stream: int = STREAM_WAIT) -> None:
        self.reward_streams[stream] += float(value)


class MasterEVPPORolloutAgent(RLRolloutAgent):
    """Override `get_recommendation` -- observasi §3.1+state-EV via
    `build_joint_obs_master_ev`. `on_decision`/`on_step_end`/`on_charge_complete`
    JUGA DIOVERRIDE (2026-08-21) -- HANYA utk memilih indeks aliran reward yang benar
    (2 aliran BAKU vs 3 aliran Prox-terpisah); logika perhitungan reward itu sendiri
    IDENTIK PERSIS `RLRolloutAgent` (disalin, bukan ditulis ulang dari nol).

    `_split_prox` (dideteksi OTOMATIS dari `policy.n_critics==3`, TAK PERNAH diketik
    manual di titik panggil -- sama pola `pref_feature_mode` auto-derive di kelas lain
    supaya latih & uji tak pernah bisa berbeda mode): True -> Prox & perbaikan-wait
    dipisah ke `STREAM_PROX`/`STREAM_WAIT` (kritik 3-kepala WAJIB, `MasterEV3Transition`
    dipakai). False -> perilaku LAMA (2 aliran, `Transition` biasa, TAK BERUBAH).

    `pref_hist_k` (2026-08-21, ABLASI): panjang jendela riwayat P yang DIAMBIL saat
    membangun `pref_hist` -- TERISOLASI PENUH dari `PDQN_HIST_K`=10 bersama (dipakai
    PDQN diskrit sesuai spesifikasi §3.4 paper & lengan +P lain apa adanya, TAK
    disentuh). Baku tetap 10 (perilaku lama persis). Dugaan yg diuji: bagi pengguna
    dgn riwayat pendek (median interaksi rendah di domain ini), jendela 10 didominasi
    padding-nol (~80% utk pengguna 2 interaksi) -- 2x lebih parah dari `hist_lstm`
    (HIST_K=5, ~40% padding utk kasus sama) -- berpotensi mengencerkan sinyal `pref_lstm`
    lebih jauh drpd `hist_lstm`."""

    def __init__(self, *args, pref_hist_k: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_prox = (int(getattr(self.policy, "n_critics", 1)) == 3)
        self.pref_hist_k = int(pref_hist_k) if pref_hist_k is not None else PDQN_HIST_K

    def _build_pref_hist(self, user):
        """Override `RLRolloutAgent._build_pref_hist` -- BEDA PENTING (2026-08-21):
        padding-nol di sini ditaruh DI BELAKANG (bukan di depan spt versi asli) --
        RIGHT-padding, syarat `nn.utils.rnn.pack_padded_sequence` (dipakai `_encode_pref`
        supaya `pref_lstm` MELEWATI langkah padding sepenuhnya, bukan memprosesnya sbg
        nol -- lihat diskusi "penyebabnya karena pooling?"/dilusi-padding). Panjang
        jendela `self.pref_hist_k` (bisa < `PDQN_HIST_K`) tetap berlaku. Deque
        penyimpanan (`self._pref_hist`, `_record_pref`) DIWARISI TANPA PERUBAHAN."""
        k = self.pref_hist_k
        arr = np.zeros((k, self._pref_hist_feat_dim), dtype=np.float32)
        h = self._pref_hist.get(user.user_id)
        if h:
            recent = list(h)[-k:]
            for t, pair in enumerate(recent):
                if self.pref_feature_mode:
                    arr[t] = pair
                else:
                    a_hat_idx, a_idx = pair
                    arr[t, a_hat_idx] = 1.0
                    arr[t, self.N + a_idx] = 1.0
        return arr

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        if not self._split_prox:
            return super().on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        if self._pending is None:
            return
        tr, _, primary_idx, wait_hat, default_idx, recent_rec_count = self._pending
        complied = chosen_spklu_id in set(recs)
        tr.complied = bool(complied)
        tr.trust_before = float(user.trust)
        # Suku shaping acceptance (opsional, BAKU MATI) -- masuk STREAM_GLOBAL3 (BUKAN
        # STREAM_WAIT lagi, lihat diagnosis 2026-08-21: acceptance_reward ber-magnitudo
        # +-1 SEGERA tiap keputusan menumpuk di stream yg sama dgn wait_reward yg kecil &
        # delayed, membuat r_bar STREAM_WAIT meledak 10-30x drpd PROX & beberapa x drpd
        # GLOBAL -- diduga penyebab dominan wait/gini liar di eval K3+P+accept 90d).
        # GLOBAL3 dipilih krn acceptance scr konseptual lebih dekat "penilaian
        # populasi/sistem" (simetris, memengaruhi keputusan agen scr keseluruhan) drpd
        # wait individual murni.
        if self.rc.alpha_accept != 0.0:
            tr.add_reward(self.rc.acceptance_reward(complied), STREAM_GLOBAL3)
        feat_rec = self._station_feat(self.sids[primary_idx], wait_hat)
        feat_chosen = self._station_feat(chosen_spklu_id, wait_hat)
        prox_value = self.rc.prox(feat_rec, feat_chosen)
        tr.add_reward(self.rc.decision_reward(prox_value), STREAM_PROX)
        flock_term = self.rc.flock_reward_rolling(recent_rec_count)
        tr.add_reward(flock_term, STREAM_GLOBAL3)
        tr.flock_penalty = self.rc.flocking_penalty_rolling(recent_rec_count)
        if self.equity_calc is not None:
            chosen_idx_eq = self.sid_to_idx.get(chosen_spklu_id)
            if chosen_idx_eq is not None:
                utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
                feasible_idx = np.nonzero(tr.mask)[0]
                tr.add_reward(self.equity_calc.decision_reward_equity(
                    utils, feasible_idx, primary_idx, default_idx, chosen_idx_eq,
                    recent_rec_count=recent_rec_count), STREAM_GLOBAL3)
        if self._use_pref:
            chosen_idx = self.sid_to_idx.get(chosen_spklu_id)
            if chosen_idx is not None:
                self._record_pref(user, primary_idx, chosen_idx, wait_hat)
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
        if not self._split_prox:
            return super().on_step_end(sim, step)
        transitions_this_step = [tr for tr in self.transitions if tr.step == step]
        if not transitions_this_step:
            return
        utils = np.array([s.get_utilization() for s in self.sim.spklus.values()])
        gini_term = self.rc.gini_reward(utils)
        for tr in transitions_this_step:
            tr.add_reward(gini_term, STREAM_GLOBAL3)

    def on_charge_complete(self, user):
        if not self._split_prox:
            return super().on_charge_complete(user)
        tr = self._user_trip_tr.pop(user.user_id, None)
        if tr is not None:
            if tr.complied:
                tr.add_reward(
                    self.rc.wait_reward(tr.wait_default, user.wait_time, tr.disp_estwait),
                    STREAM_WAIT)
                if self.rc.alpha_trust != 0.0:
                    delta_trust = float(user.trust) - tr.trust_before
                    tr.add_reward(self.rc.trust_shaping_reward(delta_trust), STREAM_WAIT)
            tr.resolved = True
            if tr.complied and user.interaction_history:
                complied_v, disp_v, wdef_v, _ = user.interaction_history[-1]
                realized_gap_norm = (user.wait_time - tr.disp_estwait) / self.wait_scale
                user.interaction_history[-1] = (complied_v, disp_v, wdef_v, realized_gap_norm)

    def get_recommendation(self, feasible_spklus: dict):
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        _rich_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)

        joint_obs = build_joint_obs_master_ev(self.sim, self.sids, time_now, user, soc)   # (N,10)
        obs_flat = joint_obs.T.reshape(-1).astype(np.float32)   # feature-major, konvensi _split_station_block
        mask = self._feasible_mask(feasible_ids)
        # Riwayat interaksi SUNGGUHAN (2026-08-21, PERBAIKAN -- SEBELUMNYA dummy nol,
        # lihat docstring `MasterEVPPOPolicy.hist_lstm`) -- `_build_hist` DIWARISI dari
        # RLRolloutAgent, sama persis dipakai HPPOPolicy/P-PPO.
        hist = self._build_hist(user)

        # Modul preferensi (opsional): _use_pref/_build_pref_hist DIWARISI dari
        # RLRolloutAgent.__init__ -- pola sama `MasterStationPPORolloutAgent`.
        if self._use_pref:
            pref_hist = self._build_pref_hist(user)
            act = self.policy.act(obs_flat, mask, hist, k=self.k, critic_obs_np=obs_flat,
                                  epsilon=self.epsilon, threshold=self.threshold,
                                  pref_hist_np=pref_hist)
        else:
            pref_hist = None
            act = self.policy.act(obs_flat, mask, hist, k=self.k, critic_obs_np=obs_flat,
                                  epsilon=self.epsilon, threshold=self.threshold)

        chosen_indices = act["chosen_indices"]
        n_rec = act["n_rec"]
        recs = [self.sids[i] for i in chosen_indices]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_indices}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}

        primary_idx = chosen_indices[0] if chosen_indices else int(np.nonzero(mask)[0][0])
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)

        idx_arr = np.zeros(self.k, dtype=np.int64)
        idx_arr[:n_rec] = chosen_indices

        TransitionCls = MasterEV3Transition if self._split_prox else Transition
        tr = TransitionCls(obs_flat, obs_flat, hist, mask, idx_arr, n_rec,
                           act["logp"], act["value"], self.sim.current_step,
                           pref_hist=pref_hist)
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs


class MasterEVPPOInferenceAgent:
    """Evaluasi bersih -- pola sama `MasterStationPPOInferenceAgent`. `pref_feature_mode`
    DIAMBIL OTOMATIS dari `policy.pref_feature_mode` (bila ada) -- latih & uji tak pernah
    bisa berbeda mode encoding riwayat (kelas bug berulang, lihat docstring kelas itu).

    `pref_hist_k`: BEDA dari `pref_feature_mode` -- TAK BISA diambil otomatis dari
    `policy` (bukan bagian bentuk jaringan, murni hyperparameter sisi rollout-agent,
    LSTM menerima panjang urutan berapa pun) -- WAJIB diteruskan MANUAL sesuai nilai
    saat latih, kalau tidak evaluasi TETAP JALAN (tak crash) tapi diam-diam SALAH
    (jendela riwayat tak cocok checkpoint)."""

    def __init__(self, policy, sim, forecaster=None, k: int = 3, epsilon: float = 0.0,
                threshold: float = 0.20, pref_hist_k: int = None):
        pref_feature_mode = bool(getattr(policy, "pref_feature_mode", False))
        self._roll = MasterEVPPORolloutAgent(policy, sim, RewardCalculatorStub(), forecaster, k=k,
                                             pref_feature_mode=pref_feature_mode,
                                             pref_hist_k=pref_hist_k)
        self._roll.epsilon = epsilon
        self._roll.threshold = threshold

    def get_recommendation(self, feasible_spklus):
        return self._roll.get_recommendation(feasible_spklus)

    def predict_waits(self, feasible_spklus):
        return self._roll.predict_waits(feasible_spklus)

    def on_decision(self, user, chosen_spklu_id, recs, feasible_spklus):
        self._roll.on_decision(user, chosen_spklu_id, recs, feasible_spklus)
        self._roll.transitions.clear()

    def on_charge_complete(self, user):
        self._roll.on_charge_complete(user)
        self._roll.transitions.clear()


class MasterEVPPOTrainer:
    """Trainer continuing-task PPO/GAE/clip -- pola SAMA PERSIS `MasterStationPPOTrainer`
    (satu-satunya penyimpangan dari `TorchContinuingTrainer` generik: carry-forward trust
    lintas pass, dipertahankan sengaja agar sepadan dgn seluruh keluarga MASTER lain)."""

    def __init__(self, dataset_path, rollout_steps: int = 96, seed: int = 0,
                verbose: bool = True, reward_calc=None, hidden: int = 64,
                critic_hidden: int = 128, k: int = 3, n_critics: int = 1,
                equity_calc=None, policy_cls=MasterEVPPOPolicy, policy_kw=None,
                pref_hist_k: int = None, **ppo_kw):
        self.dataset_path = dataset_path
        self.rollout_steps = int(rollout_steps)
        self.k = int(k)
        self.equity_calc = equity_calc
        self.pref_hist_k = pref_hist_k
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim0 = self._fresh_sim()
        self.N = len(sim0.spklus)
        self.policy = policy_cls(self.N, hidden=hidden, critic_hidden=critic_hidden,
                                 n_critics=n_critics, **(policy_kw or {}))
        self.ppo = PPOTrainer(self.policy, avg_reward=True, **ppo_kw)
        self.history = []
        self._it_global = 0
        self._logger = _make_logger(verbose)

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path)

    def _reset_rc(self):
        if hasattr(self.rc, "reset_episode_state"):
            self.rc.reset_episode_state()

    def _carry_forward(self, sim, agent):
        old_users_by_id = {u.user_id: u for u in sim.users}
        new_sim = self._fresh_sim()
        for u in new_sim.users:
            old = old_users_by_id.get(u.user_id)
            if old is not None:
                u.trust_alpha = old.trust_alpha
                u.trust_beta = old.trust_beta
                u.compliance_history = list(old.compliance_history)
                u.interaction_history = list(old.interaction_history)
        agent.sim = new_sim
        agent.sids = list(new_sim.spklus.keys())
        agent.sid_to_idx = {s: i for i, s in enumerate(agent.sids)}
        agent.transitions = []
        agent._pending = None
        agent._user_trip_tr = {}
        agent._prev_gini = None
        return new_sim

    def train(self, forecaster, n_updates: int):
        chunk = self.rollout_steps
        sim = self._fresh_sim()
        self._reset_rc()
        pref_feature_mode = bool(getattr(self.policy, "pref_feature_mode", False))
        agent = MasterEVPPORolloutAgent(self.policy, sim, self.rc, forecaster, k=self.k,
                                        equity_calc=self.equity_calc,
                                        pref_feature_mode=pref_feature_mode,
                                        pref_hist_k=self.pref_hist_k)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            boundary = False
            for _ in range(chunk):
                sim.step_once(step, agent=agent)
                step += 1
                if step >= sim.max_steps:
                    boundary = True
                    break
            resolved = [t for t in agent.transitions if t.resolved]
            pending = [t for t in agent.transitions if not t.resolved]
            self._it_global += 1
            if resolved:
                resolved[-1].done = boundary
                stats = self.ppo.update(resolved)
                rewards = np.array([t.reward for t in resolved])
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                trusts = np.array([u.trust for u in sim.users], float)
                info = {"mean_reward": float(rewards.mean()),
                       "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                       "gini_served": _gini(served), "n_tr": len(resolved),
                       "n_pending": len(pending), "trust_mean": float(trusts.mean())}
                rec = {"iter": it, **info, **stats}
                self.history.append(rec)
                if self.verbose:
                    self._logger.info(
                        "[chunk %3d] R=%+.4f accept=%.2f | KL=%.4f EV=%.2f | gini=%.3f "
                        "trust=%.3f n=%d pend=%d%s", it, info["mean_reward"],
                        info["acceptance_rate"], stats.get("approx_kl", 0),
                        stats.get("explained_var", 0), info["gini_served"],
                        info["trust_mean"], info["n_tr"], info["n_pending"],
                        " |PASS-BARU(carry-fwd)" if boundary else "")
            if boundary:
                sim = self._carry_forward(sim, agent)
                self._reset_rc()
                step = 0
            else:
                agent.transitions = pending
        return self.policy
