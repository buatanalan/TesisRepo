"""P-PPO -- H-PPO v2 (`policy.py::HPPOPolicy`) + modul "P" PDQN (Lin dkk. 2024 §IV.A:
LSTM atas riwayat pasangan (a_hat,a) + attention preferensi<->state), TANPA mengambil
bagian LAIN dari PDQN (bukan off-policy/DQN, bukan replay buffer, bukan dekomposisi nilai
VDN/QMIX Q_tot=Q(i)+Q(j)) -- lihat diskusi sesi: "adopsi bagian P-nya saja, digabung PPO".

KENAPA INI: PDQN QMix (pdqn_advanced_policy.py) sudah menguras perbaikan REPRESENTASI
value-based (shared-weight, dueling, mixing non-linier) -- masih kalah dari greedy
(§2.1-2.3 Kajian_Literatur_Solusi_Domain_Serupa.md). Sementara H-PPO SUDAH MENANG lawan
greedy (config `gabungan`), TAPI tak pernah punya modul preferensi eksplisit PDQN (LSTM
atas identitas STASIUN yang direkomendasikan-vs-dipilih) -- H-PPO cuma py `hist_lstm`
(riwayat SKALAR patuh/estwait, TANPA identitas stasiun, murni proksi trust POMDP). P-PPO
menguji: apakah menambah sinyal preferensi eksplisit itu ke arsitektur H-PPO yang SUDAH
menang bisa memperbaikinya LEBIH JAUH lagi (bukan usaha menyelamatkan PDQN yang kalah).

WARISAN DARI H-PPO (TAK DIUBAH): StationEncoder shared-weight, AttentionPooling critic,
CTDE penuh, on-policy PPO+GAE, baseline/advantage, threshold+ceiling k=2, avg-reward
continuing task. WARISAN DARI PDQN (DIAMBIL, "bagian P" saja): `PreferenceAttention` +
`pref_lstm` atas pasangan (a_hat,a) -- KEDUANYA diimpor langsung dari `pdqn_policy.py`,
bukan ditulis ulang.

INFRASTRUKTUR DATA SUDAH ADA & TAK PERLU DIUBAH (dibangun sesi jauh sebelumnya, sempat
idle krn belum ada policy yg mengaktifkannya):
  - `rollout.py::RLRolloutAgent._use_pref = hasattr(policy, "pref_lstm")` -- deteksi
    otomatis, tinggal kelas ini py atribut itu.
  - `rollout.py::_build_pref_hist`/`_record_pref` -- riwayat (a_hat=chosen_indices[0],
    a=stasiun dipilih pengguna) SUDAH direkam di `on_decision` per pengguna.
  - `ppo.py::PPOTrainer.update` -- `use_pref = hasattr(self.policy, "pref_lstm")`,
    otomatis menyusun & meneruskan `pref_hist_b` ke `evaluate()` bila terdeteksi.
Jadi py `pref_lstm` di kelas ini SAJA sudah cukup mengaktifkan seluruh jalur data --
TIDAK ADA perubahan lain diperlukan di rollout.py/ppo.py.

CATATAN VALIDITAS (tak leaking): riwayat (a_hat,a) milik SATU pengguna yang sedang
diputuskan, direkam DARI KEPUTUSAN LAMPAUNYA SENDIRI (analog `hist_lstm` yg sudah ada) --
bukan variabel privileged (trust asli, dsb). Aman utk AKTOR (bukan cuma kritik)."""
import torch
import torch.nn as nn

from marl_spklu.rl.policy import (HPPOPolicy, StationEncoder, STATION_FEAT_DIM,
                                  CRITIC_STATION_FEAT_DIM, NEG_INF)
from marl_spklu.rl.pdqn_policy import PreferenceAttention, hist_feat_dim, hist_feat_dim_feature

# DIKECILKAN 2026-08-16 dari 64 (nilai paper Lin dkk., `pdqn_policy.py`) menjadi 16 --
# DISENGAJA menyimpang dari rujukan, demi perbandingan yang terkontrol.
#
# Alasan: dgn 64, P-PPO berbeda dari H-PPO dalam DUA hal sekaligus -- (a) informasi
# preferensi, dan (b) ~28.700 parameter aktif tambahan -- sehingga kalau hasilnya berbeda,
# kontribusi keduanya TAK BISA dipisahkan. Menyamakan skala dgn `hist_lstm` H-PPO
# (HIST_HIDDEN=16) mengecilkan selisih kapasitas jadi ~3.300 parameter (8,6x lebih kecil),
# sehingga selisih hasil bisa diatribusikan ke INFORMASI, bukan kapasitas.
#
# Efek samping yang juga diperbaiki: porsi `c_t` (proksi trust POMDP) dlm konteks aktor
# naik dari 19% ke ~42% -- sebelumnya `attended_pref` mengambil 74% konteks & mengencerkan
# sinyal trust 4x. Dim masukan `pref_lstm` (12 = 2N one-hot ganda) TAK BISA diubah;
# ia ditentukan representasi datanya.
PREF_D_LSTM = 16   # d_lstm modul preferensi -- disamakan dgn HIST_HIDDEN H-PPO
PREF_D_ATTN = 16   # d_attn modul preferensi


class PPPOPolicy(HPPOPolicy):
    """HPPOPolicy v2 + modul preferensi PDQN sbg KONTEKS TAMBAHAN utk `station_encoder`
    aktor (di samping `c_t` riwayat compliance yg sudah ada). Kritik TAK diberi akses
    tambahan (preferensi bukan variabel privileged yg perlu "dibocorkan" ke kritik --
    ia sudah tersedia utk aktor sendiri, CTDE tak relevan di sini).

    `use_preference=False` -> ablasi "P-PPO tanpa P" (attended_pref dipaksa nol, arsitektur
    & jumlah parameter TETAP SAMA) -- utk mengisolasi kontribusi modul preferensi murni
    dari kontribusi kapasitas parameter tambahan semata (sama prinsip PDQNQNetwork)."""

    def __init__(self, obs_dim: int, critic_obs_dim: int, n_spklu: int, hidden: int = 128,
                 hist_hidden: int = 16, station_hidden: int = 64,
                 pref_d_lstm: int = PREF_D_LSTM, pref_d_attn: int = PREF_D_ATTN,
                 pref_feature_mode: bool = False, use_preference: bool = True,
                 n_critics: int = 1):
        super().__init__(obs_dim, critic_obs_dim, n_spklu, hidden=hidden,
                         hist_hidden=hist_hidden, station_hidden=station_hidden,
                         n_critics=n_critics)
        self.pref_feature_mode = bool(pref_feature_mode)
        self.use_preference = bool(use_preference)
        self.pref_d_attn = int(pref_d_attn)
        self.pref_hist_feat_dim = (hist_feat_dim_feature() if pref_feature_mode
                                   else hist_feat_dim(n_spklu))
        self.pref_lstm = nn.LSTM(self.pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(STATION_FEAT_DIM, pref_d_lstm, pref_d_attn)
        # GATING berinisialisasi NOL (Parisotto dkk. 2019, "Stabilizing Transformers for RL"
        # / GTrXL): modul preferensi baru diinisialisasi ACAK -- disuntik LANGSUNG penuh ke
        # context, ia jadi NOISE besar di awal training (attended_pref blm belajar apa pun
        # berguna), merusak StationEncoder yg sedianya belajar lancar (TERBUKTI EMPIRIS,
        # percobaan P-PPO v1: Gini 0,0375->0,19). Gate skalar TERLATIH, mulai PERSIS 0 ->
        # attended_pref*gate=0 vektor -> kontribusinya ke StationEncoder BENAR2 NOL di awal
        # (perkalian matriks apa pun dgn vektor nol = nol, terlepas bobot W acaknya) --
        # setara start dari "H-PPO original" scr fungsional, modul preferensi baru
        # "menyalakan diri" sendiri via gradien SEIRING ia belajar berguna.
        self.pref_gate = nn.Parameter(torch.tensor(0.0))

        # GANTI station_encoder aktor: context_dim bertambah pref_d_attn (attended
        # preferensi) -- kritik TAK diubah (critic_station_encoder/critic_pool/critic_head
        # warisan HPPOPolicy apa adanya).
        actor_context_dim = self.scalar_dim + self.hist_hidden + pref_d_attn
        self.station_encoder = StationEncoder(STATION_FEAT_DIM, actor_context_dim, self.station_hidden)

    def _encode_pref(self, pref_hist):
        """pref_hist: (B,K,pref_hist_feat_dim) -> c_pref: (B,pref_d_lstm). phi:(a_hat,a)->p
        milik paper PDQN, IDENTIK `PDQNQNetwork._encode_pref` (versi n_types=1)."""
        if not self.use_preference:
            return torch.zeros(pref_hist.shape[0], self.pref_lstm.hidden_size,
                               device=pref_hist.device, dtype=pref_hist.dtype)
        _, (h_n, _) = self.pref_lstm(pref_hist)
        return h_n[-1]

    def forward(self, obs, hist, critic_obs=None, pref_hist=None):
        """Identik `HPPOPolicy.forward`, TAMBAH `pref_hist` (B,K,dim). None (mis. dipanggil
        kode lama yg tak sadar P-PPO) -> preferensi nol, setara ablasi `use_preference=False`."""
        c_t = self._encode_hist(hist)
        scalars, station_feats = self._split_station_block(obs, STATION_FEAT_DIM, self.scalar_dim)

        if pref_hist is not None and self.use_preference:
            c_pref = self._encode_pref(pref_hist)
            attended_pref, _ = self.pref_attn(station_feats, c_pref)
            attended_pref = self.pref_gate * attended_pref   # gating -- lihat __init__
        else:
            attended_pref = torch.zeros(obs.shape[0], self.pref_d_attn, device=obs.device)

        context = torch.cat([scalars, c_t, attended_pref], dim=-1)
        emb = self.station_encoder(station_feats, context)
        logits = self.disc_head(emb).squeeze(-1)

        if critic_obs is not None:
            c_scalars, c_station_feats = self._split_station_block(
                critic_obs, CRITIC_STATION_FEAT_DIM, self.critic_scalar_dim)
            merged_station_feats = torch.cat([station_feats, c_station_feats], dim=-1)
            merged_scalars = torch.cat([scalars, c_scalars], dim=-1)
            c_context = torch.cat([merged_scalars, c_t], dim=-1)
            c_emb = self.critic_station_encoder(merged_station_feats, c_context)
            pooled = self.critic_pool(c_emb)
            # (B, n_critics) -- TIDAK di-squeeze, seragam dgn HPPOPolicy.forward
            # (kritik ganda; utk n_critics=1 bentuknya (B,1)).
            value = self.critic_head(torch.cat([pooled, merged_scalars], dim=-1))
        else:
            value = torch.zeros(obs.shape[0], self.n_critics, device=obs.device)
        return logits, value

    # ---------------- Seleksi (rollout) -- IDENTIK HPPOPolicy.act, TAMBAH pref_hist_np ----
    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 3, critic_obs_np=None,
           epsilon: float = 0.0, threshold: float = 0.20, pref_hist_np=None):
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                     if critic_obs_np is not None else None)
        pref_hist = (torch.as_tensor(pref_hist_np, dtype=torch.float32).unsqueeze(0)
                    if pref_hist_np is not None else None)

        logits, value = self.forward(obs, hist, critic_obs, pref_hist)
        if not torch.isfinite(logits).all():
            import warnings
            warnings.warn("PPPOPolicy.act: keluaran non-finite (NaN/Inf) pada logits.",
                          RuntimeWarning)
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
            # array (K,) -- lihat HPPOPolicy.act (kritik ganda, n_critics).
            "value": value[0].detach().cpu().numpy().astype("float64"),
        }

    # ---------------- Recompute (update PPO) -- IDENTIK HPPOPolicy.evaluate, +pref_hist_b ----
    def evaluate(self, obs_b, mask_b, chosen_indices_b, n_rec_b, hist_b, critic_obs_b=None,
                pref_hist_b=None):
        logits, value = self.forward(obs_b, hist_b, critic_obs_b, pref_hist_b)
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
