"""PDQN KONTINU -- varian PDQN untuk kondisi trust DINAMIS (bukan dibekukan seperti PDQN
diskrit di pdqn_policy.py/pdqn_agent.py, yang sengaja mempertahankan trust statis sbg
justifikasi PDQN murni-pemerataan pada Tahap 0). Di sini rekomendasi berbentuk KONTINU:
janji EstWait (delta menit) untuk SETIAP SPKLU feasible sekaligus -- bukan memilih satu
SPKLU secara diskrit -- lalu pengguna memilih sendiri lewat mekanisme softmax(-gamma*wait)
yang SUDAH ADA (`rec_mode="estwait"`, User.decide_spklu). Trust berkembang natural via
`update_trust(est_wait, actual_wait)` -- inilah yang membuat setup ini PERFORMATIVE: janji
EstWait yang di-broadcast kebijakan mempengaruhi kepatuhan/trust populasi, yang pd gilirannya
mengubah distribusi keputusan yang dipelajari kebijakan berikutnya.

Desain: SUBCLASS `HPPOPolicy` (arsitektur PPO kontinu yang SUDAH teruji, dgn CTDE penuh,
trust dinamis, honesty-reward) -- TIDAK menulis ulang mekanisme itu. Yang DITAMBAHKAN cuma
kontribusi ASLI paper PDQN (Lin et al. 2024, §IV.A) yang belum ada di H-PPO: modul preferensi
LSTM atas pasangan (a_hat, a) + attention preferensi<->state, digabung sbg konteks TAMBAHAN
ke `station_encoder` di samping c_t (riwayat compliance) yang sudah dipakai H-PPO. Kepala
aksi (a1 diskrit top-k / a2 kontinu delta) TETAP warisan HPPOPolicy -- untuk "PDQN kontinu
murni" (tanpa seleksi diskrit, semua kandidat feasible dijanjikan sekaligus), panggil act()
dgn k = jumlah kandidat feasible penuh (lihat PDQNContinuousRolloutAgent).

CATATAN INTEGRASI (belum selesai, lihat percakapan): agar bisa dipakai `PPOTrainer.update()`
yang sudah ada, riwayat (a_hat,a) perlu direkam di rollout.py/Transition -- BELUM dilakukan
di sini supaya tidak menyentuh jalur H-PPO/MARL yang sudah tervalidasi tanpa persetujuan
eksplisit. Modul ini SIAP dipakai begitu jalur perekaman riwayat itu ditambahkan.
"""
import torch
import torch.nn as nn

from marl_spklu.rl.policy import HPPOPolicy, STATION_FEAT_DIM
from marl_spklu.rl.pdqn_policy import PreferenceAttention, hist_feat_dim, hist_feat_dim_feature

PREF_D_LSTM = 64    # d_lstm modul preferensi PDQN (selaras pdqn_policy.py)
PREF_D_ATTN = 64    # d_attn modul preferensi PDQN


class PDQNContinuousPolicy(HPPOPolicy):
    """HPPOPolicy + modul preferensi PDQN (LSTM(a_hat,a) + attention) sbg konteks
    tambahan utk station_encoder aktor. n_spklu, obs_dim, critic_obs_dim: identik makna
    dgn HPPOPolicy (lihat policy.py) -- TIDAK ada blok fitur station baru; PDQN kontinu
    memakai persis observasi H-PPO yang sudah ada (onehot,dist,wait_hat,queue,conn)."""

    def __init__(self, obs_dim: int, critic_obs_dim: int, n_spklu: int, hidden: int = 128,
                 delta_max: float = 10.0, hist_hidden: int = 16, station_hidden: int = 64,
                 pref_d_lstm: int = PREF_D_LSTM, pref_d_attn: int = PREF_D_ATTN,
                 pref_feature_mode: bool = False):
        super().__init__(obs_dim, critic_obs_dim, n_spklu, hidden=hidden,
                         delta_max=delta_max, hist_hidden=hist_hidden,
                         station_hidden=station_hidden)
        # pref_feature_mode=True: pasangan (a_hat,a) sbg VEKTOR FITUR stasiun, bukan
        # one-hot identitas -- lihat pdqn_policy.py::hist_feat_dim_feature.
        self.pref_feature_mode = bool(pref_feature_mode)
        self.pref_hist_feat_dim = (hist_feat_dim_feature() if pref_feature_mode
                                   else hist_feat_dim(n_spklu))   # 2N: onehot(a_hat)++onehot(a)
        self.pref_lstm = nn.LSTM(self.pref_hist_feat_dim, pref_d_lstm, batch_first=True)
        self.pref_attn = PreferenceAttention(STATION_FEAT_DIM, pref_d_lstm, pref_d_attn)

        # Ganti kepala aktor: context_dim aktor bertambah pref_d_attn (attended preferensi).
        actor_context_dim = self.scalar_dim + self.hist_hidden + pref_d_attn
        from marl_spklu.rl.policy import StationEncoder
        self.station_encoder = StationEncoder(STATION_FEAT_DIM, actor_context_dim, self.station_hidden)

    def _encode_pref(self, pref_hist):
        """pref_hist: (B,K,2N) -> c_pref: (B,pref_d_lstm). phi:(a_hat,a)->p milik paper,
        lihat pdqn_policy.py::PDQNQNetwork._encode_pref (versi n_types=1)."""
        _, (h_n, _) = self.pref_lstm(pref_hist)
        return h_n[-1]

    def forward(self, obs, hist, critic_obs=None, pref_hist=None):
        """Sama seperti HPPOPolicy.forward, TAMBAH argumen `pref_hist` (B,K,2N). Bila
        None (mis. dipanggil dari kode lama yg belum sadar PDQN), preferensi diperlakukan
        nol -- setara ablasi 'tanpa modul preferensi' (paper §V baseline ketiga)."""
        c_t = self._encode_hist(hist)
        scalars, station_feats = self._split_station_block(obs, STATION_FEAT_DIM, self.scalar_dim)

        if pref_hist is not None:
            c_pref = self._encode_pref(pref_hist)
            attended_pref, _ = self.pref_attn(station_feats, c_pref)
        else:
            attended_pref = torch.zeros(obs.shape[0], self.pref_attn.d_attn, device=obs.device)

        context = torch.cat([scalars, c_t, attended_pref], dim=-1)
        emb = self.station_encoder(station_feats, context)

        logits = self.disc_head(emb).squeeze(-1)
        mean = torch.tanh(self.cont_mean_head(emb).squeeze(-1)) * self.delta_max
        logstd = self.cont_logstd_head(emb).squeeze(-1)

        if critic_obs is not None:
            from marl_spklu.rl.policy import CRITIC_STATION_FEAT_DIM
            c_scalars, c_station_feats = self._split_station_block(
                critic_obs, CRITIC_STATION_FEAT_DIM, self.critic_scalar_dim)
            merged_station_feats = torch.cat([station_feats, c_station_feats], dim=-1)
            merged_scalars = torch.cat([scalars, c_scalars], dim=-1)
            c_context = torch.cat([merged_scalars, c_t], dim=-1)
            c_emb = self.critic_station_encoder(merged_station_feats, c_context)
            pooled = self.critic_pool(c_emb)
            value = self.critic_head(torch.cat([pooled, merged_scalars], dim=-1)).squeeze(-1)
        else:
            value = torch.zeros(obs.shape[0], device=obs.device)
        return logits, mean, logstd, value

    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 1, critic_obs_np=None,
           pref_hist_np=None):
        """Identik HPPOPolicy.act, TAMBAH pref_hist_np (K,2N). Untuk 'PDQN kontinu murni'
        (janji EstWait ke SEMUA kandidat feasible, bukan top-k terpilih), panggil dgn
        k >= jumlah kandidat feasible -- act() otomatis membatasi k_eff = n_feasible."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                     if critic_obs_np is not None else None)
        pref_hist = (torch.as_tensor(pref_hist_np, dtype=torch.float32).unsqueeze(0)
                    if pref_hist_np is not None else None)

        logits, mean, logstd, value = self.forward(obs, hist, critic_obs, pref_hist)

        remaining_mask = mask[0].clone()
        n_feasible = int(remaining_mask.sum().item())
        k_eff = max(1 if n_feasible > 0 else 0, min(int(k), n_feasible))
        std_all = torch.exp(logstd[0]).clamp(1e-3, self.delta_max)

        chosen_indices, deltas = [], []
        logp_total = 0.0
        for _ in range(k_eff):
            masked_logits = logits[0].masked_fill(~remaining_mask, -1e9)
            dist_disc = torch.distributions.Categorical(logits=masked_logits)
            idx = dist_disc.sample()
            logp_total = logp_total + dist_disc.log_prob(idx)
            remaining_mask[idx] = False

            dist_cont = torch.distributions.Normal(mean[0, idx], std_all[idx])
            delta_sample = dist_cont.sample()
            logp_total = logp_total + dist_cont.log_prob(delta_sample)

            chosen_indices.append(int(idx.item()))
            deltas.append(float(delta_sample.clamp(-self.delta_max, self.delta_max).item()))

        return {
            "chosen_indices": chosen_indices, "deltas": deltas, "n_rec": k_eff,
            "logp": float(logp_total) if isinstance(logp_total, float) else float(logp_total.item()),
            "value": float(value.item()),
        }

    def evaluate(self, obs_b, mask_b, chosen_indices_b, delta_b, n_rec_b, hist_b,
                critic_obs_b=None, pref_hist_b=None):
        """Identik HPPOPolicy.evaluate, TAMBAH pref_hist_b (B,K,2N)."""
        logits, mean, logstd, value = self.forward(obs_b, hist_b, critic_obs_b, pref_hist_b)
        logits = logits.masked_fill(~mask_b, -1e9)
        std = torch.exp(logstd).clamp(1e-3, self.delta_max)

        B, k = chosen_indices_b.shape
        remaining_mask = mask_b.clone()
        logp = torch.zeros(B, device=logits.device)
        entropy = torch.zeros(B, device=logits.device)
        for j in range(k):
            valid_j = j < n_rec_b
            masked_logits = logits.masked_fill(~remaining_mask, -1e9)
            dist_disc = torch.distributions.Categorical(logits=masked_logits)
            idx_j = chosen_indices_b[:, j]
            logp_disc_j = dist_disc.log_prob(idx_j)
            ent_disc_j = dist_disc.entropy()

            mean_j = torch.gather(mean, 1, idx_j.unsqueeze(1)).squeeze(1)
            std_j = torch.gather(std, 1, idx_j.unsqueeze(1)).squeeze(1)
            dist_cont = torch.distributions.Normal(mean_j, std_j)
            logp_cont_j = dist_cont.log_prob(delta_b[:, j])
            ent_cont_j = dist_cont.entropy()

            logp = logp + torch.where(valid_j, logp_disc_j + logp_cont_j, torch.zeros_like(logp_disc_j))
            entropy = entropy + torch.where(valid_j, ent_disc_j + ent_cont_j, torch.zeros_like(ent_disc_j))
            remove = torch.zeros_like(remaining_mask)
            remove.scatter_(1, idx_j.unsqueeze(1), True)
            remaining_mask = remaining_mask & ~(remove & valid_j.unsqueeze(1))

        return logp, entropy, value
