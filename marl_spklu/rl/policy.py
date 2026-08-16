"""Jaringan PPO EV Agent — ruang aksi v2 (Spesifikasi_Teknis_RL.md, disederhanakan):
  * Aksi = HANYA pemilihan SPKLU yang direkomendasikan. TIDAK ADA lagi komponen kontinu
    (dulu "a2", modifikasi EstWait yang ditampilkan) -- EstWait sepenuhnya dihitung modul
    terpisah (forecaster), selalu jujur; SPKLU di luar rekomendasi diberi EstWait = inf
    (lihat rollout.py::get_recommendation).
  * Jaringan menghasilkan SATU skor per SPKLU feasible -> softmax -> probabilitas p_i.
    Pemilihan DETERMINISTIK dari p_i (bukan sampling stokastik spt desain lama): aturan
    gabungan -- SPKLU dgn p_i > 20% masuk rekomendasi; kalau himpunan kosong, ambil SPKLU
    probabilitas tertinggi TUNGGAL (lantai minimum 1); kalau > k yang lolos, ambil k
    teratas (langit-langit k). Eksplorasi selama training via epsilon-greedy (gaya PDQN
    diskrit, `dqn_trainer.py`), BUKAN dari stokastisitas pemilihan itu sendiri.

Satu jaringan di-share ke semua EV (parameter sharing, CTDE). Diferensiasi perilaku antar-agen
berasal murni dari perbedaan riwayat/observasi yang menjadi masukan encoder yang SERAGAM --
bukan dari parameter khusus per-pengguna (tanpa embedding identitas / "agent indication").
Critic = baseline PPO dengan encoder terpisah menerima keadaan global (CTDE).

Encoder rekuren (§POMDP, IV.1.7): trust T_i pengguna tidak teramati agen. Sebagai ganti
akses langsung, LSTM `hist_lstm` memproses riwayat interaksi K-langkah terakhir
(complied, disp_estwait_norm, wait_default_norm) menjadi vektor konteks c_t yang secara
implisit merepresentasikan trust tiruan T̂_i. c_t digabung ke fitur observasi lain sebelum
masuk ke encoder aktor/kritik -- bukan skalar acceptance-rate mentah.

`rec_activity` (Spesifikasi_Teknis_RL.md §2.3, WAJIB): blok observasi BARU di aktor --
`sim.recent_recs` (jendela 24 jam berjalan, sudah dipakai reward anti-herding & sebelumnya
HANYA ada di observasi kritik). Krn pemilihan kini deterministik (bukan lagi stokastik),
entropi tinggi pada distribusi SAJA tidak lagi menjamin variasi keputusan antar-pengguna
serupa -- aktor perlu MELIHAT aktivitas rekomendasi terkini utk bisa menurunkan skor SPKLU
yang baru saja "ramai" direkomendasikan secara proaktif, bukan cuma bereaksi lewat gradien
reward setelah herding terjadi.

Encoder PER-SPKLU (bukan flatten): observasi mentah dari rollout.py adalah konkatenasi
FLAT [scalars, onehot(N), dist(N), wait(N), queue(N), conn(N), rec_activity(N)] -- representasi
ini secara desain TIDAK cocok diumpankan langsung ke satu Linear monolitik: posisi j pada
tiap blok memang selalu merujuk stasiun yang sama, tapi Linear flat memaksa jaringan belajar
N fungsi skor terpisah (satu per kolom output) dari satu vektor gabungan, tanpa BERBAGI bobot
antar-stasiun -- boros parameter, tidak invarian-permutasi, dan tak generalisasi ke
stasiun yang jarang dilihat. `StationEncoder` di bawah membalik `obs`/`critic_obs`
kembali menjadi (N, fitur-per-stasiun), menjalankan MLP KECIL yang SAMA (dibagi/shared)
pada tiap baris (stasiun), lalu kepala skor (`disc_head`) juga dibagi antar-stasiun --
setara arsitektur Deep Sets / attention-free permutation-equivariant network atas himpunan
SPKLU. Kritik memakai attention pooling (bobot antar-stasiun DIPELAJARI, bukan rata-rata
polos) supaya tetap invarian-permutasi & N-agnostik.

CTDE PENUH (kritik = obs aktor + critic_obs global + c_t): kritik TIDAK hanya melihat
critic_obs (utilisasi/antrean/gini SELURUH SPKLU + agregat trust populasi), tapi juga
digabung dgn `obs` milik agen yang sedang dievaluasi (lokal) -- textbook CTDE: aktor
desentralisasi (hanya obs+c_t), kritik tersentralisasi (obs+critic_obs+c_t, akses
penuh state global TERMASUK agregat trust asli T_i -- variabel privileged yang
sengaja TIDAK bocor ke aktor, hanya dipakai kritik saat training, konsisten CTDE).
"""
import warnings
import torch
import torch.nn as nn

NEG_INF = -1e9
HIST_FEAT_DIM = 4     # (complied, disp_estwait_norm, wait_default_norm, realized_gap_norm)
HIST_HIDDEN = 16       # dim vektor konteks c_t

# Jumlah blok fitur per-stasiun di obs/critic_obs (lihat rollout.py _build_obs /
# _build_critic_obs) -- urutan konkatenasi TETAP:
#   obs        = [scalars(soc,x,y,sin_hour,cos_hour,battery_capacity_kwh),
#                 onehot_prev, dist, wait_hat, queue, conn, rec_activity]
#                (6 blok stasiun: onehot,dist,wait_hat,queue,conn,rec_activity; 6 skalar)
#   critic_obs = [utilisasi, antrean, rec_activity, conn, oracle_wait,
#                 gini, trust_mean, trust_std, trust_min, delta_gini,
#                 user_trust, user_freq_i, user_w4, sin_hour, cos_hour]
#                (5 blok stasiun + 10 skalar) -- oracle_wait = compute_virtual_wait
#                (ground-truth simulator), user_trust/freq_i/w4 = pengguna AKTIF
#                yg sedang dievaluasi transisinya (privileged, bukan agregat populasi).
# Ubah di sini JIKA urutan blok pada rollout.py berubah.
STATION_FEAT_DIM = 6          # v2: +1 (rec_activity) drpd v1 (5)
CRITIC_STATION_FEAT_DIM = 5   # tak berubah -- rec_activity SUDAH ada di critic_obs sejak awal


class StationEncoder(nn.Module):
    """MLP kecil yang DIBAGI (shared weights) di semua stasiun: setiap baris (stasiun)
    diproses lewat jaringan yang SAMA, digabung dgn konteks global (skalar pengguna +
    c_t) yang di-broadcast ke tiap stasiun. Permutation-equivariant atas urutan
    stasiun -- menukar urutan SPKLU menghasilkan output tertukar identik, bukan
    keluaran berbeda spt Linear flat."""

    def __init__(self, station_feat_dim: int, context_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(station_feat_dim + context_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, station_feats, context):
        """station_feats: (B, N, F). context: (B, C) -> di-broadcast ke tiap stasiun.
        Return: (B, N, hidden)."""
        n = station_feats.shape[1]
        context_exp = context.unsqueeze(1).expand(-1, n, -1)
        return self.net(torch.cat([station_feats, context_exp], dim=-1))


class AttentionPooling(nn.Module):
    """Ringkas (B,N,H) -> (B,H) via RATA-RATA BERBOBOT, bobotnya DIPELAJARI (bukan 1/N
    rata seperti mean pooling) -- kritik bisa belajar "memberi perhatian lebih" ke
    stasiun tertentu (mis. yang sedang timpang/padat), TETAP invarian-ukuran (bobot
    disusun via softmax -> selalu berjumlah 1 berapa pun N, beda dgn concat)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, emb):
        """emb: (B, N, hidden) -> (B, hidden)."""
        attn_logits = self.score(emb).squeeze(-1)          # (B, N)
        attn_weights = torch.softmax(attn_logits, dim=-1)   # (B, N), jumlah=1
        return torch.einsum("bn,bnh->bh", attn_weights, emb)


class HPPOPolicy(nn.Module):
    """`n_critics`: jumlah kepala nilai (adopsi kritik-ganda MASTER, lihat
    `Rencana_Kritik_Ganda_MASTER.md`). n_critics=1 (default) = perilaku lama persis:
    reward diskalarisasi jadi satu aliran, satu V(s). n_critics=K>1 = satu V^k(s) per
    aliran reward (lihat STREAM_* di rollout.py), advantage dihitung & dinormalisasi
    TERPISAH per aliran lalu digabung dgn bobot beta di PPOTrainer.

    Catatan adaptasi: MASTER memakai DDPG shg kritiknya Q(x,a) dan bobot masuk lewat
    grad_a Q. H-PPO memakai PPO shg kritiknya V(s) tanpa aksi -- padanannya adalah
    menggabung ADVANTAGE per aliran, bukan gradien aksi."""

    def __init__(self, obs_dim: int, critic_obs_dim: int, n_spklu: int, hidden: int = 128,
                 hist_hidden: int = HIST_HIDDEN, station_hidden: int = 64,
                 n_critics: int = 1):
        super().__init__()
        self.n_spklu = n_spklu
        self.hist_hidden = int(hist_hidden)
        self.station_hidden = int(station_hidden)
        self.n_critics = int(n_critics)

        # obs = [scalars(scalar_dim), STATION_FEAT_DIM blok masing2 panjang n_spklu].
        self.scalar_dim = obs_dim - STATION_FEAT_DIM * n_spklu
        assert self.scalar_dim >= 0, (
            f"obs_dim={obs_dim} tak konsisten dgn STATION_FEAT_DIM={STATION_FEAT_DIM} "
            f"x n_spklu={n_spklu} (scalar_dim negatif) -- cek _build_obs di rollout.py")
        self.critic_scalar_dim = critic_obs_dim - CRITIC_STATION_FEAT_DIM * n_spklu
        assert self.critic_scalar_dim >= 0, (
            f"critic_obs_dim={critic_obs_dim} tak konsisten dgn CRITIC_STATION_FEAT_DIM="
            f"{CRITIC_STATION_FEAT_DIM} x n_spklu={n_spklu}")

        # LSTM riwayat interaksi -> c_t (trust tiruan). batch_first: (B, K, HIST_FEAT_DIM).
        self.hist_lstm = nn.LSTM(HIST_FEAT_DIM, self.hist_hidden, batch_first=True)

        # Encoder aktor PER-STASIUN (bobot dibagi antar-stasiun) + SATU kepala skor
        # (disc_head) -- v2 tak lagi punya kepala kontinu (cont_mean_head/cont_logstd_head
        # dihapus bersama penghapusan a2 dari ruang aksi).
        actor_context_dim = self.scalar_dim + self.hist_hidden
        self.station_encoder = StationEncoder(STATION_FEAT_DIM, actor_context_dim, self.station_hidden)
        self.disc_head = nn.Linear(self.station_hidden, 1)          # skor rekomendasi per stasiun

        # Encoder critic PER-STASIUN (CTDE PENUH): kritik menerima GABUNGAN obs (aktor,
        # lokal) + critic_obs (state global) + c_t -- bukan critic_obs saja. obs sudah
        # tersedia sbg argumen forward(), jadi digabung langsung di sana (lihat forward())
        # tanpa perlu jalur data baru. Dim fitur per-stasiun & skalar jadi PENJUMLAHAN
        # blok aktor+kritik (lihat forward()). Attention pooling -> invarian-permutasi,
        # bobot agregasi antar-stasiun DIPELAJARI (bukan rata-rata polos).
        merged_station_feat_dim = STATION_FEAT_DIM + CRITIC_STATION_FEAT_DIM
        merged_scalar_dim = self.scalar_dim + self.critic_scalar_dim
        critic_context_dim = merged_scalar_dim + self.hist_hidden
        self.critic_station_encoder = StationEncoder(merged_station_feat_dim, critic_context_dim, hidden)
        self.critic_pool = AttentionPooling(hidden)
        # Keluaran n_critics (bukan 1): satu nilai per aliran reward. Utk n_critics=1
        # bentuknya identik dgn sebelumnya -> bobot tersimpan lama tetap bisa dimuat.
        self.critic_head = nn.Sequential(
            nn.Linear(hidden + merged_scalar_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_critics),
        )

    def _encode_hist(self, hist):
        """hist: (B, K, HIST_FEAT_DIM) -> c_t: (B, hist_hidden)."""
        _, (h_n, _) = self.hist_lstm(hist)
        return h_n[-1]

    def _split_station_block(self, flat, feat_dim, scalar_dim):
        """flat: (B, scalar_dim + feat_dim*N) -> (scalars:(B,scalar_dim), stasiun:(B,N,feat_dim)).
        Blok stasiun disimpan sbg `feat_dim` array TERPISAH sepanjang N yg dikonkatenasi
        (bukan diinterleave per-stasiun) -- lihat urutan di rollout.py -- makanya reshape
        (feat_dim, N) dulu baru transpose ke (N, feat_dim)."""
        scalars = flat[:, :scalar_dim]
        block = flat[:, scalar_dim:]
        n = block.shape[1] // feat_dim
        station_feats = block.view(-1, feat_dim, n).transpose(1, 2)
        return scalars, station_feats

    def forward(self, obs, hist, critic_obs=None):
        c_t = self._encode_hist(hist)
        scalars, station_feats = self._split_station_block(obs, STATION_FEAT_DIM, self.scalar_dim)
        context = torch.cat([scalars, c_t], dim=-1)
        emb = self.station_encoder(station_feats, context)   # (B, N, station_hidden)

        logits = self.disc_head(emb).squeeze(-1)              # (B, N)

        if critic_obs is not None:
            c_scalars, c_station_feats = self._split_station_block(
                critic_obs, CRITIC_STATION_FEAT_DIM, self.critic_scalar_dim)
            # CTDE PENUH (Prioritas 1): kritik melihat obs aktor (lokal) SEKALIGUS
            # critic_obs (global) -- bukan critic_obs saja. Digabung per-stasiun
            # (onehot,dist,wait,queue,conn,rec_activity,utilisasi,antrean,...) dan skalar.
            merged_station_feats = torch.cat([station_feats, c_station_feats], dim=-1)
            merged_scalars = torch.cat([scalars, c_scalars], dim=-1)
            c_context = torch.cat([merged_scalars, c_t], dim=-1)
            c_emb = self.critic_station_encoder(merged_station_feats, c_context)   # (B, N, hidden)
            pooled = self.critic_pool(c_emb)                    # invarian-permutasi, bobot dipelajari
            # (B, n_critics) -- TIDAK di-squeeze supaya bentuknya seragam utk K berapa pun.
            value = self.critic_head(torch.cat([pooled, merged_scalars], dim=-1))
        else:
            value = torch.zeros(obs.shape[0], self.n_critics, device=obs.device)
        return logits, value

    # ---------------- Seleksi (rollout) — deterministik + epsilon-greedy ----------------
    # v2: TIDAK LAGI sampling stokastik per-slot. Aturan GABUNGAN (Spesifikasi_Teknis_RL.md
    # §2.2): SPKLU dgn probabilitas > `threshold` masuk rekomendasi; himpunan kosong ->
    # ambil probabilitas TERTINGGI TUNGGAL (lantai 1); > k yang lolos -> ambil k teratas
    # (langit-langit k). Eksplorasi via epsilon-greedy (§2.3 KEPUTUSAN ANDA #3): dgn
    # peluang `epsilon`, GANTI seluruh aturan di atas dgn subset ACAK (ukuran seragam
    # 1..k di antara yg feasible) -- meniru mekanisme dqn_trainer.py, BUKAN dari
    # stokastisitas pemilihan itu sendiri (yg kini deterministik).
    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, k: int = 3, critic_obs_np=None,
           epsilon: float = 0.0, threshold: float = 0.20):
        """obs_np: (obs_dim,), feasible_mask_np: (N,) bool, hist_np: (K,HIST_FEAT_DIM).
        k: batas ATAS jumlah SPKLU direkomendasikan (langit-langit, bukan target tetap).
        Return dict {chosen_indices, n_rec, logp, value} -- n_rec bisa < k (aturan
        threshold) ATAU < k krn feasible < k, TIDAK PERNAH > k."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)

        if critic_obs_np is not None:
            critic_obs = torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
        else:
            critic_obs = None

        logits, value = self.forward(obs, hist, critic_obs)
        if not torch.isfinite(logits).all():
            warnings.warn("HPPOPolicy.act: keluaran non-finite (NaN/Inf) pada logits "
                          "-> kemungkinan gradien meledak; turunkan lr atau cek observasi.",
                          RuntimeWarning)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=NEG_INF)

        feasible_idx = torch.nonzero(mask[0], as_tuple=True)[0]
        n_feasible = int(feasible_idx.numel())
        k_eff = max(1 if n_feasible > 0 else 0, min(int(k), n_feasible))

        import random as _random
        if n_feasible > 0 and _random.random() < epsilon:
            # Eksplorasi: subset ACAK (ukuran seragam 1..k_eff) dari yg feasible --
            # log-prob TETAP dihitung di bawah kebijakan saat ini (bukan uniform-random),
            # persis semantik "off-policy exploration, on-policy log-prob utk PPO ratio".
            explore_size = _random.randint(1, k_eff)
            perm = feasible_idx[torch.randperm(n_feasible)][:explore_size]
            chosen_order = perm.tolist()
        else:
            masked_logits = logits[0].masked_fill(~mask[0], NEG_INF)
            probs = torch.softmax(masked_logits, dim=-1)
            above = feasible_idx[probs[feasible_idx] > threshold]
            if above.numel() == 0:
                # Lantai minimum 1: SPKLU probabilitas tertinggi tunggal.
                above = feasible_idx[probs[feasible_idx].argmax().unsqueeze(0)]
            if above.numel() > k_eff:
                # Langit-langit k: k teratas di antara yg lolos threshold.
                top = torch.topk(probs[above], k_eff).indices
                above = above[top]
            # Urutkan menurun by probabilitas (utk log-prob sequential-without-replacement
            # di bawah -- urutan evaluasi harus konsisten act() <-> evaluate()).
            order = torch.argsort(probs[above], descending=True)
            chosen_order = above[order].tolist()

        # log-prob HIMPUNAN terpilih di bawah kebijakan SAAT INI (Categorical berurutan
        # tanpa-pengembalian atas kandidat feasible) -- valid dihitung utk himpunan APA
        # PUN (dipilih deterministik/threshold ATAU acak saat eksplorasi), krn hanya
        # mengevaluasi log p(pilihan ke-j | sisa kandidat) pada urutan yg SAMA dgn yg
        # dipakai evaluate() saat update PPO (lihat catatan §2.4 Spesifikasi_Teknis_RL.md).
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
            # array (K,) -- satu nilai per aliran reward (K=1 -> array 1 elemen).
            "value": value[0].detach().cpu().numpy().astype("float64"),
        }

    # ---------------- Recompute (update PPO) ----------------
    def evaluate(self, obs_b, mask_b, chosen_indices_b, n_rec_b, hist_b, critic_obs_b=None):
        """Batch. obs_b:(B,obs), mask_b:(B,N) bool, chosen_indices_b:(B,k) long (slot
        j>=n_rec_b[i] adalah padding, TIDAK ikut logp/entropy), n_rec_b:(B,) long (jumlah
        slot valid per baris -- VARIABEL, bisa < k krn aturan threshold ATAU krn feasible
        < k; mekanisme masking ini SAMA persis dgn v1, sudah generik utk n_rec variabel),
        hist_b:(B,K,HIST_FEAT_DIM). Replay urutan evaluasi SAMA seperti act() (menurun by
        probabilitas) supaya rasio PPO (exp(logp_baru - logp_lama)) valid.
        Return logp,entropy,value (semua (B,), dijumlah atas slot valid)."""
        logits, value = self.forward(obs_b, hist_b, critic_obs_b)
        logits = logits.masked_fill(~mask_b, NEG_INF)

        B, k = chosen_indices_b.shape
        remaining_mask = mask_b.clone()
        logp = torch.zeros(B, device=logits.device)
        entropy = torch.zeros(B, device=logits.device)
        for j in range(k):
            valid_j = j < n_rec_b   # (B,) bool
            masked_logits = logits.masked_fill(~remaining_mask, NEG_INF)
            dist_disc = torch.distributions.Categorical(logits=masked_logits)
            idx_j = chosen_indices_b[:, j]
            logp_disc_j = dist_disc.log_prob(idx_j)
            ent_disc_j = dist_disc.entropy()

            logp = logp + torch.where(valid_j, logp_disc_j, torch.zeros_like(logp_disc_j))
            entropy = entropy + torch.where(valid_j, ent_disc_j, torch.zeros_like(ent_disc_j))
            # Buang slot terpilih dari kandidat tersisa HANYA utk baris valid (baris
            # invalid/padding tak dipakai lagi di langkah berikutnya, aman dibiarkan).
            remove = torch.zeros_like(remaining_mask)
            remove.scatter_(1, idx_j.unsqueeze(1), True)
            remaining_mask = remaining_mask & ~(remove & valid_j.unsqueeze(1))

        return logp, entropy, value
