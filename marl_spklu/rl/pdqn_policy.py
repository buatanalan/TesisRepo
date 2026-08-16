"""Jaringan PDQN diskrit -- mengikuti Lin, Li, Cao, Labiod & Ahmad, "PDQN: User
Preference-Based Charging Station Recommendation", IEEE Trans. Consumer Electronics
70(3):5238-5250, Aug 2024 (paper asli, PDF ada di root repo).

Tiga komponen inti paper yang direplikasi di sini (§IV.A paper):

  1. PREFERENCE EXTRACTION phi:(a_hat, a) -> p  [kontribusi utama paper]
     LSTM atas barisan PASANGAN AKSI: a_hat = SPKLU yang DIREKOMENDASIKAN sistem,
     a = SPKLU yang AKHIRNYA DIPILIH pengguna. Paper: "the relationship between these
     two actions indicates the preference of a user" -- kalau a_hat selalu == a berarti
     pengguna tak punya preferensi kuat (selalu menurut); kalau sering menyimpang ke
     stasiun tertentu, itulah profil preferensinya. Tiap langkah riwayat dikodekan
     one-hot GANDA [onehot(a_hat) ++ onehot(a)] -> dim 2N, sehingga IDENTITAS stasiun
     benar-benar masuk ke encoder (bukan cuma skalar patuh/tidak).

  2. ATTENTION MODULE (§IV.A): menghubungkan fitur preferensi dgn fitur state. Paper
     eksplisit MENOLAK konkatenasi polos ("a simple way ... is to concatenate both the
     state vector and preference vector, but this cannot establish well the relationship").
     Di sini: c_t (preferensi) jadi QUERY, fitur per-stasiun jadi KEY/VALUE -> bobot
     attention atas stasiun -> ringkasan state yang RELEVAN thd preferensi pengguna ini.

  3. FC + RESIDUAL (§IV.A): "we set six layers for this network and each layer has 128
     neurons. Moreover, a residual network is defined ... Each residual block contains
     two fully connected layers." -> 3 blok residual x 2 lapis = 6 lapis @128.

Keluaran: Q(s,p;a) untuk SEMUA aksi sekaligus (kepala FC datar -> |J| nilai), sesuai
paper (bukan kepala per-stasiun berbagi-bobot ala Deep Sets -- kelas fungsi datar
memungkinkan jaringan menghafal bias khas tiap stasiun, mis. "SPKLU 3 kronis padat").

REDESAIN v2 (2026-08-13, N_REC=2 stasiun/keputusan): sempat dicoba Q GABUNGAN atas
C(N,2)=15 kombinasi pasangan (v1 redesain) -- terbukti EMPIRIS memburuk (Gini naik
0,12->0,22 seiring training, q_mean nyaris nol/negatif, lihat memori sesi) krn sinyal
belajar terbagi 15 aksi dlm anggaran yg sama, jauh lebih sparse drpd 6 aksi tunggal.
**Dibalik ke Q INDEPENDEN per-stasiun (6, spt semula)**, tapi `act()` mengambil TOP-2
dari 6 nilai itu (bukan argmax tunggal) -- meniru prinsip H-PPO (bobot dibagi
antar-stasiun = generalisasi jauh lebih hemat data drpd kepala datar/gabungan). Satu
KEPUTUSAN kini menghasilkan DUA transisi training terpisah (lihat pdqn_agent.py) yg
sama2 melatih Q-network 6-dim yg SAMA, bukan satu Q gabungan.
"""
import torch
import torch.nn as nn

NEG_INF = -1e9

N_REC = 2   # jumlah stasiun direkomendasikan/keputusan (tetap, utk riwayat LSTM &
            # jumlah transisi training per keputusan -- BUKAN lagi dim keluaran Q).

# Blok fitur per-stasiun pada obs: onehot_prev, dist, wait_hat, queue, conn, util(D_hat),
# rec_activity (BARU -- opsi #1 "bagaimana agar PDQN menang": sinyal overshoot yg Greedy
# tak mungkin punya, lihat catatan pdqn_agent.py::_build_obs).
STATION_FEAT_DIM_PDQN = 7

PDQN_HIST_K = 10    # T: panjang riwayat KEPUTUSAN (spesifikasi §3.4) -- REDESAIN: tiap
                    # keputusan kini menyumbang N_REC=2 sub-langkah ke LSTM (satu per
                    # stasiun dlm pasangan yg direkomendasikan, dibandingkan thd pilihan
                    # SAMA), jadi panjang urutan LSTM sebenarnya = PDQN_HIST_K * N_REC.
D_LSTM = 64         # d_lstm (spesifikasi §3.4)
D_ATTN = 64         # d_attn (spesifikasi §3.4)
FC_WIDTH = 128      # lebar tiap lapis FC (paper §IV.A)
N_RESIDUAL_BLOCKS = 3   # 3 blok x 2 lapis = 6 lapis (paper §IV.A)


def hist_feat_dim(n_spklu: int) -> int:
    """Dim fitur PER SUB-LANGKAH riwayat: one-hot(a_hat_i) ++ one-hot(a) -- SATU dari
    N_REC pasangan per keputusan (lihat PDQN_HIST_K di atas)."""
    return 2 * n_spklu


# Dim vektor fitur per-stasiun dipakai pasangan preferensi mode FITUR (bukan one-hot
# identitas): [jarak, est_wait, antrean, konektor, utilisasi] -- lihat PREF_STATION_FEAT_DIM
# di rollout.py::RLRolloutAgent._pref_station_feat utk definisi lengkap tiap elemen.
PREF_STATION_FEAT_DIM = 5


def hist_feat_dim_feature(station_feat_dim: int = PREF_STATION_FEAT_DIM) -> int:
    """Dim fitur per-langkah riwayat MODE FITUR: feat(a_hat) ++ feat(a) (bukan one-hot
    identitas) -- pasangan vektor deskriptif stasiun (jarak/wait/antrean/konektor/util),
    supaya LSTM belajar preferensi dari KARAKTERISTIK stasiun, bukan cuma memorisasi
    indeks. Berguna terutama saat N besar/heterogen (one-hot jadi sparse & sulit
    digeneralisasi lintas stasiun serupa)."""
    return 2 * int(station_feat_dim)


class PreferenceAttention(nn.Module):
    """Attention (paper §IV.A): fitur preferensi c_t sbg QUERY, fitur per-stasiun sbg
    KEY/VALUE. Menghasilkan ringkasan state yang dibobot menurut relevansinya thd
    preferensi pengguna yang sedang dilayani -- bukan konkatenasi polos."""

    def __init__(self, station_feat_dim: int, pref_dim: int, d_attn: int):
        super().__init__()
        self.kv_proj = nn.Linear(station_feat_dim, d_attn)
        self.q_proj = nn.Linear(pref_dim, d_attn)
        self.d_attn = int(d_attn)

    def forward(self, station_feats, pref):
        """station_feats: (B,N,F). pref: (B,pref_dim). Return (attended:(B,d_attn), w:(B,N))."""
        kv = self.kv_proj(station_feats)                                  # (B,N,d)
        q = self.q_proj(pref)                                             # (B,d)
        scores = torch.einsum("bnd,bd->bn", kv, q) / (self.d_attn ** 0.5)  # (B,N)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bn,bnd->bd", weights, kv)                # (B,d)
        return attended, weights


class ResidualBlock(nn.Module):
    """Blok residual paper: dua lapis FC + koneksi lompat (cegah vanishing gradient)."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)


class PDQNQNetwork(nn.Module):
    """`n_types`: jumlah LSTM preference-extractor.

      n_types=1 (default) -> SATU LSTM dibagi semua pengguna.
      n_types=5           -> LIMA LSTM, satu per TIPE preferensi, di-switch menurut tipe
                             pengguna yang sedang dilayani. Ini desain paper (§IV.A):
                             "It is possible to design one LSTM for all types of users.
                             However, it cannot achieve good performance according to
                             experience. This is why, we assign one LSTM for one user type."

    Catatan keabsahan: memakai tipe sbg pemilih LSTM berarti sistem DIANGGAP MENGETAHUI
    tipe pengguna. Paper memang mengasumsikan itu (tipe `tp` disimpan di replay buffer,
    Algorithm 1 baris 9). Di deployment nyata tipe adalah variabel laten yang tak teramati,
    jadi asumsi ini harus dinyatakan eksplisit saat membandingkan dgn artefak yang
    menginfer preferensi tanpa label tipe."""

    def __init__(self, obs_dim: int, n_spklu: int, d_lstm: int = D_LSTM,
                 d_attn: int = D_ATTN, width: int = FC_WIDTH,
                 n_blocks: int = N_RESIDUAL_BLOCKS, n_types: int = 1,
                 use_preference: bool = True, pref_feature_mode: bool = False):
        super().__init__()
        # pref_feature_mode=True: pasangan (a_hat,a) dikodekan sbg PASANGAN VEKTOR FITUR
        # stasiun (jarak/wait/antrean/konektor/util), BUKAN one-hot identitas paper asli --
        # lihat pdqn_policy.py::hist_feat_dim_feature & rollout.py::_pref_station_feat.
        self.pref_feature_mode = bool(pref_feature_mode)
        self.n_spklu = int(n_spklu)
        self.n_types = int(n_types)
        # use_preference=False -> ABLASI "DQN tanpa preferensi" (baseline ketiga paper §V:
        # "a deep learning based mechanism ... that the system only considers the state
        # information but no user preference information"). Vektor preferensi c_t dipaksa
        # NOL sehingga modul LSTM+attention tak menyumbang informasi apa pun, TAPI ukuran
        # & kapasitas jaringan tetap sama -- jadi selisih kinerja bisa diatribusikan ke
        # INFORMASI preferensi, bukan ke perbedaan jumlah parameter.
        self.use_preference = bool(use_preference)
        self.scalar_dim = obs_dim - STATION_FEAT_DIM_PDQN * n_spklu
        assert self.scalar_dim >= 0, (
            f"obs_dim={obs_dim} tak konsisten dgn STATION_FEAT_DIM_PDQN="
            f"{STATION_FEAT_DIM_PDQN} x n_spklu={n_spklu}")

        # (1) Preference extraction: LSTM atas pasangan (a_hat, a) -- one-hot identitas
        #     (default, paper asli) atau pasangan vektor fitur (pref_feature_mode=True).
        #     Satu LSTM per tipe pengguna bila n_types > 1.
        self.hist_feat_dim = (hist_feat_dim_feature() if self.pref_feature_mode
                              else hist_feat_dim(n_spklu))
        self.hist_lstms = nn.ModuleList([
            nn.LSTM(self.hist_feat_dim, d_lstm, batch_first=True)
            for _ in range(self.n_types)])

        # (2) Attention preferensi <-> state.
        self.attn = PreferenceAttention(STATION_FEAT_DIM_PDQN, d_lstm, d_attn)

        # (3) FC datar 6 lapis @128 + residual -> Q INDEPENDEN per stasiun (v2 -- lihat
        #     REDESAIN di atas; act() mengambil top-N_REC dari sini, bukan argmax kombinasi).
        in_dim = obs_dim + d_lstm + d_attn
        self.input_proj = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU())
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(n_blocks)])
        self.q_head = nn.Linear(width, n_spklu)

    def _encode_pref(self, hist, types=None):
        """hist: (B,K,2N), types: (B,) long atau None -> c_t: (B,d_lstm).
        Ini phi:(a_hat,a)->p milik paper. Bila n_types>1, tiap baris batch diproses oleh
        LSTM milik TIPE-nya; batch dikelompokkan per tipe supaya tetap satu panggilan
        per LSTM (bukan per-sampel), lalu hasilnya dikembalikan ke urutan semula."""
        if not self.use_preference:
            return torch.zeros(hist.shape[0], self.hist_lstms[0].hidden_size,
                               device=hist.device, dtype=hist.dtype)
        if self.n_types == 1 or types is None:
            _, (h_n, _) = self.hist_lstms[0](hist)
            return h_n[-1]

        out = torch.zeros(hist.shape[0], self.hist_lstms[0].hidden_size,
                          device=hist.device, dtype=hist.dtype)
        for t in range(self.n_types):
            idx = torch.nonzero(types == t, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            _, (h_n, _) = self.hist_lstms[t](hist[idx])
            out[idx] = h_n[-1]
        return out

    def _split_station_block(self, obs):
        """obs: (B, scalar_dim + F*N) -> (scalars, station_feats:(B,N,F)). Blok stasiun
        disimpan sbg F array terpisah sepanjang N yg dikonkatenasi (lihat _build_obs)."""
        scalars = obs[:, :self.scalar_dim]
        block = obs[:, self.scalar_dim:]
        n = block.shape[1] // STATION_FEAT_DIM_PDQN
        station_feats = block.view(-1, STATION_FEAT_DIM_PDQN, n).transpose(1, 2)
        return scalars, station_feats

    def forward(self, obs, hist, types=None):
        """obs: (B,obs_dim), hist: (B,K*N_REC,2N), types: (B,) long|None
        -> Q: (B,n_spklu) -- v2: SATU nilai INDEPENDEN per stasiun (bukan per kombinasi)."""
        c_t = self._encode_pref(hist, types)
        _, station_feats = self._split_station_block(obs)
        attended, _ = self.attn(station_feats, c_t)
        x = torch.cat([obs, c_t, attended], dim=-1)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.q_head(h)

    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, epsilon: float = 0.0, pref_type=None):
        """v2: Q INDEPENDEN per stasiun (spt semula) -- ambil TOP-N_REC dari situ (bukan
        argmax kombinasi gabungan), meniru prinsip H-PPO (bobot dibagi antar-stasiun =
        generalisasi lbh hemat data). `pref_type`: tipe pengguna (memilih LSTM bila
        n_types>1); None -> LSTM ke-0.

        Return {"combo": (i,j)} -- DUA indeks stasiun (urutan menurun nilai-Q, i=terbaik).
        Bila feasible < N_REC (mis. kompatibilitas konektor terbatas), stasiun kedua
        diisi ulang dari stasiun terbaik yg sama (duplikasi, lebih baik drpd rekomendasi
        tak-feasible) -- kasus degenerate, jarang terjadi krn N=6 stasiun.

        epsilon-greedy: dgn peluang epsilon, GANTI SELURUH pasangan dgn 2 stasiun feasible
        acak (bukan cuma satu slot) -- konsisten dgn semantik "eksplorasi keputusan",
        bukan eksplorasi per-slot."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool)
        types = (torch.tensor([int(pref_type)], dtype=torch.long)
                 if (pref_type is not None and self.n_types > 1) else None)

        q = self.forward(obs, hist, types).squeeze(0)   # (n_spklu,)
        feasible_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if feasible_idx.numel() == 0:
            # Degenerate total: tak ada stasiun feasible -- abaikan mask, argmax mentah.
            top2 = torch.topk(q, k=min(N_REC, q.numel())).indices
        elif torch.rand(1).item() < epsilon:
            perm = feasible_idx[torch.randperm(feasible_idx.numel())]
            top2 = perm[:N_REC]
        else:
            q_masked = q.masked_fill(~mask, NEG_INF)
            k_eff = min(N_REC, feasible_idx.numel())
            top2 = torch.topk(q_masked, k=k_eff).indices

        combo = [int(x) for x in top2.tolist()]
        while len(combo) < N_REC:      # duplikasi bila feasible < N_REC (jarang, N=6)
            combo.append(combo[-1] if combo else 0)
        return {"combo": tuple(combo), "q_values": [float(q[c].item()) for c in combo]}
