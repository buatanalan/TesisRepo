"""PDQN ADVANCED -- varian PDQNQNetwork (pdqn_policy.py) yang menerapkan 2 solusi dari
kajian literatur (`Dokumen_Penting/Kajian_Literatur_Solusi_Domain_Serupa.md` §2.2 & §2.3):

  1. SHARED-WEIGHT PER-STATION HEAD (Deep Sets, Zaheer dkk. 2017; identik prinsip
     `StationEncoder` H-PPO/rollout.py): q_head PDQN ASLI adalah `Linear(width, n_spklu)`
     FLAT -- satu matriks bobot memetakan representasi global langsung ke N nilai-Q
     terpisah, TIDAK permutation-equivariant (tiap kolom bobot "menghafal" identitas
     stasiun sbg fitur output, bukan diproses via fungsi BERBAGI-BOBOT yg sama utk tiap
     kandidat). Di sini diganti: MLP BERBAGI BOBOT diterapkan ke tiap [fitur_stasiun_i ++
     konteks_global] (i=1..N) secara independen -> permutation-equivariant, generalize ke
     kandidat mana pun tanpa perlu "melihat" indeksnya scr eksplisit di bobot.

  2. DUELING HEAD (Wang dkk. 2016, "Dueling Network Architectures"): Q(s,i) = V(s) +
     (A(s,i) - mean_j A(s,j)). V(s) dihitung SEKALI dari konteks global (BUKAN per-stasiun)
     -> secara arsitektural bisa menyerap suku reward GLOBAL/broadcast (Gini) tanpa perlu
     dipelajari ulang per-pasangan aksi (i,j), sedangkan A(s,i) (per-stasiun, shared-weight
     spt poin 1) fokus ke komponen INDIVIDUAL (wait/prox). ANALOG konsep baseline/advantage
     PPO yg TAK DIMILIKI PDQN asli (DQN meregresi Q mentah tanpa baseline).

Q_tot(s) = Q(s,i)+Q(s,j) (dekomposisi VDN linier utk k=2) TETAP DIPAKAI -- perbaikan ini
TIDAK mengubah dekomposisi linier itu sendiri (itu perbaikan §2.1/QMIX yg lebih mahal,
BELUM diimplementasikan di sini), HANYA memperbaiki bagaimana Q(s,i) individual dihitung.

Preference extraction (LSTM+attention, phi:(a_hat,a)->p) TAK DIUBAH -- bagian ini SUDAH
terbukti bekerja (acceptance PDQN konsisten tinggi di semua eksperimen sebelumnya),
kajian literatur tak menemukan masalah pada bagian ini.

Interface (constructor, forward, act) SENGAJA identik `PDQNQNetwork` -> drop-in
replacement di `dqn_trainer.py` lewat parameter `network_cls`, TANPA mengubah
`pdqn_agent.py`/`PDQNRolloutAgent` sama sekali (agen hanya memanggil `q_net.act(...)`,
tak peduli arsitektur internal).

EVALUASI BERSIH (2026-08-16): `PDQNAdvancedQNetwork` (shared-weight+dueling SAJA, TANPA
mengubah dekomposisi VDN linier `Q_tot=Q(i)+Q(j)`) TERBUKTI TAK MENJAWAB masalah -- Gini
eval bersih (epsilon=0, 30 hari penuh) 0,1014, MASIH kalah dari greedy (0,088-0,092) DAN
sedikit LEBIH BURUK dari PDQN original (0,0991). Kesimpulan: akar masalah bukan CARA
Q(s,i) individual dihitung, melainkan CARA Q(s,i) & Q(s,j) DIGABUNG jadi Q_tot -- dekomposisi
linier (VDN) itu sendiri (§2.1 `Kajian_Literatur_Solusi_Domain_Serupa.md`). Perbaikan
LANJUTAN di bawah (`PDQNQMixQNetwork`) mengganti penjumlahan linier dgn MIXING NETWORK
non-linier MONOTONIK ala QMIX (Rashid dkk. 2018) -- mempertahankan properti IGM
(Individual-Global-Max, shg `act()`/pemilihan top-2 individual TETAP VALID tanpa
perubahan) tapi kapasitas representasi Q_tot jauh lebih luas dari sekadar penjumlahan."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_spklu.rl.pdqn_policy import (
    NEG_INF, N_REC, STATION_FEAT_DIM_PDQN, D_LSTM, D_ATTN, FC_WIDTH,
    N_RESIDUAL_BLOCKS, hist_feat_dim, hist_feat_dim_feature, PreferenceAttention,
    ResidualBlock,
)


class SharedStationDuelingHead(nn.Module):
    """MLP berbagi-bobot diterapkan ke tiap stasiun (Deep Sets), dipecah jadi 2 kepala
    (dueling): V(s) dari konteks global SAJA (bukan per-stasiun), A(s,i) per-stasiun."""

    def __init__(self, station_feat_dim: int, global_dim: int, hidden: int = 64):
        super().__init__()
        # Kepala per-stasiun: input = [fitur_stasiun_i ++ konteks_global] -- BERBAGI
        # BOBOT lintas i (diterapkan via broadcasting, bukan Linear(width, N) terpisah).
        self.station_mlp = nn.Sequential(
            nn.Linear(station_feat_dim + global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.advantage_head = nn.Linear(hidden, 1)   # shared -> A(s,i), diterapkan per-i
        self.value_head = nn.Linear(global_dim, 1)   # HANYA dari konteks global -> V(s)

    def forward(self, station_feats, global_ctx):
        """station_feats: (B,N,F). global_ctx: (B,G). -> Q: (B,N)."""
        B, N, F = station_feats.shape
        ctx_bcast = global_ctx.unsqueeze(1).expand(B, N, global_ctx.shape[-1])   # (B,N,G)
        x = torch.cat([station_feats, ctx_bcast], dim=-1)                        # (B,N,F+G)
        h = self.station_mlp(x)                                                  # (B,N,hidden)
        adv = self.advantage_head(h).squeeze(-1)                                 # (B,N)
        val = self.value_head(global_ctx)                                        # (B,1)
        # Dueling combine: pusatkan advantage (rata2 atas SEMUA N stasiun -- masking
        # kelayakan tetap ditangani hilir di act()/training loop, sama spt PDQNQNetwork
        # asli yg juga menghitung Q utk semua stasiun lalu mask belakangan).
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q


class PDQNAdvancedQNetwork(nn.Module):
    """Drop-in replacement `PDQNQNetwork` (pdqn_policy.py) -- preference extraction
    (LSTM+attention) IDENTIK, q_head diganti `SharedStationDuelingHead` (shared-weight +
    dueling, lihat docstring modul). Constructor/forward/act SENGAJA sama signature."""

    def __init__(self, obs_dim: int, n_spklu: int, d_lstm: int = D_LSTM,
                 d_attn: int = D_ATTN, width: int = FC_WIDTH,
                 n_blocks: int = N_RESIDUAL_BLOCKS, n_types: int = 1,
                 use_preference: bool = True, pref_feature_mode: bool = False,
                 duel_hidden: int = 64):
        super().__init__()
        self.pref_feature_mode = bool(pref_feature_mode)
        self.n_spklu = int(n_spklu)
        self.n_types = int(n_types)
        self.use_preference = bool(use_preference)
        self.scalar_dim = obs_dim - STATION_FEAT_DIM_PDQN * n_spklu
        assert self.scalar_dim >= 0, (
            f"obs_dim={obs_dim} tak konsisten dgn STATION_FEAT_DIM_PDQN="
            f"{STATION_FEAT_DIM_PDQN} x n_spklu={n_spklu}")

        # (1) Preference extraction -- IDENTIK PDQNQNetwork.
        self.hist_feat_dim = (hist_feat_dim_feature() if self.pref_feature_mode
                              else hist_feat_dim(n_spklu))
        self.hist_lstms = nn.ModuleList([
            nn.LSTM(self.hist_feat_dim, d_lstm, batch_first=True)
            for _ in range(self.n_types)])

        # (2) Attention preferensi <-> state -- IDENTIK.
        self.attn = PreferenceAttention(STATION_FEAT_DIM_PDQN, d_lstm, d_attn)

        # (3) Backbone global (FC+residual) -- IDENTIK secara STRUKTUR (menghasilkan
        # konteks global `h`), TAPI TIDAK LAGI langsung diproyeksikan flat ke N -- `h`
        # dipakai sbg `global_ctx` masukan SharedStationDuelingHead di bawah.
        in_dim = obs_dim + d_lstm + d_attn
        self.input_proj = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU())
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(n_blocks)])

        # (4) BARU: kepala shared-weight + dueling (ganti `q_head` flat).
        self.q_head_advanced = SharedStationDuelingHead(
            STATION_FEAT_DIM_PDQN, width, hidden=duel_hidden)

    def _encode_pref(self, hist, types=None):
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
        scalars = obs[:, :self.scalar_dim]
        block = obs[:, self.scalar_dim:]
        n = block.shape[1] // STATION_FEAT_DIM_PDQN
        station_feats = block.view(-1, STATION_FEAT_DIM_PDQN, n).transpose(1, 2)
        return scalars, station_feats

    def forward_with_ctx(self, obs, hist, types=None):
        """-> (Q:(B,n_spklu), h:(B,width)) -- `h` = konteks global (keluaran backbone
        residual, SEBELUM kepala dueling per-stasiun), diekspos utk dipakai QMixer
        (`PDQNQMixQNetwork` di bawah) sbg state conditioning mixing network."""
        c_t = self._encode_pref(hist, types)
        _, station_feats = self._split_station_block(obs)
        attended, _ = self.attn(station_feats, c_t)
        x = torch.cat([obs, c_t, attended], dim=-1)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        q = self.q_head_advanced(station_feats, h)
        return q, h

    def forward(self, obs, hist, types=None):
        """-> Q: (B,n_spklu), lewat SharedStationDuelingHead (bukan q_head flat)."""
        q, _ = self.forward_with_ctx(obs, hist, types)
        return q

    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, hist_np, epsilon: float = 0.0, pref_type=None):
        """IDENTIK `PDQNQNetwork.act` -- top-N_REC dari Q, epsilon-greedy pasangan penuh."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        hist = torch.as_tensor(hist_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool)
        types = (torch.tensor([int(pref_type)], dtype=torch.long)
                 if (pref_type is not None and self.n_types > 1) else None)

        q = self.forward(obs, hist, types).squeeze(0)
        feasible_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if feasible_idx.numel() == 0:
            top2 = torch.topk(q, k=min(N_REC, q.numel())).indices
        elif torch.rand(1).item() < epsilon:
            perm = feasible_idx[torch.randperm(feasible_idx.numel())]
            top2 = perm[:N_REC]
        else:
            q_masked = q.masked_fill(~mask, NEG_INF)
            k_eff = min(N_REC, feasible_idx.numel())
            top2 = torch.topk(q_masked, k=k_eff).indices

        combo = [int(x) for x in top2.tolist()]
        while len(combo) < N_REC:
            combo.append(combo[-1] if combo else 0)
        return {"combo": tuple(combo), "q_values": [float(q[c].item()) for c in combo]}


class QMixer(nn.Module):
    """Mixing network ala QMIX (Rashid dkk. 2018, "QMIX: Monotonic Value Function
    Factorisation") -- ganti Q_tot=Q(i)+Q(j) (VDN, penjumlahan TETAP/bobot=1 utk semua
    state) dgn Q_tot = f_state(Q(i), Q(j)) NON-LINIER, tapi MONOTONIK naik thd tiap
    Q individual (bobot hypernetwork dipaksa non-negatif via `abs`) -- mempertahankan
    properti IGM (Individual-Global-Max): argmax joint (i,j) atas Q_tot SELALU sama dgn
    menggabungkan argmax individual tiap Q, sehingga `act()` (pilih top-2 Q individual
    tertinggi) TETAP VALID tanpa perlu enumerasi C(N,2) pasangan secara eksplisit.

    Hypernetwork 2 lapis (w1,b1,w2,b2) dikondisikan pada `state` (konteks global backbone,
    BUKAN observasi mentah) -- persis pola QMIX asli, disederhanakan utk n_agents=2 (SATU
    keputusan k=2, bukan multi-agen literal)."""

    def __init__(self, state_dim: int, n_agents: int = N_REC, embed_dim: int = 32):
        super().__init__()
        self.n_agents = int(n_agents)
        self.embed_dim = int(embed_dim)
        self.hyper_w1 = nn.Linear(state_dim, self.n_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, qs, state):
        """qs: (B,n_agents) nilai-Q individual stasiun terpilih. state: (B,state_dim).
        -> Q_tot: (B,)."""
        B = qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(qs.unsqueeze(1), w1) + b1)          # (B,1,embed_dim)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2                            # (B,1,1)
        return q_tot.view(B)


class PDQNQMixQNetwork(PDQNAdvancedQNetwork):
    """`PDQNAdvancedQNetwork` (shared-weight+dueling per-stasiun) + `QMixer` (§2.1 kajian
    literatur) menggabungkan Q(s,i)&Q(s,j) jadi Q_tot -- perbaikan LANJUTAN setelah
    `PDQNAdvancedQNetwork` sendirian terbukti TAK CUKUP (lihat catatan di docstring modul
    atas). `self.mixer` jadi submodule -> ikut ter-`load_state_dict` bersama q_net/
    target_net di `dqn_trainer.py` (tak perlu mixer/target-mixer terpisah).

    `act()`/`forward()` TAK BERUBAH (masih pilih top-2 Q individual tertinggi, valid via
    IGM) -- HANYA `dqn_trainer.py::_dqn_update` yg perlu tahu memakai `.mixer` utk
    menghitung target TD (dideteksi via `hasattr(self.q_net, 'mixer')`, backward-compatible
    dgn `PDQNQNetwork`/`PDQNAdvancedQNetwork` polos yg tak punya atribut itu)."""

    def __init__(self, obs_dim: int, n_spklu: int, d_lstm: int = D_LSTM,
                 d_attn: int = D_ATTN, width: int = FC_WIDTH,
                 n_blocks: int = N_RESIDUAL_BLOCKS, n_types: int = 1,
                 use_preference: bool = True, pref_feature_mode: bool = False,
                 duel_hidden: int = 64, mixer_embed_dim: int = 32):
        super().__init__(obs_dim, n_spklu, d_lstm=d_lstm, d_attn=d_attn, width=width,
                         n_blocks=n_blocks, n_types=n_types, use_preference=use_preference,
                         pref_feature_mode=pref_feature_mode, duel_hidden=duel_hidden)
        self.mixer = QMixer(state_dim=width, n_agents=N_REC, embed_dim=mixer_embed_dim)
