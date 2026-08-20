"""MASTER perspektif-EV: aktor DESENTRALISASI per-PERMINTAAN (bukan per-stasiun,
lihat `master_ddpg_policy.py` utk versi asli) + Centralized Attentive Critic yang
menilai SELURUH keputusan pada satu TIMESTEP sekaligus (bukan seluruh stasiun pada
satu keputusan). Unit agen = permintaan pengisian EV, sejalan `[[peran-agen-permintaan-ev]]`.

ARSITEKTUR (persis rancangan yang disepakati -- lihat diagram mermaid terkait)
-------------------------------------------------------------------------------
Aktor  pi(.|o_i) -> rekomendasi LANGSUNG (kategorikal atas stasiun kandidat, BUKAN
       bid/lelang) per permintaan EV_i. SATU jaringan, bobot dibagi lintas permintaan
       DAN lintas kandidat (StationEncoder, sama konvensi `policy.py`).
Kritik V_i = f(token_i | token_j, j pada timestep SAMA) -- kritik terpusat (CTDE)
       yang MENGUMPULKAN semua keputusan pada satu langkah simulasi sekaligus, lalu
       memakai MULTI-HEAD SELF-ATTENTION (blok Transformer, H kepala) antar token-EV
       supaya nilai tiap EV sadar konteks EV lain yang bersaing pada saat sama --
       inilah realisasi literal "kritik ber-atensi yang condition pada aksi bersama"
       (poin yang di audit ditemukan TAK setia di `master_bidding_policy.py`/lengan
       lama: atensi mereka condition pada FITUR, bukan AKSI bersama).

BEDA dgn `master_ddpg_policy.py` (MASTER "asli", stasiun-sbg-agen):
    - Unit agen: permintaan EV (di sini) vs stasiun (asli).
    - Bentuk aksi: rekomendasi kategorikal langsung (di sini) vs bid kontinu (asli).
    - Unit batching kritik: TIMESTEP (semua EV yang berkeputusan sama saat itu, di
      sini) vs KEPUTUSAN TUNGGAL (semua stasiun, pada satu keputusan, asli).
    - Backbone: PPO on-policy (di sini, cocok utk aksi kategorikal) vs DDPG off-policy
      (asli, perlu utk aksi kontinu/bid).
"""
import torch
import torch.nn as nn

from marl_spklu.rl.master_paper_obs import STATION_FEAT_DIM_MASTER, STATION_FEAT_DIM_MASTER_EV

# --- Token kritik: state-EV ringkas (4) + fitur stasiun yang DIREKOMENDASIKAN (7) ---
EV_STATE_DIM = 4        # soc_norm, baterai_norm, urgency_norm, jarak_ke_terpilih_norm
ACTION_FEAT_DIM = STATION_FEAT_DIM_MASTER   # 7 -- fitur §3.1 stasiun terpilih SAJA
TOKEN_DIM = EV_STATE_DIM + ACTION_FEAT_DIM  # 11
PRIV_DIM = 2             # future_avail_terpilih (Delayed Access), gini_now


class MasterEVActor(nn.Module):
    """pi(.|o_i): MLP kecil, bobot dibagi semua kandidat DAN semua permintaan --
    dipanggil batched atas (B,N,10) [obs §3.1 + state pemohon, lihat master_paper_obs.py
    ::build_joint_obs_master_ev]. Kategorikal (BUKAN bid) -- cocok backbone PPO on-policy,
    karena aktor di sini memberi REKOMENDASI LANGSUNG per permintaan, bukan menawar."""

    def __init__(self, station_feat_dim: int = STATION_FEAT_DIM_MASTER_EV, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(station_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        """obs:(B,N,10) mask:(B,N) bool (feasible=True) -> logits:(B,N), stasiun
        tak-feasible diberi -inf SEBELUM softmax (bukan sesudah) -- konsisten dgn
        `AttentiveJointPooling` (master_ddpg_policy.py)."""
        emb = self.encoder(obs)                    # (B,N,H)
        logits = self.score(emb).squeeze(-1)        # (B,N)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class MasterEVJointCritic(nn.Module):
    """Kritik terpusat (CTDE) yang menilai SEMUA permintaan EV pada satu TIMESTEP
    sekaligus -- blok Transformer standar (bukan pooling-ke-satu-vektor spt
    `AttentiveJointPooling`, krn di sini butuh nilai TERPISAH per-EV, bukan satu Q
    global): self-attention antar token-EV -> feed-forward -> Q-head per token.

    `priv` (privileged, Delayed Access Strategy): future_avail stasiun yang
    DIREKOMENDASIKAN pada t+d (kebenaran, bukan prediksi) + gini_now -- HANYA dilihat
    kritik, tak pernah bocor ke aktor (sama batas `MasterAttentiveCritic`)."""

    def __init__(self, token_dim: int = TOKEN_DIM, d_model: int = 128, n_heads: int = 4,
                priv_dim: int = PRIV_DIM, n_critics: int = 1):
        super().__init__()
        self.n_critics = int(n_critics)
        self.embed = nn.Sequential(nn.Linear(token_dim, d_model), nn.ReLU())
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + priv_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, self.n_critics),
        )

    def forward(self, tokens, key_padding_mask, priv):
        """tokens:(B,Nt,11) key_padding_mask:(B,Nt) bool (True=PAD, dikeluarkan dari
        atensi) priv:(B,Nt,2) -> Q:(B,Nt,n_critics), attn_weights:(B,Nt,Nt) rerata
        kepala (dikembalikan `nn.MultiheadAttention` bawaan, avg antar-kepala)."""
        e = self.embed(tokens)                                          # (B,Nt,128)
        attn_out, attn_w = self.attn(e, e, e, key_padding_mask=key_padding_mask,
                                     need_weights=True)
        e = self.ln1(e + attn_out)
        e = self.ln2(e + self.ff(e))
        out = self.head(torch.cat([e, priv], dim=-1))                    # (B,Nt,K)
        return out, attn_w
