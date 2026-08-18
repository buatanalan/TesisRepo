"""Registri kelas kebijakan -- SATU sumber untuk seluruh skrip evaluasi.

LATAR
-----
Sebelum berkas ini ada, tiap skrip evaluasi menyalin sendiri logika resolusi kelas, dan
sebagian masih memakai bentuk biner lama:

    c = PPPOPolicy if meta["policy_cls"] == "PPPOPolicy" else HPPOPolicy

Bentuk itu memuat SELURUH lengan MASTER sebagai `HPPOPolicy`. Untungnya gagal keras
(bentuk bobot tak cocok pada `load_state_dict`), bukan diam-diam salah -- tetapi tiap
lengan baru mengulang jebakan yang sama. Registri di sini menutupnya sekali.

Menambah lengan baru: cukup daftarkan di `KELAS_KEBIJAKAN`, dan bila kelasnya butuh
argumen konstruktor tambahan, tambahkan di `_kwargs_khusus`.
"""
from marl_spklu.rl.policy import HPPOPolicy
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.rl.master_policy import (MasterPolicy, MasterPolicyEqCap,
                                         MasterPrefPolicy)
from marl_spklu.rl.master_bidding_policy import (MasterBiddingPolicy,
                                                 BiddingHistPolicy,
                                                 BiddingPrefPolicy)

KELAS_KEBIJAKAN = {
    "HPPOPolicy": HPPOPolicy,
    "PPPOPolicy": PPPOPolicy,
    "MasterPolicy": MasterPolicy,
    "MasterPolicyEqCap": MasterPolicyEqCap,
    "MasterPrefPolicy": MasterPrefPolicy,
    "MasterBiddingPolicy": MasterBiddingPolicy,
    "BiddingHistPolicy": BiddingHistPolicy,
    "BiddingPrefPolicy": BiddingPrefPolicy,
}

# Kelas yang memuat modul preferensi PDQN -- dimensinya harus dibaca dari meta, karena
# nilai bakunya (64) BERBEDA dari nilai yang dipakai eksperimen (16).
_BUTUH_DIM_PREF = (PPPOPolicy, MasterPrefPolicy, BiddingPrefPolicy)


def kelas_dari_meta(meta):
    """Kelas kebijakan sesuai `meta["policy_cls"]`.

    Baku `HPPOPolicy` dipertahankan demi checkpoint lama yang metanya belum memuat kunci
    ini. Nama yang TIDAK dikenal dianggap kesalahan -- membiarkannya jatuh ke baku akan
    mengulang persis bug yang berkas ini tutup.
    """
    nama = meta.get("policy_cls", "HPPOPolicy")
    if nama not in KELAS_KEBIJAKAN:
        raise KeyError(
            f"policy_cls tak dikenal: {nama!r}. Daftarkan di "
            f"marl_spklu/rl/registry.py::KELAS_KEBIJAKAN. "
            f"Terdaftar: {sorted(KELAS_KEBIJAKAN)}")
    return KELAS_KEBIJAKAN[nama]


def _kwargs_khusus(cls, meta):
    kw = dict(n_critics=meta.get("n_critics", 1))
    if cls in _BUTUH_DIM_PREF:
        kw.update(pref_d_lstm=meta.get("pref_d_lstm", 64),
                  pref_d_attn=meta.get("pref_d_attn", 64))
    return kw


def bangun_kebijakan(meta, state_dict=None, eval_mode=True):
    """Bangun kebijakan dari meta; muat bobot bila `state_dict` diberikan."""
    cls = kelas_dari_meta(meta)
    pol = cls(meta["obs_dim"], meta["critic_obs_dim"], meta["N"],
              **_kwargs_khusus(cls, meta))
    if state_dict is not None:
        pol.load_state_dict(state_dict)
    if eval_mode:
        pol.eval()
    return pol
