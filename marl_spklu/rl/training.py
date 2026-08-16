"""Orkestrasi pelatihan EV Agent + forecaster — mode A/B/C/D.
Selaras Spesifikasi_RL_EV_LoadBalancing.md §9.1.

Sisi FORECASTER (non-RL) diimplementasi penuh & teruji di sini. Sisi POLICY (PPO/H-PPO)
adalah **titik-colok**: injeksikan trainer nyata (mis. Stable-Baselines3 / torch) lewat
protokol `PolicyTrainer`/`OnlinePolicyTrainer`. `FakePolicyTrainer` disediakan hanya
untuk smoke-test alur (tidak melatih apa pun).

Perbedaan gradien penting: forecaster (supervised) & policy (RL) tak berbagi gradien,
sehingga keempat mode aman — tidak ada non-stationarity antar-pembelajar RL.
"""
from typing import Protocol, runtime_checkable
import numpy as np

from marl_spklu.env.simulator import Simulator
from marl_spklu.rl.forecaster import (
    FormulaForecaster, LearnedForecaster, collect_forecast_dataset,
)


# ----------------------------- Protokol trainer (titik-colok) -----------------------------
@runtime_checkable
class PolicyTrainer(Protocol):
    """Trainer PPO offline/batch. Implementasi nyata membungkus H-PPO (§8)."""
    policy: object

    def train(self, forecaster, n_updates: int) -> object:
        """Latih policy `n_updates` update dgn `forecaster` memasok ŵ. Return policy."""
        ...

    def make_agent(self, forecaster):
        """Bungkus policy terkini jadi agent Simulator (get_recommendation/predict_waits),
        untuk mengumpulkan trajektori distribusi-baru (mode C). Boleh None (pakai baseline)."""
        ...


@runtime_checkable
class OnlinePolicyTrainer(Protocol):
    """Trainer untuk mode D: satu langkah PPO sekaligus menghasilkan pasangan
    (fitur, wait_aktual) dari rollout untuk partial_fit forecaster."""
    policy: object

    def train_step(self, forecaster):
        """Satu update PPO. Return (X, y) pasangan supervised dari rollout ini."""
        ...


class FakePolicyTrainer:
    """Placeholder non-fungsional untuk smoke-test alur mode. Ganti dgn PPO nyata."""

    def __init__(self):
        self.policy = None
        self.log = []

    def train(self, forecaster, n_updates: int):
        self.log.append(("train", n_updates, type(forecaster).__name__))
        return self.policy

    def make_agent(self, forecaster):
        return None  # -> pengumpulan mode C jatuh ke baseline (None agent)

    def train_step(self, forecaster):
        self.log.append(("train_step", type(forecaster).__name__))
        return np.zeros((0, 7)), np.zeros(0)


def _fresh_sim(dataset_path, willingness_ratio=None, willingness_radius_km=None) -> Simulator:
    """Perbaikan T1-serupa (Validasi_Generik/LAPORAN_VALIDASI.md, Addendum 2 §"Fase 4"):
    default SEMULA `willingness_ratio=5.0` menyimpang dari jalur kalibrasi (`None`, Tier
    6/9) -- sama seperti bug lama `harness.DEFAULT_WILLINGNESS_RATIO`. `wait_mean` baseline
    yang jadi basis `tol_low`/`tol_high`/`gamma` (marl_spklu/env/user.py) dikalibrasi pada
    `willingness_ratio=None`; memakai `5.0` di sini membuat ambang trust training salah
    skala relatif dinamika wait yang sesungguhnya terjadi."""
    sim = Simulator({}, [], None,
                    user_willingness_radius_km=willingness_radius_km,
                    user_willingness_ratio=willingness_ratio)
    sim.load_from_dataset(dataset_path)
    return sim


# ----------------------------- Mode A -----------------------------
def train_mode_a(dataset_path, policy_trainer: PolicyTrainer, total_updates: int = 1000,
                 learned_forecaster: bool = True, collect_steps: int = 2880,
                 baseline_agent_factory=None):
    """Forecaster tetap selama training policy. Default (learned_forecaster=True):
    pretrain LearnedForecaster offline dari trajektori baseline lalu bekukan -- prediktor
    sadar-koordinasi (fitur aktivitas sistem phi_activity, §IV.1.5) hanya aktif pada
    LearnedForecaster, sehingga inilah pilihan yang menutup rantai K1 (non-stationarity
    endogen). Set learned_forecaster=False untuk forecaster formula murni tanpa
    pelatihan (ablasi/pembanding lama)."""
    if learned_forecaster:
        sim = _fresh_sim(dataset_path)
        baseline = baseline_agent_factory() if baseline_agent_factory else None
        X, y = collect_forecast_dataset(sim, min(collect_steps, sim.max_steps), agent=baseline)
        forecaster = LearnedForecaster().fit(X, y)
    else:
        forecaster = FormulaForecaster()
    policy = policy_trainer.train(forecaster, total_updates)
    return policy, forecaster


# ----------------------------- Mode B -----------------------------
def train_mode_b(dataset_path, policy_trainer: PolicyTrainer, total_updates: int = 1000,
                 baseline_agent_factory=None, collect_steps: int = 2880):
    """Pretrain forecaster offline pada trajektori baseline → freeze → train policy."""
    sim = _fresh_sim(dataset_path)
    baseline = baseline_agent_factory() if baseline_agent_factory else None
    X, y = collect_forecast_dataset(sim, min(collect_steps, sim.max_steps), agent=baseline)
    forecaster = LearnedForecaster().fit(X, y)     # offline; setelah ini dibekukan
    policy = policy_trainer.train(forecaster, total_updates)
    return policy, forecaster, (X.shape[0],)


# ----------------------------- Mode C -----------------------------
def train_mode_c(dataset_path, policy_trainer: PolicyTrainer, rounds: int = 5,
                 updates_per_round: int = 200, collect_steps: int = 2880):
    """Alternating (DAgger-style): latih policy → kumpulkan distribusi baru → refit forecaster."""
    forecaster = LearnedForecaster()
    X_buf, y_buf = [], []
    policy = None
    for r in range(rounds):
        policy = policy_trainer.train(forecaster, updates_per_round)
        sim = _fresh_sim(dataset_path)
        agent_factory = policy_trainer.make_agent(forecaster)   # factory: sim -> agent
        agent = agent_factory(sim) if agent_factory else None   # policy terkini sbg agent
        X, y = collect_forecast_dataset(sim, min(collect_steps, sim.max_steps), agent=agent)
        X_buf.append(X); y_buf.append(y)
        forecaster.fit(np.vstack(X_buf), np.concatenate(y_buf))   # refit atas buffer kumulatif
    return policy, forecaster


# ----------------------------- Mode D -----------------------------
def train_mode_d(dataset_path, online_trainer: OnlinePolicyTrainer, total_updates: int = 1000):
    """Simultan: interleave update PPO & partial_fit forecaster. Forecaster memakai
    fallback formula sampai partial_fit pertama, menjaga observasi stabil di awal."""
    forecaster = LearnedForecaster()
    for u in range(total_updates):
        X, y = online_trainer.train_step(forecaster)
        if X is not None and len(X) > 0:
            forecaster.partial_fit(X, y)
    return online_trainer.policy, forecaster
