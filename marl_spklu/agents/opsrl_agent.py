"""Baseline 3 -- adaptasi OP-SRL (Liao dkk., 2024; §II.6.3 tesis).

OP-SRL asli menyapu tingkat kepatuhan 0-100% dan memakai constrained-PPO sistem-level
dengan batasan keamanan jaringan listrik. Tesis ini mengadaptasinya sebagai baseline
pembanding literatur dengan penyederhanaan yang eksplisit dinyatakan (Bab IV, Baseline 3):
kepatuhan EKSOGEN dan PENUH (ditetapkan tetap = 1, bukan tumbuh/menyusut dari pengalaman
seperti trust endogen pada agen MARL penelitian ini), dan rekomendasi berbasis prediksi
waktu tunggu sistem-level (bukan multi-agen berbagi-parameter). Kepatuhan penuh diterapkan
di level harness (lihat `experiments.harness.force_full_compliance`), bukan pada agen ini --
agen di sini murni menghasilkan rekomendasi.
"""
from marl_spklu.agents.wait_predictor import DeterministicWaitPredictor


class OPSRLAgent:
    """S3: rekomendasi tunggal ke SPKLU dengan estimasi waktu tunggu PREDIKTIF terendah
    (bukan hanya antrean instan seperti S1_greedy) -- dipasangkan dengan kepatuhan penuh
    eksogen (lihat harness) untuk mereplikasi asumsi OP-SRL."""

    def __init__(self, wait_predictor=None):
        self.wait_predictor = wait_predictor or DeterministicWaitPredictor()

    def predict_waits(self, spklus: dict) -> dict:
        return self.wait_predictor.predict(spklus)

    def get_recommendation(self, spklus: dict) -> list:
        if not spklus:
            return []
        waits = self.predict_waits(spklus)
        best = min(spklus.keys(), key=lambda sid: waits.get(sid, 0.0))
        return [best]
