import numpy as np

from marl_spklu.agents.wait_predictor import DeterministicWaitPredictor


class DiversifiedAgent:
    """
    S2: Agen heuristik ber-diversifikasi (Non-RL).

    Memperbaiki kelemahan `GreedyAgent` (S1) yang selalu memilih SPKLU ber-utilisasi
    global-minimum -> semua spawn dalam window yang sama diarahkan ke SPKLU yang sama
    -> herding maksimum. Agen ini menyebar rekomendasi lewat salah satu dari dua mode
    load-balancing klasik:

    - mode="d_choices" (default): "power-of-d-choices". Sampel `d` SPKLU acak, lalu
      rekomendasikan yang beban (antre+charging) terendah di antara sampel itu. Karena
      himpunan kandidat berbeda tiap panggilan, spawn berbeda mendapat rekomendasi
      berbeda -> herding turun drastis, tetapi tetap mengarahkan ke SPKLU relatif sepi.

    - mode="proportional": pilih SPKLU dengan probabilitas softmax atas beban negatif
      (exp(-load/tau)). Menyebar rekomendasi secara halus; `tau` mengatur ketajaman
      (tau kecil -> mendekati greedy; tau besar -> mendekati acak-seragam).

    Sengaja memakai antarmuka tipis `get_recommendation(spklus)` yang sama dengan
    GreedyAgent (hanya butuh info beban), sehingga bisa langsung dipasang di Simulator
    tanpa perubahan arsitektur. Randomness memakai np.random global (di-seed harness)
    demi reprodusibilitas lintas seed.
    """

    def __init__(self, mode: str = "d_choices", d: int = 2, tau: float = 1.0,
                 wait_predictor=None):
        assert mode in ("d_choices", "proportional"), f"mode tidak dikenal: {mode}"
        self.mode = mode
        self.d = max(2, int(d))
        self.tau = float(tau)
        # Agen kini MEMILIKI prediktor waktu tunggu (default: deterministik = status-quo).
        self.wait_predictor = wait_predictor or DeterministicWaitPredictor()

    def predict_waits(self, spklus: dict) -> dict:
        """EstWait yang dijanjikan agen ke pengguna (dipakai Simulator)."""
        return self.wait_predictor.predict(spklus)

    @staticmethod
    def _load(spklu) -> int:
        """Beban instan = jumlah EV yang sedang antre + sedang charging."""
        q = sum(len(x) for x in spklu.queues.values())
        c = sum(len(x) for x in spklu.charging.values())
        return q + c

    def get_recommendation(self, spklus: dict) -> list:
        sids = list(spklus.keys())
        if not sids:
            return []

        if self.mode == "d_choices":
            k = min(self.d, len(sids))
            # Sampel k kandidat acak tanpa pengembalian, pilih beban terendah.
            idx = np.random.choice(len(sids), size=k, replace=False)
            cand = [sids[i] for i in idx]
            chosen = min(cand, key=lambda s: self._load(spklus[s]))
            return [chosen]

        # mode == "proportional"
        loads = np.array([self._load(spklus[s]) for s in sids], dtype=float)
        logits = -loads / max(self.tau, 1e-6)
        logits -= logits.max()  # stabilitas numerik
        probs = np.exp(logits)
        probs /= probs.sum()
        chosen = str(np.random.choice(sids, p=probs))
        return [chosen]
