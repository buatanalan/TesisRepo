from marl_spklu.agents.wait_predictor import VirtualWaitPredictor


class GreedyAgent:
    """
    S1: Rule-based agent (Non-RL). Merekomendasikan SPKLU "terbaik saat ini" menurut satu
    sinyal instan. Secara teoritis memicu Herding Effect besar (semua pemohon dalam satu
    langkah diarahkan ke SPKLU yang sama).

    `mode` menentukan SINYAL yang dipakai -- pilihan ini TIDAK netral, ia menentukan
    kekuatan baseline secara drastis (uji 7-hari, mu_hat=0.8, Gini Natural 0.5303):
      "utilization" : (antre+charging)/kapasitas       -> Gini 0.4539  (jauh lebih baik)
      "queue"       : panjang antrean mentah           -> Gini 0.5514  (LEBIH BURUK dari Natural)
      "wait"        : estimasi waktu tunggu (M/M/c-ish)-> lihat SPKLU.estimate_wait_time

    Sebabnya: metrik evaluasi (Gini utilisasi) tersusun PERSIS dari besaran yang dipakai
    mode "utilization", sehingga aturan itu praktis melakukan penurunan langsung pada
    fungsi tujuan. Mode "queue"/"wait" memakai sinyal yang berbeda (lagging & bising),
    sehingga tidak menikmati keuntungan itu.

    Acuan literatur:
      - `spesifikasi_teknis_pdqn_baseline.md` §2.3 mendefinisikan Greedy sebagai
        argmin_j S_j(k)/kapasitas_j  -> mode "utilization".
      - Lin et al. 2024 (paper PDQN) memakai "minimum queuing mechanism", yaitu SPKLU
        dgn waktu antre terkecil -> mode "wait" (atau "queue" sbg proksi kasarnya).
    """

    MODES = ("utilization", "queue", "wait")

    def __init__(self, wait_predictor=None, mode: str = "utilization", top_k: int = 2):
        if mode not in self.MODES:
            raise ValueError(f"mode harus salah satu dari {self.MODES}, dapat {mode!r}")
        self.mode = mode
        # Perbaikan T2 (Validasi_Generik/LAPORAN_VALIDASI.md): default kini
        # VirtualWaitPredictor (presisi -- pakai compute_virtual_wait, memperhitungkan
        # EV yang sedang menuju stasiun), bukan lagi DeterministicWaitPredictor (selalu
        # 0 saat ada slot bebas sesaat, akar penyebab EstWait degenerat).
        self.wait_predictor = wait_predictor or VirtualWaitPredictor()
        # Perbaikan T2 (Validasi_Generik/LAPORAN_VALIDASI.md, penyebab kedua): Top-K=1
        # (lama) membuat P_rec = softmax atas SATU elemen SELALU 1,0 apa pun `gamma`-nya
        # -- gamma jadi inert secara struktural, terlepas dari akurasi EstWait (penyebab
        # pertama, sudah diperbaiki di SPKLU.estimate_wait_time). top_k>=2 membuat
        # decide_spklu (marl_spklu/env/user.py) benar2 membandingkan >1 kandidat via
        # softmax(exp(-gamma*wait)), sesuai spesifikasi P_rec §3.2.
        self.top_k = max(1, int(top_k))

    def predict_waits(self, spklus: dict, sim=None, user=None, time_now: float = 0.0) -> dict:
        """EstWait yang dijanjikan agen ke pengguna (dipakai Simulator). `sim`/`user`/
        `time_now` opsional -- diteruskan ke VirtualWaitPredictor kalau tersedia;
        diabaikan dgn aman oleh prediktor lama (DeterministicWaitPredictor)."""
        return self.wait_predictor.predict(spklus, sim=sim, user=user, time_now=time_now)

    def _score(self, spklu) -> float:
        """Skor yang DIMINIMALKAN. Lihat catatan kelas soal dampak pilihan mode."""
        if self.mode == "utilization":
            total_evs = sum(len(q) for q in spklu.queues.values())
            total_evs += sum(len(c) for c in spklu.charging.values())
            total_cap = sum(spklu.capacities.values())
            return total_evs / total_cap if total_cap > 0 else 0.0
        if self.mode == "queue":
            return float(spklu.get_queue_length())
        # mode "wait": estimasi tunggu terkecil antar tipe konektor yang tersedia
        waits = [spklu.estimate_wait_time(ct) for ct in spklu.capacities]
        waits = [w for w in waits if w is not None]
        return min(waits) if waits else float("inf")

    def get_recommendation(self, spklus: dict) -> list:
        """Merekomendasikan `top_k` SPKLU dgn skor terendah menurut `mode`, terurut
        skor menaik (elemen [0] = pilihan terbaik/utama, dipakai Simulator utk
        pembukuan herding/flocking -- tak berubah dari perilaku lama). Elemen
        selanjutnya memperluas himpunan rekomendasi yg dilihat `User.decide_spklu`
        utk P_rec (lihat catatan `top_k` di __init__)."""
        scores = {sid: self._score(spklu) for sid, spklu in spklus.items()}
        # Urutkan (skor menaik, lalu ID abjad) supaya deterministik saat ada yg seri.
        ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
        return [sid for sid, _ in ranked[:self.top_k]]
