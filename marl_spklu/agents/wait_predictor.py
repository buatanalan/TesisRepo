"""Prediktor waktu tunggu yang DIMILIKI AGEN.

Perubahan arsitektur penting: EstWait yang "dijanjikan" ke pengguna kini diproduksi
oleh AGEN (via prediktor ini), bukan lagi diambil langsung dari estimasi deterministik
Simulator yang sama untuk semua. Dengan begitu, AKURASI prediksi menjadi properti agen
yang bisa berbeda-beda / dipelajari (RL) -> dan dampaknya terhadap laju pembangunan
trust bisa dipelajari secara terpisah.

Ground-truth (waktu tunggu aktual) tetap dari Simulator; selisihnya-lah yang menggerakkan
update trust (lihat User.update_trust). Prediktor di sini hanya menghasilkan JANJI.

Kelas:
  - DeterministicWaitPredictor : mereplikasi estimasi lama (status-quo, default -> tanpa
    perubahan perilaku). Basis untuk perbandingan.
  - NoisyWaitPredictor         : membungkus prediktor basis + galat (bias & noise) untuk
    MEMPELAJARI sensitivitas trust/outcome terhadap akurasi prediksi.
"""
import numpy as np


class DeterministicWaitPredictor:
    """Estimasi deterministik per-SPKLU (min antar tipe konektor), identik dengan
    logika lama Simulator.get_spklu_wait_estimates -> default backward-compatible.

    `**_ignored`: menerima (tapi mengabaikan) `sim`/`user`/`time_now` supaya bisa
    dipanggil seragam dgn `VirtualWaitPredictor` tanpa TypeError (lihat GreedyAgent)."""

    def predict(self, spklus: dict, **_ignored) -> dict:
        est = {}
        for sid, s in spklus.items():
            waits = []
            if s.capacities.get("AC", 0) > 0:
                waits.append(s.estimate_wait_time("AC"))
            if s.capacities.get("DC", 0) > 0:
                waits.append(s.estimate_wait_time("DC"))
            est[sid] = min(waits) if waits else float("inf")
        return est


class VirtualWaitPredictor:
    """Perbaikan T2 (Validasi_Generik/LAPORAN_VALIDASI.md): channel EstWait presisi
    tinggi, membungkus `Simulator.compute_virtual_wait` -- memperhitungkan travel time
    user PENGQUERY itu sendiri, `remaining_time` AKTUAL tiap EV yang sedang charging,
    dan waktu tiba EV lain yang SEDANG MENUJU stasiun (bukan hanya hitungan kasar spt
    `SPKLU.estimate_wait_time`, yang tak tahu identitas/ETA presisi pengquery).

    Butuh `sim` (instance Simulator) & `time_now` saat `predict()` dipanggil -- kalau
    `sim` tak diberikan (mis. dipanggil di luar Simulator oleh kode lama), fallback ke
    `DeterministicWaitPredictor` (perilaku lama, aman)."""

    def __init__(self):
        self._fallback = DeterministicWaitPredictor()

    def predict(self, spklus: dict, sim=None, user=None, time_now: float = 0.0) -> dict:
        if sim is None:
            return self._fallback.predict(spklus)
        return {sid: float(sim.compute_virtual_wait(user, s, time_now)) for sid, s in spklus.items()}


class NoisyWaitPredictor:
    """Prediktor tak-sempurna: membungkus prediktor basis lalu menambah bias & noise
    proporsional. Dipakai untuk skenario 'prediksi buruk' -> menguji apakah akurasi
    prediksi memang menentukan laju trust & efektivitas (mengisolasi kontribusi
    'agen memprediksi tunggu dengan akurat').

    rel_noise : std noise relatif (dikali max(wait, floor)); floor mencegah noise nol
                pada stasiun kosong (wait=0) agar galat tetap bisa muncul.
    bias      : geser sistematis (menit). bias<0 = cenderung under-promise (dihukum
                lebih keras oleh trust, alpha>beta).
    """

    def __init__(self, base=None, rel_noise: float = 0.0, bias: float = 0.0,
                 floor: float = 5.0):
        self.base = base or DeterministicWaitPredictor()
        self.rel_noise = float(rel_noise)
        self.bias = float(bias)
        self.floor = float(floor)

    def predict(self, spklus: dict, **kwargs) -> dict:
        est = self.base.predict(spklus, **kwargs)
        out = {}
        for sid, v in est.items():
            if v == float("inf"):
                out[sid] = v
                continue
            scale = max(v, self.floor)
            noise = np.random.normal(0.0, self.rel_noise * scale) if self.rel_noise > 0 else 0.0
            out[sid] = max(0.0, v + self.bias + noise)
        return out
