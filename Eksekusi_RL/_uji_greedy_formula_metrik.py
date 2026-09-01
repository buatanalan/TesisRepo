"""Evaluasi greedy dgn ESTIMATOR DISAMAKAN dengan lengan RL (`FormulaForecaster`).

MENGAPA ADA: `_uji_greedy_metrik.py` menjalankan `GreedyAgent` dengan prediktor
bawaannya, `VirtualWaitPredictor` -> `sim.compute_virtual_wait` (presisi tinggi:
`remaining_time` AKTUAL tiap EV yang sedang charging + EV yang sudah TRAVELING menuju
stasiun). Sementara lengan RL menampilkan janji dari `FormulaForecaster` (KASAR:
rerata waktu-charge TETAP per tipe konektor, AC~136mnt/DC~49mnt, buta thd
`remaining_time` sesungguhnya).

Akibatnya metrik yang bersandar pada GALAT JANJI -- `galat_*`, `pct_tepat`,
`pct_telat`, dan sebagian dinamika `trust` -- TIDAK SETARA antar-lengan: greedy
diuntungkan estimator yang lebih presisi, bukan oleh kualitas rekomendasinya.
Terukur pada evaluasi 90d: galat_mean greedy_queue -3,4 mnt vs lengan RL -8,8 mnt,
pct_tepat 47,6% vs 37,7%.

Skrip ini menyamakan basis janji supaya perbandingan trust/kalibrasi menjadi sah.

CATATAN PENTING -- yang berubah HANYA janji, bukan kebijakan greedy:
    `GreedyAgent._score()` (penentu SPKLU mana yang direkomendasikan) memakai
    queue_length / utilisasi langsung dari objek SPKLU, TIDAK lewat `wait_predictor`.
    Jadi urutan rekomendasi greedy IDENTIK dengan sebelumnya; yang berbeda semata
    angka waktu tunggu yang DITAMPILKAN ke pengguna (dan karenanya menjadi dasar
    `User.update_trust` serta `P_rec`). Ini persis isolasi yang dikehendaki.

    Konsekuensi lanjutan yang WAJIB diakui saat menafsirkan: karena `P_rec` ikut
    memakai angka ini, keputusan pengguna juga bergeser -- jadi `acc`/`gini`/`wait`
    greedy di sini TIDAK identik dengan berkas lama. Yang dibandingkan bukan "greedy
    lama vs baru", melainkan "greedy dan RL pada basis janji yang sama".

Keluaran ke berkas TERPISAH (`uji_greedy_formula_metrik_*.json`) -- berkas acuan
`uji_greedy_metrik_*.json` TIDAK ditimpa.

Pemakaian:
    python _uji_greedy_formula_metrik.py 0,1,2 90d          # top_k=2 (baku greedy)
    python _uji_greedy_formula_metrik.py 0,1,2 90d 3        # top_k=3 (setara lengan RL)
    python _uji_greedy_formula_metrik.py 0,1,2 90d 3 penuh  # + 4 kombinasi mode x rezim
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

_argv_asli = sys.argv
sys.argv = ["_uji_konsolidasi.py", "0", "30d"]
import _uji_konsolidasi as K
sys.argv = _argv_asli

from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.rl.forecaster import FormulaForecaster

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "90d"
# `top_k` = ukuran himpunan rekomendasi. KONFOUND KEDUA yang ditemukan 2026-08-31:
# `User.decide_spklu` mendefinisikan `complied = chosen_spklu in rec_set`, sehingga
# himpunan yang lebih besar menaikkan `acc` SECARA MEKANIS terlepas dari kualitas
# rekomendasi. Lengan RL memakai k=3 (`_uji_master_pure_hybrid_ppo_metrik.py::K_REC`)
# sedangkan greedy baku top_k=2 -- perbandingan `acc` antar keduanya TIDAK setara
# sampai k disamakan.
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
PENUH = len(sys.argv) > 4 and sys.argv[4].lower() == "penuh"
K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])


class AdapterFormula:
    """Menyesuaikan antarmuka `FormulaForecaster` ke kontrak `wait_predictor` yang
    dipanggil `GreedyAgent.predict_waits(spklus, sim=, user=, time_now=)`.

    Perbedaan antarmuka yang dijembatani: forecaster memakai `time_now_min` (dan
    menerima `soc`), sedangkan prediktor greedy memakai `time_now`. Tanpa adapter,
    `FormulaForecaster` akan menerima `time_now` sebagai argumen tak dikenal.
    """

    def __init__(self):
        self.f = FormulaForecaster()

    def predict(self, spklus: dict, sim=None, user=None, time_now: float = 0.0) -> dict:
        return self.f.predict(spklus, time_now_min=time_now, user=user, sim=sim)


ARMS = [
    ("greedy_util",  lambda sim: GreedyAgent(mode="utilization", top_k=TOP_K,
                                             wait_predictor=AdapterFormula())),
    ("greedy_queue", lambda sim: GreedyAgent(mode="queue", top_k=TOP_K,
                                             wait_predictor=AdapterFormula())),
]

KOMBO = ([(m, b) for m in ("abs", "signed") for b in (False, True)] if PENUH
         else [("signed", False)])


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)
    print("estimator janji: FormulaForecaster (DISAMAKAN dgn lengan RL, "
          "bukan VirtualWaitPredictor bawaan greedy)", flush=True)
    print(f"top_k = {TOP_K}" + ("  (DISAMAKAN dgn K_REC=3 lengan RL)" if TOP_K == 3
          else "  (baku greedy; lengan RL memakai 3 -- `acc` TIDAK setara)"), flush=True)
    print(f"kombinasi: {KOMBO}\n", flush=True)

    per_seed, agregat, harian = {}, {}, {}
    for lbl, fac in ARMS:
        for mode, beku in KOMBO:
            label = f"{lbl}|{mode}|{'beku' if beku else 'dinamis'}"
            runs = []
            for sd in SEEDS:
                print(f"  [{label}] seed={sd} ...", flush=True)
                runs.append(K.satu_run(fac, mode, sd, beku))
            per_seed[label] = runs
            agregat[label] = K.agg(runs)
            harian[label] = K.agg_harian(runs)
            a = agregat[label]
            print(f"  [{label}] gini={a['gini']:.4f} wait={a['wait']:.1f} "
                 f"trust={a['trust']:.3f} acc={a['acc']:.3f} "
                 f"galat_mean={a.get('galat_mean', float('nan')):+.2f} "
                 f"pct_tepat={a.get('pct_tepat', float('nan')):.1f}", flush=True)

    out = dict(horizon=TAG, seeds=SEEDS, per_seed=per_seed, agregat=agregat,
               harian=harian, estimator_janji="FormulaForecaster", top_k=TOP_K)
    nama = (f"uji_greedy_formula_metrik_{TAG}.json" if TOP_K == 2
            else f"uji_greedy_formula_k{TOP_K}_metrik_{TAG}.json")
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    main()
