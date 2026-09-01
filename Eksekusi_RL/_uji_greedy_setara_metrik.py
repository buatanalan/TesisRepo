"""Evaluasi greedy dengan KANAL JANJI DISETARAKAN penuh dengan lengan RL.

Menutup TIGA ketidaksetaraan yang ditemukan 2026-08-31 saat inspeksi Blok A. Ketiganya
membuat perbandingan greedy-vs-RL tidak sah, dan ketiganya berada di sisi greedy:

1. ESTIMATOR. Greedy baku memakai `VirtualWaitPredictor` -> `sim.compute_virtual_wait`
   (presisi: `remaining_time` aktual tiap EV + EV yang sedang TRAVELING). Lengan RL
   memakai `FormulaForecaster` (kasar: rerata waktu-charge TETAP per tipe konektor).
   -> di sini greedy dipaksa memakai `FormulaForecaster` yang sama.

2. UKURAN HIMPUNAN REKOMENDASI. `User.decide_spklu` mendefinisikan
   `complied = chosen_spklu in rec_set`, sehingga himpunan yang lebih besar menaikkan
   `acc` SECARA MEKANIS. Greedy baku top_k=2 (nilai itu dipilih semata untuk menambal
   bug P_rec pada k=1, bukan untuk menyamai RL), lengan RL k=3.
   -> `TOP_K` di sini disetel eksplisit; pakai 3 untuk setara.

3. CAKUPAN KANAL JANJI. Lengan RL hanya menerbitkan estimasi untuk stasiun yang
   DIREKOMENDASIKAN; sisanya `inf` (= tidak ada janji yang dibuat) --
   `master_pure_hybrid_trainer.py:154`. Greedy baku menerbitkan estimasi untuk SELURUH
   stasiun feasible, termasuk yang tak pernah ia rekomendasikan -- "janji hantu" yang
   tak pernah ditampilkan ke pengguna. Akibatnya `est_pref` greedy nyaris selalu
   terdefinisi (terukur 0,0% terekslusi) sementara lengan RL kehilangan setiap kasus
   yang stasiun preferensinya di luar rekomendasi, sehingga metrik manfaat
   (`jn*_expost_*`) kedua lengan TIDAK dihitung dari populasi yang sama.
   -> `GreedySetara` di bawah menutup stasiun non-rekomendasi menjadi `inf`.

CATATAN penafsiran: `inf` adalah semantik yang BENAR (tidak ada janji = tidak ada angka
sah untuk dibandingkan), jadi perbaikan dilakukan dengan menyamakan greedy ke perilaku
RL, bukan sebaliknya. Konsekuensinya kedua lengan kini sama-sama membuang kasus
"preferensi di luar rekomendasi" dari metrik manfaat -- perbandingan menjadi setara,
tetapi cakupannya menyempit dan hal itu WAJIB dinyatakan saat melaporkan.

Keluaran ke berkas TERPISAH; berkas acuan `uji_greedy_metrik_*.json` TIDAK ditimpa.

Pemakaian:
    python _uji_greedy_setara_metrik.py 0,1,2 90d 3        # setara penuh dgn lengan RL
    python _uji_greedy_setara_metrik.py 0,1,2 90d 2        # k=2 (baku greedy lama)
    python _uji_greedy_setara_metrik.py 0,1,2 90d 3 penuh  # + 4 kombinasi mode x rezim
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

import re as _re

SEEDS = ([int(s) for s in sys.argv[1].replace(" ", "").split(",") if s]
        if len(sys.argv) > 1 else [0, 1, 2])
TAG = sys.argv[2] if len(sys.argv) > 2 else "90d"
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 3

# --- KONDISI LINGKUNGAN (ditambahkan 2026-09-01) --------------------------------------
# Blok E membandingkan 4 sel PURE3 lintas 8 kondisi TANPA pembanding non-RL, sehingga
# tak dapat dibedakan "arsitektur mana yang lebih tahan" dari "kondisi ini memang lebih
# sulit bagi agen apa pun". Greedy tidak perlu dilatih ulang -- cukup dievaluasi di
# lingkungan yang sama -- jadi acuan itu dapat dilengkapi tanpa pelatihan baru.
#
# Token kondisi memakai EJAAN YANG SAMA dgn akhiran TAG_ARM lengan RL
# (`_uji_master_pure_hybrid_ppo_metrik.py` baris 45-50), supaya kesetaraan lingkungan
# dapat diperiksa dgn membandingkan string, bukan dengan mengingat.
#
#   it0.3 / it0.7        -> ablations.initial_trust
#   gw0.0279513 / gw0.111805 -> ablations.gamma_est_wait  (sensitivitas P_rec, BUKAN
#                               diskon PPO)
#   load6x               -> dataset scenario_dataset_klaster12_6x_90d.json
#
# TIDAK ADA padanan greedy untuk `g0.95` / `g0.999`: itu faktor diskon PPO/GAE, parameter
# ALGORITMA. Greedy tak punya fungsi nilai, jadi kedua kondisi itu memakai acuan `baku`
# -- dan fakta bahwa acuannya identik justru informasi yang perlu dilaporkan.
PENUH = any(a.lower() == "penuh" for a in sys.argv[4:])
_kond = [a for a in sys.argv[4:] if a.lower() != "penuh"]
KOND = _kond[0] if _kond else "baku"

DATASET_KOND = {"load6x": "scenario_dataset_klaster12_6x_90d.json"}
if KOND in DATASET_KOND:
    K.DS = os.path.join(common.ROOT, DATASET_KOND[KOND])
else:
    K.DS = os.path.join(common.ROOT, K.HORIZON[TAG])

_m_it = _re.search(r"^it([0-9.]+)$", KOND)
INIT_TRUST = float(_m_it.group(1)) if _m_it else None
_m_gw = _re.search(r"^gw([0-9.]+)$", KOND)
GAMMA_EST_WAIT = float(_m_gw.group(1)) if _m_gw else None

if KOND != "baku" and INIT_TRUST is None and GAMMA_EST_WAIT is None         and KOND not in DATASET_KOND:
    raise SystemExit(f"kondisi tak dikenal: {KOND!r} "
                     f"(baku | it<x> | gw<x> | {' | '.join(DATASET_KOND)})")


class AdapterFormula:
    """Menjembatani antarmuka `FormulaForecaster` ke kontrak `wait_predictor` yang
    dipanggil `GreedyAgent.predict_waits(spklus, sim=, user=, time_now=)` --
    forecaster memakai `time_now_min`, prediktor greedy memakai `time_now`."""

    def __init__(self):
        self.f = FormulaForecaster()

    def predict(self, spklus: dict, sim=None, user=None, time_now: float = 0.0) -> dict:
        return self.f.predict(spklus, time_now_min=time_now, user=user, sim=sim)


class GreedySetara(GreedyAgent):
    """Greedy dgn cakupan kanal janji disamakan (ketidaksetaraan #3 di docstring modul).

    Hanya stasiun yang benar-benar DIREKOMENDASIKAN memperoleh estimasi; sisanya `inf`,
    persis `master_pure_hybrid_trainer.py:154`. Aman karena `Simulator` memanggil
    `get_recommendation` (baris 436) SEBELUM `predict_waits` (baris 451) pada iterasi
    pengguna yang sama, sehingga himpunan terakhir selalu mutakhir saat dipakai.

    Kebijakan pemilihan greedy TIDAK berubah: `_score()` menilai stasiun langsung dari
    objek SPKLU, tak pernah lewat `wait_predictor`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recs_terakhir = []

    def get_recommendation(self, spklus: dict) -> list:
        recs = super().get_recommendation(spklus)
        self._recs_terakhir = list(recs)
        return recs

    def predict_waits(self, spklus: dict, sim=None, user=None,
                      time_now: float = 0.0) -> dict:
        est = super().predict_waits(spklus, sim=sim, user=user, time_now=time_now)
        direkomendasikan = set(self._recs_terakhir)
        return {sid: (v if sid in direkomendasikan else float("inf"))
                for sid, v in est.items()}


ARMS = [
    ("greedy_util",  lambda sim: GreedySetara(mode="utilization", top_k=TOP_K,
                                              wait_predictor=AdapterFormula())),
    ("greedy_queue", lambda sim: GreedySetara(mode="queue", top_k=TOP_K,
                                              wait_predictor=AdapterFormula())),
]

KOMBO = ([(m, b) for m in ("abs", "signed") for b in (False, True)] if PENUH
         else [("signed", False)])


def main():
    print(f"horizon={TAG} ({K.DS})", flush=True)
    print(f"seed: {SEEDS}", flush=True)
    print("PENYETARAAN thd lengan RL:", flush=True)
    print("  1. estimator  : FormulaForecaster (bukan VirtualWaitPredictor)", flush=True)
    print(f"  2. top_k      : {TOP_K}" +
          ("  <- setara K_REC=3" if TOP_K == 3 else "  <- BELUM setara (RL memakai 3)"),
          flush=True)
    print("  3. kanal janji: hanya stasiun terekomendasi; sisanya inf", flush=True)
    print(f"kondisi   : {KOND}  (dataset {os.path.basename(K.DS)})", flush=True)
    if INIT_TRUST is not None:
        print(f"  initial_trust  = {INIT_TRUST:g}", flush=True)
    if GAMMA_EST_WAIT is not None:
        print(f"  gamma_est_wait = {GAMMA_EST_WAIT:g}", flush=True)
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
                 f"pct_tepat={a.get('pct_tepat', float('nan')):.1f} "
                 f"untung%={a.get('jnpatuh_expost_frac_untung', float('nan')):.3f}",
                 flush=True)

    out = dict(horizon=TAG, seeds=SEEDS, per_seed=per_seed, agregat=agregat,
               harian=harian, estimator_janji="FormulaForecaster", top_k=TOP_K,
               kanal_janji="hanya_terekomendasi_sisanya_inf",
               kondisi=KOND, dataset=os.path.basename(K.DS),
               initial_trust=INIT_TRUST, gamma_est_wait=GAMMA_EST_WAIT)
    # Akhiran kondisi disisipkan SEBELUM "_metrik_" agar tetap cocok dgn pola
    # `uji_{tag}_metrik_{horizon}.json` yang dibaca `_analisis_bab5.muat()`.
    sfx = "" if KOND == "baku" else f"_{KOND}"
    nama = f"uji_greedy_setara_k{TOP_K}{sfx}_metrik_{TAG}.json"
    common.save_json(out, nama)
    print(f"\nSAVED -> outputs/{nama}", flush=True)


if __name__ == "__main__":
    # Konteks membentang menutupi SELURUH main(): `K.satu_run` membentuk simulator baru
    # tiap run dan `initial_trust` menambal `User.__init__`, jadi harus aktif saat setiap
    # simulator dibentuk -- bukan sekali di awal. Pola ini disalin persis dari
    # `_uji_master_pure_hybrid_ppo_metrik.py` agar kedua sisi perbandingan menjalankan
    # ablasi yang sama dengan cara yang sama.
    import contextlib
    _ctx = contextlib.ExitStack()
    if INIT_TRUST is not None:
        from marl_spklu.experiments.ablations import initial_trust as _initial_trust
        _ctx.enter_context(_initial_trust(INIT_TRUST))
    if GAMMA_EST_WAIT is not None:
        from marl_spklu.experiments.ablations import gamma_est_wait as _gamma_est_wait
        _ctx.enter_context(_gamma_est_wait(GAMMA_EST_WAIT))
    with _ctx:
        main()
