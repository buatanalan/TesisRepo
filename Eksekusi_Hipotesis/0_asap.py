"""E0 -- uji asap: memastikan pengaturan kepercayaan BENAR-BENAR tersambung.

Bukan sekadar "jalan tanpa galat". Yang diperiksa: apakah `--initial-trust 0.3` benar
menghasilkan populasi ber-kepercayaan awal 0,3, dan apakah `--constant-trust` benar
membekukannya. Pipeline yang menerima pengaturan lalu mengabaikannya akan lolos uji
"tidak galat" tetapi membuat SELURUH E2/E4/E7 tak berarti -- kegagalan senyap yang
paling mahal, karena baru ketahuan setelah berhari-hari komputasi.

Tidak melatih apa pun. Hanya membuat simulasi, mengambil cuplikan kepercayaan, dan
menjalankan simulasi pendek untuk melihat apakah nilainya bergerak.

    python _asap_e0.py
"""
import sys, os, random, contextlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Eksekusi_RL"))
sys.path.insert(0, _ROOT)

import numpy as np
import common
from marl_spklu.agents.greedy_agent import GreedyAgent
from marl_spklu.experiments.ablations import (initial_trust, constant_trust,
                                              constant_trust_shadow)

DATASET = os.path.join(common.ROOT, "scenario_dataset_klaster12_4x.json")
LANGKAH = 400          # cukup untuk memicu sejumlah pembaruan kepercayaan
TOLERANSI = 0.02
lulus, gagal = [], []


def cek(nama, syarat, rincian=""):
    (lulus if syarat else gagal).append(nama)
    print(f"  [{'OK  ' if syarat else 'GAGAL'}] {nama}{'  ' + rincian if rincian else ''}",
          flush=True)


def trust_awal(ctx):
    with ctx:
        sim = common.fresh_sim(DATASET)
        return np.array([u.trust for u in sim.users], float)


def trust_sesudah_jalan(ctx, langkah=LANGKAH):
    with ctx:
        sim = common.fresh_sim(DATASET)
        random.seed(0); np.random.seed(0)
        awal = np.array([u.trust for u in sim.users], float)
        sim.run(max_steps=min(langkah, sim.max_steps), agent=GreedyAgent(mode="queue"))
        akhir = np.array([u.trust for u in sim.users], float)
        return awal, akhir


print("=" * 68)
print("E0 -- UJI ASAP PENGATURAN KEPERCAYAAN")
print("=" * 68)

# ---------------------------------------------------------------- 1. baku
print("\n1. Tanpa pengaturan (bawaan)")
t = trust_awal(contextlib.nullcontext())
baku = float(t.mean())
cek("simulasi terbentuk", t.size > 0, f"n={t.size} pengguna")
cek("kepercayaan bawaan di (0,1)", 0.0 < baku < 1.0, f"rata-rata={baku:.3f}")

# ------------------------------------------------- 2. initial_trust berpengaruh
print("\n2. --initial-trust: nilai awal berubah, dinamika TETAP jalan")
for v in (0.3, 0.5, 0.7):
    t = trust_awal(initial_trust(value=v))
    m = float(t.mean())
    cek(f"initial_trust({v}) -> awal = {v}", abs(m - v) < TOLERANSI,
        f"terukur={m:.4f}")

t3 = trust_awal(initial_trust(value=0.3))
t7 = trust_awal(initial_trust(value=0.7))
cek("0.3 dan 0.7 memang berbeda", abs(t3.mean() - t7.mean()) > 0.3,
    f"selisih={abs(t3.mean()-t7.mean()):.3f}")

awal, akhir = trust_sesudah_jalan(initial_trust(value=0.3))
bergerak = float(np.abs(akhir - awal).max())
cek("dinamika masih hidup (tidak dibekukan)", bergerak > 1e-9,
    f"pergeseran terbesar={bergerak:.4f}")

# ------------------------------------------------- 3. constant_trust membekukan
print("\n3. --constant-trust: nilai dibekukan, dinamika MATI")
awal, akhir = trust_sesudah_jalan(constant_trust(value=0.5))
beku = float(np.abs(akhir - awal).max())
cek("nilai awal = 0.5", abs(float(awal.mean()) - 0.5) < TOLERANSI,
    f"terukur={awal.mean():.4f}")
cek("kepercayaan TIDAK bergerak", beku < 1e-9, f"pergeseran terbesar={beku:.2e}")

# --------------------------------------- 4. shadow: nilai bergerak, keputusan beku
print("\n4. --constant-trust-shadow: nilai bergerak, keputusan memakai nilai beku")
awal, akhir = trust_sesudah_jalan(constant_trust_shadow(value=0.5))
gerak = float(np.abs(akhir - awal).max())
cek("kepercayaan TETAP bergerak (beda dari constant_trust)", gerak > 1e-9,
    f"pergeseran terbesar={gerak:.4f}")

# --------------------------------------------- 5. konteks benar dilepas kembali
print("\n5. Konteks dilepas bersih setelah selesai")
t = trust_awal(contextlib.nullcontext())
cek("kembali ke bawaan setelah semua konteks", abs(float(t.mean()) - baku) < TOLERANSI,
    f"terukur={t.mean():.4f} bawaan={baku:.4f}")

# ------------------------------------------------------------ 6. pipeline utuh
print("\n6. Pipeline bisa diurai (parse) dengan pengaturan baru")
import subprocess
r = subprocess.run([sys.executable, os.path.join(_HERE, "_pipeline_hipotesis.py"), "--help"],
                   capture_output=True, text=True)
teks = r.stdout + r.stderr
for flag in ("--initial-trust", "--constant-trust", "--constant-trust-shadow", "--tag"):
    cek(f"pipeline menerima {flag}", flag in teks)

print("\n" + "=" * 68)
print(f"LULUS {len(lulus)} / GAGAL {len(gagal)}")
if gagal:
    print("\nYang gagal:")
    for g in gagal:
        print(f"  - {g}")
    print("\nJANGAN lanjut ke E1 sebelum semua hijau.")
    sys.exit(1)
print("\nSemua hijau. E0 selesai -- lanjut ke E1.")
print("=" * 68)
