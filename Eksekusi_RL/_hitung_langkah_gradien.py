"""Hitung ULANG jumlah langkah gradien SESUNGGUHNYA dari training_results.json yang
SUDAH ADA (2026-08-31) -- menjawab apakah perbandingan DDPG vs PPO pada MASTER murni
setara secara anggaran komputasi, bukan cuma jumlah chunk lingkungan.

DDPG (`master_pure_trainer.py::_run_one_chunk`): per chunk melakukan
    min(n_new, updates_per_chunk=20)
langkah gradien terpisah (masing2 1 batch=32, off-policy REPLAY dari buffer).

PPO (`master_pure_ppo_trainer.py`): per chunk melakukan
    epochs(=10) x ceil(n_ready / minibatch(=32))
langkah gradien (on-policy, batch chunk itu SAJA, dibuang sesudahnya).

Kedua nilai TIDAK dicatat langsung, tapi bisa direkonstruksi dari `n_new`/`n_ready`
yang SUDAH tersimpan di `history` tiap baris `training_results.json` -- TIDAK PERLU
melatih ulang untuk mengukurnya.

Pemakaian:
    python Eksekusi_RL/_hitung_langkah_gradien.py <tag_ddpg> <tag_ppo>
Contoh:
    python Eksekusi_RL/_hitung_langkah_gradien.py \
        master_pure_dgr_90d_cwtfail120pen-2 master_pure_ppo_dgr_90d_cwtfail120pen-2
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

DDPG_UPDATES_PER_CHUNK = 20
PPO_EPOCHS = 10
PPO_MINIBATCH = 32


def muat(tag):
    path = os.path.join(common.OUTDIR, f"{tag}_training_results.json")
    assert os.path.exists(path), f"tak ditemukan: {path}"
    return json.load(open(path, encoding="utf-8"))


def langkah_ddpg(history):
    return sum(min(h.get("n_new", 0), DDPG_UPDATES_PER_CHUNK) for h in history)


def langkah_ppo(history):
    total = 0
    for h in history:
        n_ready = h.get("n_ready", 0)
        if n_ready <= 0:
            continue
        minibatch_per_epoch = -(-n_ready // PPO_MINIBATCH)  # ceil
        total += PPO_EPOCHS * minibatch_per_epoch
    return total


def main():
    tag_ddpg, tag_ppo = sys.argv[1], sys.argv[2]
    rows_ddpg = muat(tag_ddpg)
    rows_ppo = muat(tag_ppo)

    print(f"=== DDPG: {tag_ddpg} ===")
    tot_d = []
    tot_transisi_d = []
    for r in rows_ddpg:
        h = r["history"]
        s = langkah_ddpg(h)
        nt = sum(x.get("n_new", 0) for x in h)
        tot_d.append(s); tot_transisi_d.append(nt)
        print(f"  seed={r['seed']}: {len(h)} chunk tercatat, "
             f"{nt} transisi baru total, {s} langkah gradien (batch=32)")

    print(f"\n=== PPO: {tag_ppo} ===")
    tot_p = []
    tot_transisi_p = []
    for r in rows_ppo:
        h = r["history"]
        s = langkah_ppo(h)
        nt = sum(x.get("n_ready", 0) for x in h)
        tot_p.append(s); tot_transisi_p.append(nt)
        print(f"  seed={r['seed']}: {len(h)} chunk tercatat, "
             f"{nt} transisi diproses total, {s} langkah gradien (batch=32)")

    md, mp = sum(tot_d) / len(tot_d), sum(tot_p) / len(tot_p)
    ntd, ntp = sum(tot_transisi_d) / len(tot_transisi_d), sum(tot_transisi_p) / len(tot_transisi_p)
    print(f"\n=== RINGKASAN (rerata 3 seed) ===")
    print(f"  langkah gradien   : DDPG={md:.0f}   PPO={mp:.0f}   rasio PPO/DDPG={mp/md:.2f}x")
    print(f"  transisi diproses : DDPG={ntd:.0f}   PPO={ntp:.0f}   rasio PPO/DDPG={ntp/ntd:.2f}x")
    print()
    if mp > md * 1.2:
        faktor = mp / md
        print(f"PPO menerima ~{faktor:.1f}x lebih banyak langkah gradien drpd DDPG.")
        print(f"Utk menyetarakan: naikkan `updates_per_chunk` DDPG dari {DDPG_UPDATES_PER_CHUNK} "
             f"ke ~{DDPG_UPDATES_PER_CHUNK*faktor:.0f} (bulatkan), ATAU turunkan `n_updates` "
             f"(jumlah chunk) PPO dgn faktor yg sama.")
    elif md > mp * 1.2:
        faktor = md / mp
        print(f"DDPG menerima ~{faktor:.1f}x lebih banyak langkah gradien drpd PPO.")
        print(f"Utk menyetarakan: turunkan `updates_per_chunk` DDPG, ATAU naikkan `epochs`/"
             f"kecilkan `minibatch` PPO dgn faktor serupa.")
    else:
        print("Anggaran gradien SUDAH cukup setara (selisih <20%) -- tak perlu penyesuaian.")


if __name__ == "__main__":
    main()
