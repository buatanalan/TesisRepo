"""Eksekusi Tahap 2 v3 (pasca perbaikan bug penyemaian) -- lihat
`Dokumen_Penting/Rencana_Tahap2_v3_pasca_bug_seeding.md`.

Menjalankan satu KELOMPOK lengan × beberapa seed, menyimpan checkpoint + meta + log JSONL.
Dipanggil per-tahap supaya bisa dijalankan bertahap di latar:

    python _tahap2_jalankan.py A     # baseline bersih H-PPO K=1
    python _tahap2_jalankan.py B     # kritik ganda: K=2 fixed & K=2 gap_ratio
    python _tahap2_jalankan.py C     # metode usulan: P-PPO gated (+ablasi)

Semua lengan memakai konfigurasi IDENTIK kecuali yang sedang diteliti:
trust statis 0,5, reward `gabungan`, 300 iterasi, rollout_steps=288, k=2.
"""
import sys, os, time, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common
from marl_spklu.rl.forecaster import ForecasterBase
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.p_ppo_policy import PPPOPolicy
from marl_spklu.experiments.ablations import constant_trust_shadow

SEEDS = [0, 1, 2]
N_UPDATES = 200
ROLLOUT_STEPS = 288
TRUST = 0.5
DATASET = "scenario_dataset_klaster12_4x.json"

# TAHAP 3 = trust DINAMIS (performativitas hidup). Tahap 2 membekukannya sbg KONTROL.
# Nama tahap yang terdaftar di sini dijalankan TANPA `constant_trust_shadow`, sehingga
# `update_trust` benar-benar memengaruhi keputusan berikutnya (loop performatif tertutup).
TAHAP_TRUST_DINAMIS = {"T3"}

# Preset reward -- IDENTIK di seluruh lengan (bukan variabel yang diteliti).
#
# DIGANTI 2026-08-16 dari `gabungan` (alpha_wait=0, beta_prox=0.4467, alpha_gini=0.3672,
# alpha_flock=0.0795) ke `seimbang4x`. Alasan: dekomposisi varians pada rezim 4x
# menunjukkan `gabungan` punya varians-individual hanya 2,6% DAN kontribusi suku gini di
# kanal global hanya 0,1% -- objektif pemerataan praktis tak menghasilkan gradien sama
# sekali. Preset lama (`seimbang`) juga 0,1% krn dikalibrasi pada rezim BERBEDA sebelum
# Tahap 1 membekukan 4x. Lihat rewards.py::seimbang4x & _kalibrasi_reward_4x.py.
REWARD_KW = dict(alpha_wait=0.0046, beta_prox=0.1, alpha_gini=2.6019,
                 alpha_flock=0.0208, use_delta_gini=True)


class VirtualWaitForecaster(ForecasterBase):
    """Prediktor EstWait SERAGAM lintas semua metode (H-PPO/P-PPO/PDQN/greedy) -- syarat
    perbandingan setara: satu tujuan tak boleh dikerjakan mekanisme berbeda."""
    def predict(self, spklus, time_now_min=0.0, user=None, soc=50.0, sim=None):
        if sim is None or user is None:
            return {sid: 0.0 for sid in spklus}
        return {sid: float(sim.compute_virtual_wait(user, s, time_now_min))
                for sid, s in spklus.items()}


def _trainer_cls(policy_cls):
    if policy_cls is None:
        return TorchContinuingTrainer
    return type("T_" + policy_cls.__name__, (TorchContinuingTrainer,),
                {"POLICY_CLS": policy_cls})


def jalankan(nama, seed, policy_cls=None, policy_kw=None, trust_dinamis=False, **trainer_kw):
    """Satu run. `nama` dipakai sbg stem berkas keluaran."""
    stem = f"t2_{nama}_seed{seed}"
    ck = os.path.join(common.OUTDIR, stem + ".pt")
    if os.path.exists(ck):
        print(f"[LEWATI] {stem} sudah ada", flush=True)
        return
    ds = os.path.join(common.ROOT, DATASET)
    rc = RewardCalculator(**REWARD_KW)
    log_path = os.path.join(common.OUTDIR, stem + ".jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)          # jangan menumpuk log run sebelumnya

    t0 = time.time()
    ctx = (contextlib.nullcontext() if trust_dinamis
           else constant_trust_shadow(value=TRUST))
    with ctx:
        Trainer = _trainer_cls(policy_cls)
        tr = Trainer(ds, k=2, rollout_steps=ROLLOUT_STEPS, reward_calc=rc, seed=seed,
                     verbose=True, log_path=log_path, **trainer_kw)
        # policy_kw (mis. use_preference=False utk ablasi) diterapkan SETELAH konstruksi
        # supaya tak perlu mengubah signature trainer.
        if policy_kw:
            for k, v in policy_kw.items():
                setattr(tr.policy, k, v)
        policy = tr.train(VirtualWaitForecaster(), n_updates=N_UPDATES)

    torch.save(policy.state_dict(), ck)
    meta = dict(nama=nama, seed=seed, obs_dim=tr.obs_dim, critic_obs_dim=tr.critic_obs_dim,
                N=tr.N, k=tr.k, n_critics=getattr(tr.policy, "n_critics", 1),
                trainer_kw={k: v for k, v in trainer_kw.items()},
                policy_cls=(policy_cls.__name__ if policy_cls else "HPPOPolicy"),
                policy_kw=(policy_kw or {}), reward=REWARD_KW,
                trust=(None if trust_dinamis else TRUST), trust_dinamis=trust_dinamis,
                elapsed_s=time.time() - t0, history=tr.history)
    if hasattr(policy, "pref_gate"):
        meta["final_gate"] = float(policy.pref_gate.item())
        # Dimensi modul preferensi WAJIB dicatat: nilainya pernah diubah (64 -> 16), jadi
        # tanpa ini checkpoint lama tak bisa dimuat ulang (bentuk bobot tak cocok).
        meta["pref_d_lstm"] = int(policy.pref_lstm.hidden_size)
        meta["pref_d_attn"] = int(policy.pref_attn.d_attn)
    common.save_json(meta, stem + "_meta.json")
    print(f"[SELESAI] {stem} ({time.time()-t0:.0f}s)", flush=True)


TAHAP = {
    # A -- baseline bersih: adaptasi MASTER dgn kritik TUNGGAL (prior art cabang pemerataan)
    #      Sufiks `sb4x` menandai preset reward `seimbang4x` -- WAJIB dibedakan dari
    #      checkpoint `hppo_K1_*` lama yang memakai `gabungan` (kanal individual mati,
    #      gini dekoratif); keduanya TIDAK sebanding.
    "A": [("hppo_K1_sb4x", None, None, dict(n_critics=1))],
    # Afix -- SAMA PERSIS dgn A, dijalankan SETELAH perbaikan bug `_prev_gini` tak di-reset
    #      di batas horizon (rewards.py::reset_episode_state + ppo.py::_reset_rc).
    #      Sufiks `fixgini` WAJIB: lengan `hppo_K1_sb4x` dilatih SEBELUM perbaikan, jadi
    #      keduanya TIDAK sebanding. Menguji apakah kolaps entropi hilang.
    "Afix": [("hppo_K1_sb4x_fixgini", None, None, dict(n_critics=1))],
    # Cfix -- ablasi modul preferensi pada KODE FINAL (pasca perbaikan delta-Gini).
    #      Memakai d=16 (kapasitas sepadan H-PPO) supaya selisih hasil bisa diatribusikan
    #      ke INFORMASI preferensi, bukan ke kapasitas parameter. Varian d=64 (setia nilai
    #      paper Lin dkk.) sudah diuji lebih awal pada kode pra-perbaikan.
    "Cfix": [("pppo_sb4x_d16_fixgini", PPPOPolicy, None, dict(n_critics=1))],
    # Abnd -- UJI perbaikan batas horizon: transisi tak-resolved DIBUANG (bukan dipaksa
    #      masuk update dgn reward tak lengkap). Prediksi yang diuji: approx_kl di
    #      mod10=9 turun ke level normal (~0,04 dari 0,156), dan onset kolaps di mod10=1
    #      berkurang drastis (sebelumnya 20 dari 33).
    "Abnd": [("hppo_K1_sb4x_bnd", None, None, dict(n_critics=1))],
    # Cbnd -- P-PPO pada KODE FINAL yang sama dgn `Abnd` (perbaikan batas horizon +
    #      delta-Gini + penyemaian + presisi). Satu-satunya beda thd `hppo_K1_sb4x_bnd`
    #      adalah modul preferensi -> selisihnya bisa diatribusikan.
    #      d=16 (kapasitas sepadan H-PPO); varian d=64 setia-paper sudah diuji lebih awal.
    #      CATATAN: greedy & S0 TIDAK perlu diulang -- keduanya tanpa training, jadi tak
    #      tersentuh perbaikan apa pun di jalur training.
    "Cbnd": [("pppo_sb4x_d16_bnd", PPPOPolicy, None, dict(n_critics=1))],
    # ================= TAHAP 3 -- TRUST DINAMIS (kontribusi inti) =================
    # Lengan IDENTIK dgn kontrol Tahap 2 (`*_bnd`), satu-satunya beda: trust TIDAK
    # dibekukan. Perbandingan Tahap2 vs Tahap3 pada lengan yang sama inilah yang menguji
    # apakah performativitas mengubah perilaku/peringkat metode.
    "T3": [("hppo_t3", None, None, dict(n_critics=1)),
           ("pppo_t3", PPPOPolicy, None, dict(n_critics=1))],
    # B -- adopsi kritik ganda MASTER (M1: pemisahan saja; M2: + bobot adaptif)
    "B": [("hppo_K2_fixed", None, None, dict(n_critics=2, beta_mode="fixed")),
          ("hppo_K2_gap",   None, None, dict(n_critics=2, beta_mode="gap_ratio"))],
    # C -- METODE USULAN (modul P dari PDQN, digating gaya GTrXL) + ablasi sinyal-preferensi.
    #      Memakai reward `seimbang4x` (sufiks sb4x) & n_critics=1, IDENTIK dgn baseline
    #      `hppo_K1_sb4x` -- satu-satunya beda adalah modul preferensi, sehingga selisihnya
    #      bisa diatribusikan. Kritik ganda (Tahap B) sengaja DILEWATI: ia perbaikan, bukan
    #      bagian formulasi.
    #      `nopref` = arsitektur & jumlah parameter SAMA, sinyal preferensi dimatikan --
    #      memisahkan kontribusi INFORMASI preferensi dari KAPASITAS parameter tambahan.
    #      DIPISAH jadi C dan C2: ablasi `nopref` nilainya terutama sbg KONTROL utk
    #      Tahap 3 (agar lengan yang sama terukur di kondisi kontrol & perlakuan), bukan
    #      utk Tahap 2 sendiri -- jadi bisa ditunda tanpa merugikan analisis Tahap 2.
    "C": [("pppo_sb4x", PPPOPolicy, None, dict(n_critics=1))],
    "C2": [("pppo_sb4x_nopref", PPPOPolicy, dict(use_preference=False), dict(n_critics=1))],
    # Cd16 -- P-PPO dgn modul preferensi DIKECILKAN (d_lstm=d_attn=16, disamakan dgn
    #      HIST_HIDDEN H-PPO). Bersama lengan `pppo_sb4x` (d=64, sudah selesai) dan
    #      `hppo_K1_sb4x` (tanpa modul), tiga titik ini MEMISAHKAN kontribusi INFORMASI
    #      preferensi dari kontribusi KAPASITAS parameter:
    #        H-PPO      : tanpa informasi, tanpa kapasitas tambahan
    #        P-PPO d=16 : dgn informasi, kapasitas +3.329
    #        P-PPO d=64 : dgn informasi, kapasitas +28.673
    #      Kalau d=16 ~ d=64 -> yang menentukan INFORMASI, bukan kapasitas.
    #      CATATAN: bergantung pada default PREF_D_LSTM/PREF_D_ATTN=16 di
    #      p_ppo_policy.py -- kalau default itu diubah lagi, lengan ini ikut berubah.
    "Cd16": [("pppo_sb4x_d16", PPPOPolicy, None, dict(n_critics=1))],
    # K -- KALIBRASI ent_coef (disisipkan sebelum Tahap B). Tahap A menunjukkan 1 dari 3
    #      seed KOLAPS jadi penumpukan satu stasiun (served [1377,238,...], tunggu 622 mnt).
    #      Polanya persis risiko yang dicatat Rumusan_Masalah_Teknis_RL.md §3.3: ent_coef
    #      rendah -> kolaps entropi -> herding -> Gini memburuk SELAMA training. Kalibrasi
    #      sebelumnya (memilih 0,01) dikerjakan di bawah bug non-determinisme & 1 seed,
    #      jadi tak lagi sah.
    #      ent_coef=0,01 TIDAK diulang di sini -- lengan `hppo_K1` Tahap A SUDAH memakai
    #      nilai itu (default PPOTrainer), jadi dipakai langsung sbg pembanding.
    #      Catatan §3.3: nilai historis 0,3 ditala pada ruang aksi LAMA (1 stasiun);
    #      ruang aksi kini lebih besar -> 0,05 disertakan sbg titik tengah.
    #      KRITERIA PEMILIHAN: tidak ada seed yang kolaps (BUKAN Gini terbaik -- supaya
    #      tak sirkular, sesuai prinsip Bagian 1 Laporan_Tahap2).
    #      DIPERBARUI 2026-08-16 pasca-`seimbang4x`: hipotesis "kolaps disebabkan
    #      spesifikasi reward" DITOLAK -- setelah reward diperbaiki, kolaps TETAP 1/3,
    #      hanya PINDAH seed (dulu seed1, kini seed2). Jadi kolaps bersifat stokastik dan
    #      terpisah dari spesifikasi reward -> kalibrasi entropi kini beralasan sendiri.
    #      Pembanding (ent_coef=0,01 default) = lengan `hppo_K1_sb4x` Tahap A.
    #      Dijalankan 0,05 DULU (versi minimal); 0,30 hanya bila 0,05 tak cukup.
    "K": [("hppo_sb4x_ent0.05", None, None, dict(n_critics=1, ent_coef=0.05))],
    "K2": [("hppo_sb4x_ent0.30", None, None, dict(n_critics=1, ent_coef=0.30))],
}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "A"
    # Pencocokan TAK PEKA huruf besar-kecil: `.upper()` polos sempat memecahkan kunci
    # bercampur huruf spt "Cd16" (jadi "CD16" -> tak ditemukan).
    cocok = {k.upper(): k for k in TAHAP}
    if arg.upper() not in cocok:
        raise SystemExit(f"tahap tak dikenal: {arg} (pilih {list(TAHAP)})")
    tahap = cocok[arg.upper()]
    daftar = TAHAP[tahap]
    # Argumen KEDUA (opsional) menimpa daftar seed, spy seed tambahan bisa dijalankan di
    # server tanpa mengedit berkas:  python _tahap2_jalankan.py T3 3,4
    seeds = ([int(s) for s in sys.argv[2].replace(" ", "").split(",") if s]
             if len(sys.argv) > 2 else SEEDS)
    print(f"=== TAHAP {tahap}: {len(daftar)} lengan x {len(seeds)} seed {seeds} ===",
          flush=True)
    for nama, pcls, pkw, tkw in daftar:
        for sd in seeds:
            jalankan(nama, sd, policy_cls=pcls, policy_kw=pkw,
                     trust_dinamis=(tahap in TAHAP_TRUST_DINAMIS), **tkw)
    print(f"=== TAHAP {tahap} SELESAI ===", flush=True)
