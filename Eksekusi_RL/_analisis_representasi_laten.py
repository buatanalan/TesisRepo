"""Analisis representasi laten (Uji A, bagian 2) -- baca `representasi_*.npz` hasil
`_ekstrak_representasi_laten.py`, uji dua klaim:

  (1) Apakah pengguna dgn preferensi (sid_pref) SERUPA berkumpul dlm ruang h_n?
      -> classifier k-NN sederhana: tebak sid_pref dari h_n saja, bandingkan thd
         akurasi kebetulan (1/6, enam stasiun) DAN thd akurasi decoding LANGSUNG dari
         keputusan sistem (acc/pref_dalam, sbg batas atas informasi yg tersedia).
      -> silhouette score PCA-2D by sid_pref -- >0 berarti klaster nyata terpisah,
         mendekati 0 berarti representasi tak terstruktur oleh label ini.

  (2) Apakah pengguna trust TINGGI dan RENDAH terpisah dlm ruang h_n?
      -> regresi linear h_n -> trust, laporkan R^2 (bukan cuma korelasi 1 dimensi,
         krn pemisahan bisa terjadi pd kombinasi linear arah mana pun di ruang 8-dim).

Jalankan UNTUK KEDUA lengan lalu bandingkan -- P-MASTER menunjukkan struktur signifikan
sementara MASTER (gate~0, "mati") TIDAK adalah bukti struktur berasal dari pelatihan
modul aktif, bukan artefak LSTM semata (lih. catatan di `_ekstrak_representasi_laten.py`).

Pemakaian:
    python _analisis_representasi_laten.py master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3 master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, common
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA

ARMS = sys.argv[1:] or [
    "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_pg0.1_pure3",
    "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout_noattn_pure3",
]


def muat(tag_arm):
    f = os.path.join(common.OUTDIR, f"representasi_{tag_arm}.npz")
    d = np.load(f)
    return d["h"], d["sid_pref"], d["trust"], d["user_id"], d["seed"]


def uji_preferensi(h, sid_pref, uid):
    """k-NN k=15 memakai GroupKFold by user_id -- MENCEGAH kebocoran (satu pengguna
    bisa punya banyak keputusan dgn h_n hampir identik krn riwayatnya belum banyak
    berubah; split acak akan menaruh salinan hampir sama di train & test, menggelembung-
    kan akurasi secara palsu). Kebetulan = 1/jumlah_stasiun_unik."""
    n_kelas = len(set(sid_pref))
    kebetulan = 1.0 / n_kelas
    gkf = GroupKFold(n_splits=5)
    knn = KNeighborsClassifier(n_neighbors=15)
    skor = cross_val_score(knn, h, sid_pref, groups=uid, cv=gkf, scoring="accuracy")
    # Silhouette pd PCA-2D (utk visualisasi & ukuran keterpisahan tunggal)
    h2 = PCA(n_components=2, random_state=0).fit_transform(h)
    sil = silhouette_score(h2, sid_pref) if n_kelas > 1 else float("nan")
    return dict(akurasi_knn=float(skor.mean()), akurasi_knn_sd=float(skor.std()),
               kebetulan=kebetulan, silhouette_pca2=float(sil))


def uji_trust(h, trust, uid):
    """Regresi linear (Ridge, alpha kecil utk stabilitas numerik dim=8) h_n -> trust,
    R^2 out-of-fold via GroupKFold (sama alasan anti-kebocoran spt di atas)."""
    gkf = GroupKFold(n_splits=5)
    model = Ridge(alpha=1.0)
    skor = cross_val_score(model, h, trust, groups=uid, cv=gkf, scoring="r2")
    return dict(r2_knn=float(skor.mean()), r2_sd=float(skor.std()))


def main():
    print("UJI A -- representasi laten pref_lstm (2026-09-01)")
    print(f"{'lengan':55s}{'n':>8s}{'akurasi pref':>14s}{'kebetulan':>11s}"
         f"{'silhouette':>12s}{'R2(trust)':>11s}")
    for arm in ARMS:
        h, sid_pref, trust, uid, seed = muat(arm)
        rp = uji_preferensi(h, sid_pref, uid)
        rt = uji_trust(h, trust, uid)
        print(f"{arm:55s}{len(h):8,d}{rp['akurasi_knn']:13.3f}±{rp['akurasi_knn_sd']:.3f}"
             f"{rp['kebetulan']:11.3f}{rp['silhouette_pca2']:12.4f}{rt['r2_knn']:11.4f}")
        print(f"   -> akurasi PREF {'DI ATAS' if rp['akurasi_knn'] > rp['kebetulan']+2*rp['akurasi_knn_sd'] else 'TIDAK terpisah dari'} "
             f"kebetulan; R^2(trust) {'bermakna (>0.1)' if rt['r2_knn']>0.1 else 'lemah/nol'}")
    print()
    print("Pembacaan: bila lengan P-MASTER (gate aktif) menunjukkan akurasi PREF di atas")
    print("kebetulan DAN/ATAU R2(trust) bermakna, sementara MASTER (gate~0) TIDAK --")
    print("itu representasi laten TERSTRUKTUR akibat pelatihan, bukan artefak LSTM semata.")
    print("Bila KEDUANYA gagal pd sisi preferensi -- konsisten dgn temuan §5.7 Bab V")
    print("(pref_dalam/pref_primer tak terpisah dari agen buta-user) pd tingkat REPRESENTASI,")
    print("bukan cuma tingkat KELUARAN -- argumen 'cuma LSTM' makin sulit dibantah utk sisi itu.")


if __name__ == "__main__":
    main()
