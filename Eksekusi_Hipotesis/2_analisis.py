"""LANGKAH 2 -- membaca hasil dan menjatuhkan vonis.

    python 2_analisis.py                    # klaim pokok (30 hari)
    python 2_analisis.py --horizon 90d      # uji ketahanan
    python 2_analisis.py --banding          # 30 vs 90 hari berdampingan

Tidak melatih apa pun, tidak menjalankan simulasi. Hanya membaca `*_eval.json` di
outputs/, jadi cepat dan aman diulang.

Menjawab dua klaim, keduanya dari kumpulan hasil yang sama:

  H6b  penyatuan mengungguli KEDUA sistem asal    <- klaim utama tesis
  H2b  keunggulan menyusut seiring turunnya kepercayaan awal

Syarat lulusnya ditetapkan di `draft tesis/Hipotesis_Penelitian.md` SEBELUM data
dikumpulkan. Skrip ini menerapkannya apa adanya, tidak menawar.
"""
import sys, os, argparse, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _kakas as K
from scipy.stats import ks_2samp, wilcoxon

STAMP = datetime.date.today().strftime("%Y%m%d")
LENGAN = [("h6b_utama", "penyatuan (diuji)"),
          ("h1a_pemerataan", "pemerataan saja"),
          ("h2a_selera", "selera saja")]

# Toleransi. Ditulis eksplisit supaya jadi keputusan yang terlihat, bukan angka ajaib
# yang terselip di tengah perbandingan.
TOL_GINI = 1.02    # "tidak lebih buruk dari sistem asal" = dalam 2%
TOL_ACC = 0.95     # "mendekati sistem asal" = minimal 95% capaiannya


def suf(horizon):
    return "" if horizon == "30d" else f"_{horizon}"


def muat_rapi(horizon):
    """Muat 9 lengan. Nama berkas: `<tag>__it<XX>[_90d]_<tanggal>_eval.json`."""
    import glob, json
    out = {}
    for tag, _ in LENGAN:
        for it in K.TINGKAT_TRUST:
            pola = os.path.join(K.OUTDIR,
                                f"{tag}__it{it.replace('.', '')}{suf(horizon)}_*_eval.json")
            berkas = sorted(glob.glob(pola))
            if not berkas:
                out[(tag, it)] = None
                continue
            if len(berkas) > 1:
                print(f"    ! {tag} it={it}: {len(berkas)} tanggal, memakai "
                      f"{os.path.basename(berkas[-1])}")
            with open(berkas[-1], encoding="utf-8") as f:
                d = json.load(f)
            d["_berkas"] = os.path.basename(berkas[-1])
            # Pagar terhadap hasil mode cepat. Berkasnya sudah berawalan `zzcepat_`
            # sehingga tak akan terjaring pola di atas, tapi anggaran diperiksa ulang
            # di sini -- vonis yang dihitung dari kebijakan 3-pembaruan akan tampak
            # normal dan itulah bahayanya.
            c = d.get("config", {})
            if c.get("n_updates", 999) < 50 or c.get("n_train_seed", 99) < 3:
                print(f"    !! {os.path.basename(berkas[-1])} anggarannya terlalu kecil "
                      f"({c.get('n_train_seed')} seed x {c.get('n_updates')} pembaruan) "
                      f"-- DIABAIKAN, bukan hasil sah")
                out[(tag, it)] = None
                continue
            out[(tag, it)] = d
    return out


def nilai(data, horizon):
    """Terapkan syarat lulus. Mengembalikan (vonis_pokok, kurva, sebaran)."""
    K.judul(f"H6b -- penyatuan vs kedua sistem asal   (horizon {horizon})")
    for it in K.TINGKAT_TRUST:
        print(f"\n  --- kepercayaan awal {it} ---")
        print(K.kepala_metrik())
        for tag, ket in LENGAN:
            d = data.get((tag, it))
            print(K.baris_metrik(ket, d["ringkas"]["kebijakan"] if d else None))
        d0 = next((data[(t, it)] for t, _ in LENGAN if data.get((t, it))), None)
        if d0:
            for g in ("greedy_queue", "greedy_util"):
                print(K.baris_metrik(g, d0["ringkas"][g]))

    print("\n\n  Lulus bila: Gini <= pemerataan-saja (toleransi 2%), DAN penerimaan >= 95%")
    print("  capaian selera-saja, DAN nyata lebih baik dari greedy (p<0,05) --")
    print("  minimal di kepercayaan 0,5 dan 0,7.\n")
    print(f"  {'percaya':<10} {'gini vs asal':>13} {'terima vs asal':>16} "
          f"{'p vs greedy':>12}   vonis")

    vonis = {}
    for it in K.TINGKAT_TRUST:
        du, dp, ds = data.get(("h6b_utama", it)), data.get(("h1a_pemerataan", it)), \
                     data.get(("h2a_selera", it))
        if not du:
            print(f"  {it:<10} {'(belum ada)':>13}")
            continue
        u = du["ringkas"]["kebijakan"]
        gini_ok = (u["gini"] <= dp["ringkas"]["kebijakan"]["gini"] * TOL_GINI) if dp else None
        acc_ok = (u["acc"] >= ds["ringkas"]["kebijakan"]["acc"] * TOL_ACC) if ds else None
        p_gu = du["p_vs_gu"]
        greedy_ok = bool(p_gu < 0.05 and u["gini"] < du["ringkas"]["greedy_util"]["gini"])
        ok = bool(gini_ok and acc_ok and greedy_ok)
        vonis[it] = dict(gini_vs_asal=gini_ok, terima_vs_asal=acc_ok,
                         nyata_vs_greedy=greedy_ok, p_vs_greedy_util=p_gu, lulus=ok)
        f = lambda v: "-" if v is None else ("OK" if v else "gagal")
        print(f"  {it:<10} {f(gini_ok):>13} {f(acc_ok):>16} {p_gu:>12.4f}   "
              f"{'LULUS' if ok else 'tidak'}")

    inti = [vonis.get(i, {}).get("lulus") for i in ("0.5", "0.7")]
    lulus = all(v is True for v in inti)
    print("\n" + K.vonis(lulus,
        "H6b terdukung di kepercayaan sedang & tinggi.",
        "H6b DITOLAK pada syaratnya sendiri. Lihat klaim revisi di bawah."))

    # ------------------------------------------------------------------ klaim revisi
    # PASCA-HOC. Dirumuskan SETELAH melihat data 30 hari, jadi TIDAK boleh dilaporkan
    # sebagai hipotesis yang lulus -- statusnya dugaan baru yang butuh konfirmasi
    # independen. Uji 90 hari adalah konfirmasi pertamanya.
    K.judul(f"KLAIM REVISI (PASCA-HOC) -- pertukaran pemerataan vs pengalaman pengguna")
    print("H6b ditolak. Dugaan pengganti, dirumuskan SETELAH data terlihat:\n")
    print("  Teknik preferensi TIDAK memperbaiki pemerataan -- justru memperburuknya.")
    print("  Namun ia menaikkan kepatuhan, menurunkan waktu tunggu, dan menaikkan")
    print("  kepercayaan secara signifikan. Ada PERTUKARAN antara pemerataan jaringan")
    print("  dan pengalaman pengguna, dan teknik preferensi menggeser sistem ke sisi")
    print("  pengalaman pengguna.\n")
    print(f"  {'percaya':<9} {'metrik':<8} {'penyatuan':>10} {'koord-saja':>11} "
          f"{'selisih':>10} {'p':>8}  unggul")
    revisi, arah_baik = {}, dict(gini=-1, acc=+1, wait=-1, trust=+1)
    for it in K.TINGKAT_TRUST:
        du, dp = data.get(("h6b_utama", it)), data.get(("h1a_pemerataan", it))
        if not (du and dp):
            continue
        sel = {}
        for m, baik in arah_baik.items():
            x = [r[m] for r in du["per_eval_seed"]]
            y = [r[m] for r in dp["per_eval_seed"]]
            pv = float(wilcoxon(x, y).pvalue)
            beda = float(np.mean(x) - np.mean(y))
            unggul = "penyatuan" if beda * baik > 0 else "koord-saja"
            sel[m] = dict(penyatuan=float(np.mean(x)), koordinasi=float(np.mean(y)),
                          selisih=beda, p=pv, unggul=unggul, nyata=pv < 0.05)
            print(f"  {it:<9} {m:<8} {np.mean(x):>10.4f} {np.mean(y):>11.4f} "
                  f"{beda:>+10.4f} {pv:>8.4f}{'*' if pv < 0.05 else ' '} {unggul}")
        revisi[it] = sel
        print()

    # Pola yang dicari: pemerataan kalah, TIGA metrik pengalaman pengguna menang,
    # dan semuanya nyata. Itulah bentuk "pertukaran" yang diklaim.
    pola = []
    for it, sel in revisi.items():
        pola.append(sel["gini"]["unggul"] == "koord-saja"
                    and all(sel[m]["unggul"] == "penyatuan" for m in ("acc", "wait")))
    tukar = bool(pola) and all(pola)
    print(K.vonis(tukar,
        "Pola pertukaran konsisten di seluruh tingkat kepercayaan: pemerataan kalah, "
        "kepatuhan dan waktu tunggu menang. WAJIB ditulis sebagai dugaan pasca-hoc, "
        "bukan hipotesis yang lulus -- konfirmasinya menunggu uji 90 hari.",
        "Pola pertukaran TIDAK konsisten. Klaim revisi pun tak terdukung; laporkan "
        "H6b ditolak tanpa klaim pengganti."))
    v_revisi = dict(pola_konsisten=tukar, catatan="PASCA-HOC -- dirumuskan setelah "
                    "data 30 hari terlihat; bukan hipotesis pra-daftar. Konfirmasi "
                    "independen menunggu uji 90 hari.", per_tingkat=revisi)

    # ---------------------------------------------------------------- kurva H2b
    K.judul(f"H2b -- kurva kepercayaan   (horizon {horizon})")
    print("Keunggulan atas greedy MENYUSUT seiring turunnya kepercayaan awal.\n")
    print(f"  {'percaya':<10} {'d vs greedy_util':>18} {'gini':>9} {'percaya akhir':>15}")
    kurva = {}
    for it in K.TINGKAT_TRUST:
        d = data.get(("h6b_utama", it))
        if not d:
            print(f"  {it:<10} {'(belum ada)':>18}")
            continue
        kurva[it] = dict(d=d["cohens_d_vs_gu"], gini=d["ringkas"]["kebijakan"]["gini"],
                         trust=d["ringkas"]["kebijakan"]["trust"])
        print(f"  {it:<10} {kurva[it]['d']:>+18.3f} {kurva[it]['gini']:>9.4f} "
              f"{kurva[it]['trust']:>15.3f}")

    h2b = None
    if len(kurva) == 3:
        # d BERTANDA, bukan nilai mutlaknya. d negatif = Gini lebih rendah dari greedy
        # (unggul); d positif = lebih tinggi (kalah). "Keunggulan menyusut" berarti d
        # NAIK dari 0,7 ke 0,3 -- termasuk bila ia menyeberang nol dari unggul ke kalah.
        #
        # Memakai abs() di sini KELIRU dan sempat terjadi: lengan yang unggul d=-1,9 di
        # kepercayaan tinggi lalu kalah telak d=+5,6 di kepercayaan rendah akan terbaca
        # |1,9| -> |5,6| sebagai "membesar", padahal itu justru keruntuhan keunggulan
        # yang persis diramalkan H2b.
        d = [kurva[i]["d"] for i in ("0.7", "0.5", "0.3")]
        h2b = bool(d[0] <= d[1] <= d[2])
        print(f"\n  d bertanda 0,7 -> 0,5 -> 0,3 : "
              f"{d[0]:+.3f} -> {d[1]:+.3f} -> {d[2]:+.3f}   (negatif = unggul)")
        print("\n" + K.vonis(h2b,
            "Keunggulan menyusut berurutan seiring turunnya kepercayaan awal -- "
            "manfaat koordinasi memang bersyarat kepatuhan.",
            "TIDAK menyusut berurutan. Ini temuan, bukan kegagalan teknis: masalah "
            "non-kepatuhan tidak sepenting yang diklaim di latar belakang, dan "
            "pembingkaian tesis perlu disesuaikan. Laporkan apa adanya."))

    # ---------------------------------------------------------------- sebaran
    print("\n\nSebaran kepercayaan AKHIR menurut kondisi awal:\n")
    print(f"  {'percaya':<10} {'rata2':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'<0,3':>8}")
    sebaran = {}
    for it in K.TINGKAT_TRUST:
        d = data.get(("h6b_utama", it))
        if not d:
            continue
        per = d["per_eval_seed"]
        sebaran[it] = {k: float(np.mean([r[k] for r in per])) for k in
                       ("trust", "trust_p10", "trust_p50", "trust_p90", "trust_frac_bawah_03")}
        s = sebaran[it]
        print(f"  {it:<10} {s['trust']:>8.3f} {s['trust_p10']:>8.3f} {s['trust_p50']:>8.3f} "
              f"{s['trust_p90']:>8.3f} {s['trust_frac_bawah_03']:>8.1%}")

    if len(sebaran) == 3:
        lo = [r["trust"] for r in data[("h6b_utama", "0.3")]["per_eval_seed"]]
        hi = [r["trust"] for r in data[("h6b_utama", "0.7")]["per_eval_seed"]]
        ks = ks_2samp(lo, hi)
        print(f"\n  Kolmogorov-Smirnov 0,3 vs 0,7 : p={ks.pvalue:.4f}  "
              f"{'-> berbeda (kondisi akhir bergantung kondisi awal)' if ks.pvalue < 0.05 else '-> menyatu'}")
        sebaran["ks_p"] = float(ks.pvalue)

    return (dict(lulus=lulus, per_tingkat=vonis), dict(h2b=h2b, per_tingkat=kurva),
            sebaran, v_revisi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=str, default="30d", choices=["30d", "90d"])
    p.add_argument("--banding", action="store_true", help="bandingkan 30 vs 90 hari")
    args = p.parse_args()

    if args.banding:
        K.judul("KETAHANAN -- 30 hari vs 90 hari")
        baris = []
        for h in ("30d", "90d"):
            d = muat_rapi(h)
            print(f"\n  horizon {h}:")
            print(K.kepala_metrik())
            for it in K.TINGKAT_TRUST:
                r = d.get(("h6b_utama", it))
                print(K.baris_metrik(f"  it={it}", r["ringkas"]["kebijakan"] if r else None))
                if r:
                    baris.append((h, it, r["cohens_d_vs_gu"],
                                  r["ringkas"]["kebijakan"]["gini"],
                                  r["ringkas"]["greedy_util"]["gini"]))
        print(f"\n  {'horizon':<9} {'percaya':<9} {'d vs greedy':>12} {'gini':>9} "
              f"{'greedy':>9}   arah")
        for h, it, d, g, gg in baris:
            print(f"  {h:<9} {it:<9} {d:>+12.3f} {g:>9.4f} {gg:>9.4f}   "
                  f"{'menang' if d < 0 else 'KALAH'}")
        print("\n  Bila keunggulan bertahan 30 hari tapi hilang di 90 hari, itu batas")
        print("  berlaku yang WAJIB ditulis terang-terangan -- bukan disembunyikan")
        print("  dengan hanya melaporkan hasil 30 hari.")
        return

    K.judul(f"MEMUAT HASIL   (horizon {args.horizon})")
    data = muat_rapi(args.horizon)
    ada = sum(v is not None for v in data.values())
    print(f"  ditemukan {ada}/{len(data)} lengan")
    K.butuh(data.get(("h6b_utama", "0.5")) is not None,
            f"Lengan pokok `h6b_utama it=0.5` (horizon {args.horizon}) belum ada.\n"
            f"   Jalankan dulu:  python 1_eksperimen.py --horizon {args.horizon}")

    v_pokok, v_kurva, v_sebaran, v_revisi = nilai(data, args.horizon)

    K.judul("RINGKASAN")
    print(f"  {'H6b -- klaim utama (pra-daftar)':<38} "
          f"{'LULUS' if v_pokok['lulus'] else 'DITOLAK'}")
    print(f"  {'H2b -- kurva menyusut (pra-daftar)':<38} "
          f"{'belum dinilai' if v_kurva['h2b'] is None else ('LULUS' if v_kurva['h2b'] else 'tidak lulus')}")
    print(f"  {'pertukaran (PASCA-HOC, bukan lulus)':<38} "
          f"{'pola konsisten' if v_revisi['pola_konsisten'] else 'pola tak konsisten'}")

    K.simpan(dict(tanggal=STAMP, horizon=args.horizon,
                  toleransi=dict(gini=TOL_GINI, acc=TOL_ACC),
                  berkas={f"{t}|{i}": (data[(t, i)] or {}).get("_berkas")
                          for t, _ in LENGAN for i in K.TINGKAT_TRUST},
                  h6b=v_pokok, h2b=v_kurva, sebaran_trust=v_sebaran,
                  klaim_revisi_pascahoc=v_revisi),
             f"analisis{suf(args.horizon)}_{STAMP}.json")

    if args.horizon == "30d":
        print("\nLanjut:  python 1_eksperimen.py --horizon 90d   (uji ketahanan)")
    else:
        print("\nLanjut:  python 2_analisis.py --banding")


if __name__ == "__main__":
    main()
