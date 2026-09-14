"""Ringkasan uji penilaian relatif untuk keempat sel faktorial.

    .venv/bin/python Eksekusi_RL/_ringkas_penilaian_relatif.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

B = "master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout"
SEL = [("MASTER", "_noattn_pure3"), ("MASTER+Atensi", "_pure3"),
       ("MASTER+P", "_pg0.1_noattn_pure3"), ("P-MASTER", "_pg0.1_pure3")]
METRIK = [("M1_rasio_silang", "M1 rasio sensitivitas silang"),
          ("M2_laju_pembalikan", "M2 laju pembalikan peringkat"),
          ("M3_silang_rerata", "M3 efek silang pada p_j (rerata)"),
          ("M3_frac_arah_benar", "M3 proporsi arah benar (<0)"),
          ("M3_sendiri_rerata", "M3 efek sendiri pada p_j (rerata)"),
          ("M5_kesesuaian_wait", "M5 kesesuaian urutan dgn wait"),
          ("M4_gate", "M4 gerbang atensi sigmoid(gate_raw)"),
          ("M4_rasio_residu", "M4 ||gate*attended|| / ||vec||"),
          ("tanpa_atensi_M1_rasio_silang", "M1 bila atensi dimatikan (bobot sama)"),
          ("tanpa_atensi_M2_laju_pembalikan", "M2 bila atensi dimatikan (bobot sama)")]


def main():
    data = {}
    for lbl, suf in SEL:
        p = os.path.join(common.OUTDIR, f"uji_penilaian_relatif_{B}{suf}.json")
        if os.path.exists(p):
            data[lbl] = json.load(open(p, encoding="utf-8"))["ringkas"]
        else:
            print(f"[belum ada] {p}")
    lebar = 42
    print("\n" + "metrik".ljust(lebar) + "".join(l.rjust(20) for l, _ in SEL))
    for k, judul in METRIK:
        baris = judul.ljust(lebar)
        for lbl, _ in SEL:
            r = data.get(lbl, {}).get(k)
            baris += (f"{r['mean']:.4f}±{r['sd']:.4f}".rjust(20) if r else "-".rjust(20))
        print(baris)
    common.save_json(data, "uji_penilaian_relatif_ringkasan.json")
    print("\nSAVED -> outputs/uji_penilaian_relatif_ringkasan.json")


if __name__ == "__main__":
    main()
