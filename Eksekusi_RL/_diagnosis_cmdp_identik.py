"""Diagnosis: kenapa eval K3-base dan CMDP menghasilkan angka identik meski checkpoint
beda hash/ukuran file. Bandingkan tensor bobot aktual (state_dict), bukan hash biner
mentah (yg bisa beda krn metadata torch.save meski isi sama, atau sebaliknya)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import common

CKPT_BASE = os.path.join(common.OUTDIR, "master_ev_ppo_vwf_seimbang4x_K3_gap_sig1_actor_seed0.pt")
CKPT_CMDP = os.path.join(common.OUTDIR, "master_ev_ppo_cmdpE0.07lr0.5_vwf_seimbang4x_K3_gap_sig1_actor_seed0.pt")

sd_base = torch.load(CKPT_BASE, map_location="cpu")
sd_cmdp = torch.load(CKPT_CMDP, map_location="cpu")

print("=== Perbandingan state_dict (bobot mentah) ===")
keys_base = set(sd_base.keys())
keys_cmdp = set(sd_cmdp.keys())
print("kunci sama?", keys_base == keys_cmdp)
n_diff = 0
n_same = 0
for k in sorted(keys_base & keys_cmdp):
    a, b = sd_base[k], sd_cmdp[k]
    if torch.equal(a, b):
        n_same += 1
    else:
        n_diff += 1
        maxdiff = (a.float() - b.float()).abs().max().item()
        print(f"  BEDA  {k}: shape={tuple(a.shape)} max|diff|={maxdiff:.6g}")
print(f"\ntotal tensor: {len(keys_base)}, SAMA={n_same}, BEDA={n_diff}")
if n_diff == 0:
    print("\n!! BOBOT SAMA PERSIS -- checkpoint CMDP identik dgn K3-base meski hash file beda !!")
else:
    print(f"\nBobot GENUINELY beda ({n_diff} tensor) -- kalau eval tetap identik, "
         "masalahnya ada di jalur INFERENSI (act()/get_recommendation), bukan checkpoint.")
