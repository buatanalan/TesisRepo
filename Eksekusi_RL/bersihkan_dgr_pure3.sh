#!/bin/bash
# Hapus artefak tahap DGR mode aliran-murni (pure3) supaya bisa dilatih ULANG dari nol
# setelah perbaikan penyebut gap-ratio (beta_denom: |r_star| -> std(return), 2026-08-30).
#
# HANYA TAHAP DGR. Tahap spesialis SENGAJA TIDAK DIHAPUS -- perbaikan itu tak
# menyentuhnya sama sekali: pd spesialis K=1, sehingga softmax atas satu elemen selalu
# menghasilkan beta=[1.0] apa pun penyebutnya (terverifikasi), dan `r_star` yang
# dihasilkan pun identik. Melatih ulang spesialis = membuang 3/4 anggaran komputasi
# tanpa mengubah apa pun.
#
# CATATAN PENTING: `--overwrite` pd pipeline TIDAK menghapus checkpoint. Resume
# dikendalikan `{TAG}_training_results.json` -- selama seed masih terdaftar di sana ia
# akan di-SKIP. Karena itu file itulah yang WAJIB dihapus, bukan sekadar .pt-nya.
#
# Pemakaian:
#   bash Eksekusi_RL/bersihkan_dgr_pure3.sh          # UJI COBA: hanya menampilkan
#   bash Eksekusi_RL/bersihkan_dgr_pure3.sh --hapus  # benar-benar menghapus
set -e
OUT=Eksekusi_RL/outputs
B=master_hybrid_ppo_dgr_90d_cwtfail120pen-2_preffeat_pairout

TAGS=(
  "${B}_noattn_pure3"          # sel1: attn OFF, P OFF
  "${B}_pure3"                 # sel2: attn ON , P OFF
  "${B}_pg0.1_noattn_pure3"    # sel3: attn OFF, P ON
  "${B}_pg0.1_pure3"           # sel4: attn ON , P ON
)

MODE_HAPUS=0
[ "$1" = "--hapus" ] && MODE_HAPUS=1

echo "=== Artefak DGR pure3 yang menjadi sasaran ==="
JML=0
for t in "${TAGS[@]}"; do
  for f in "$OUT/${t}_training_results.json" \
           "$OUT/${t}_eval_results.json" \
           "$OUT/${t}"_actor_seed*.pt \
           "$OUT/${t}"_critic_seed*.pt; do
    [ -e "$f" ] || continue
    echo "  $f"
    JML=$((JML+1))
    [ $MODE_HAPUS -eq 1 ] && rm -f "$f"
  done
done

echo
if [ $JML -eq 0 ]; then
  echo "Tidak ada artefak DGR pure3 -- sudah bersih, langsung latih saja."
elif [ $MODE_HAPUS -eq 1 ]; then
  echo "$JML berkas DIHAPUS."
else
  echo "$JML berkas akan dihapus. Ini UJI COBA -- belum ada yang dihapus."
  echo "Jalankan ulang dengan --hapus untuk benar-benar menghapus."
fi

echo
echo "Spesialis TIDAK disentuh (sengaja). Yang masih ada:"
ls "$OUT"/master_hybrid_ppo_specialist*_pure3_r_star_seed*.json 2>/dev/null \
  | sed 's#^#  #' || echo "  (belum ada -- spesialis perlu dilatih dulu)"
