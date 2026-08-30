#!/bin/bash
# Hapus artefak tahap tertentu pada eksperimen aliran-murni (pure3), supaya bisa
# dilatih ULANG dari nol. Menggantikan `bersihkan_dgr_pure3.sh` (yg hanya bisa DGR).
#
# Pemakaian:
#   bash Eksekusi_RL/bersihkan_pure3.sh <tahap>[,<tahap>...] [--hapus]
#     <tahap> : wait | gini | accept | dgr | semua
#     tanpa --hapus = UJI COBA (hanya menampilkan, tidak menghapus apa pun)
#
# Contoh:
#   bash Eksekusi_RL/bersihkan_pure3.sh gini              # lihat dulu
#   bash Eksekusi_RL/bersihkan_pure3.sh gini --hapus      # hapus spesialis gini + DGR
#   bash Eksekusi_RL/bersihkan_pure3.sh semua --hapus     # ulang total
#
# KETERGANTUNGAN OTOMATIS: menghapus spesialis mana pun SELALU ikut menghapus DGR.
# Alasannya: DGR sudah "memakan" r_star lama. Bila spesialis dilatih ulang tapi DGR
# dibiarkan, DGR akan di-SKIP (seed-nya masih terdaftar di training_results.json) dan
# r_star BARU tak pernah terpakai -- diam-diam menghasilkan hasil campuran lama+baru.
#
# CATATAN: `--overwrite` pd pipeline TIDAK menghapus checkpoint. Resume dikendalikan
# `{TAG}_training_results.json`; selama seed terdaftar di sana ia di-SKIP. Karena itu
# berkas ITU yang wajib dihapus, bukan cuma .pt-nya.
set -e
OUT=Eksekusi_RL/outputs
SUF=90d_cwtfail120pen-2_preffeat_pairout

# 4 sel faktorial 2x2 -- akhiran yg membedakannya
SEL=("_noattn_pure3" "_pure3" "_pg0.1_noattn_pure3" "_pg0.1_pure3")

[ -z "$1" ] && { echo "Pemakaian: bash $0 <wait|gini|accept|dgr|semua>[,...] [--hapus]"; exit 1; }
PILIH="$1"
MODE_HAPUS=0
[ "$2" = "--hapus" ] && MODE_HAPUS=1

# Terjemahkan pilihan -> awalan tag
PREFIX=()
add() { PREFIX+=("$1"); }
IFS=',' read -ra TAHAP <<< "$PILIH"
ADA_SPESIALIS=0
for t in "${TAHAP[@]}"; do
  case "$t" in
    wait)   add "master_hybrid_ppo_specialist0_wait_${SUF}";   ADA_SPESIALIS=1 ;;
    gini)   add "master_hybrid_ppo_specialist1_gini_${SUF}";   ADA_SPESIALIS=1 ;;
    accept) add "master_hybrid_ppo_specialist2_accept_${SUF}"; ADA_SPESIALIS=1 ;;
    dgr)    add "master_hybrid_ppo_dgr_${SUF}" ;;
    semua)  add "master_hybrid_ppo_specialist0_wait_${SUF}"
            add "master_hybrid_ppo_specialist1_gini_${SUF}"
            add "master_hybrid_ppo_specialist2_accept_${SUF}"
            add "master_hybrid_ppo_dgr_${SUF}"; ADA_SPESIALIS=1 ;;
    *) echo "Tahap tak dikenal: '$t'"; exit 1 ;;
  esac
done
# Ketergantungan: spesialis dihapus -> DGR wajib ikut
if [ $ADA_SPESIALIS -eq 1 ] && [[ ",$PILIH," != *",dgr,"* ]] && [[ ",$PILIH," != *",semua,"* ]]; then
  add "master_hybrid_ppo_dgr_${SUF}"
  echo "(!) Spesialis dihapus -> DGR ikut dihapus otomatis (r_star lama sudah terpakai di sana)"
  echo
fi

echo "=== Berkas sasaran ==="
JML=0
for p in "${PREFIX[@]}"; do
  for s in "${SEL[@]}"; do
    T="${p}${s}"
    for f in "$OUT/${T}_training_results.json" \
             "$OUT/${T}_eval_results.json" \
             "$OUT/${T}"_r_star_seed*.json \
             "$OUT/${T}"_actor_seed*.pt \
             "$OUT/${T}"_critic_seed*.pt; do
      [ -e "$f" ] || continue
      echo "  $f"
      JML=$((JML+1))
      [ $MODE_HAPUS -eq 1 ] && rm -f "$f"
    done
  done
done

echo
if [ $JML -eq 0 ]; then
  echo "Tidak ada berkas yang cocok -- sudah bersih."
elif [ $MODE_HAPUS -eq 1 ]; then
  echo "$JML berkas DIHAPUS. Jalankan ulang retrain_hybrid_ppo_pure3_2x2.sh --"
  echo "tahap yang checkpoint-nya masih ada akan otomatis di-SKIP."
else
  echo "$JML berkas akan dihapus. Ini UJI COBA -- belum ada yang dihapus."
  echo "Tambahkan --hapus untuk benar-benar menghapus."
fi
