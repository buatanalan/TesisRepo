#!/usr/bin/env bash
# Kirim hasil eksperimen dari server ke git.
#
# Sebagian besar isi outputs/ sengaja diabaikan .gitignore supaya repo tidak membengkak.
# Skrip ini menambahkan HANYA berkas yang diminta, dengan `git add -f`, setelah lebih dulu
# menampilkan apa yang akan dikirim beserta ukurannya. Tak ada yang di-commit sebelum
# Anda mengonfirmasi.
#
# Pemakaian:
#   ./kirim_hasil.sh 30d                 # semua hasil bertanda 30d
#   ./kirim_hasil.sh 90d                 # semua hasil bertanda 90d
#   ./kirim_hasil.sh 30d --dengan-bobot  # ikutkan checkpoint .pt (besar)
#   ./kirim_hasil.sh 30d --lihat         # hanya tampilkan, jangan commit
#
# Jalankan dari dalam Eksekusi_RL/.

set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
    echo "pemakaian: $0 <tag>  [--dengan-bobot] [--lihat]"
    echo "  <tag> mis. 30d, 90d, t3"
    exit 1
fi
shift || true

DENGAN_BOBOT=0
LIHAT=0
for arg in "$@"; do
    case "$arg" in
        --dengan-bobot) DENGAN_BOBOT=1 ;;
        --lihat)        LIHAT=1 ;;
        *) echo "argumen tak dikenal: $arg"; exit 1 ;;
    esac
done

cd "$(dirname "$0")"
if [ ! -d outputs ]; then
    echo "outputs/ tak ditemukan -- jalankan dari dalam Eksekusi_RL/"
    exit 1
fi

# Kumpulkan kandidat. JSONL = kurva per-iterasi (yang dipakai menganalisis kapan run
# memburuk); meta = konfigurasi + riwayat; uji_*.json = tabel hasil akhir.
# Pola master_ev_ppo_*/master_ddpg_* (2026-08) DITAMBAHKAN di sini -- skrip lama hanya
# cocok skema t2_*/_eval.log, sehingga hasil pipeline PPO/DDPG lebih baru (eval_results.
# json, training_results.json, actor_seed*.pt, *_pipeline.log) tak pernah ikut terkirim
# meski sudah lama ada di outputs/ server (ditemukan 2026-08-23, log accW1 terkirim
# manual tapi eval_results/training_results/checkpoint TIDAK -- akar penyebabnya di sini).
mapfile -t BERKAS < <(
    ls -1 outputs/t2_*"$TAG"*.jsonl                2>/dev/null || true
    ls -1 outputs/t2_*"$TAG"*_meta.json            2>/dev/null || true
    ls -1 outputs/uji_*"$TAG"*.json                2>/dev/null || true
    ls -1 outputs/*"$TAG"*_eval.log                2>/dev/null || true
    ls -1 outputs/*"$TAG"*_eval_results.json       2>/dev/null || true
    ls -1 outputs/*"$TAG"*_training_results.json   2>/dev/null || true
    ls -1 outputs/*"$TAG"*_pipeline.log            2>/dev/null || true
    ls -1 outputs/*"$TAG"*.jsonl                   2>/dev/null || true
    if [ "$DENGAN_BOBOT" = "1" ]; then
        ls -1 outputs/t2_*"$TAG"*.pt               2>/dev/null || true
        ls -1 outputs/*"$TAG"*_actor_seed*.pt      2>/dev/null || true
        ls -1 outputs/*"$TAG"*_critic_seed*.pt     2>/dev/null || true
    fi
)
# Buang duplikat (beberapa pola di atas bisa saling tumpang tindih) sambil
# mempertahankan urutan -- penting krn `mapfile` di atas bisa mendaftarkan berkas
# yang sama dua kali (mis. *_eval_results.json matched oleh uji_*.json juga).
mapfile -t BERKAS < <(printf '%s\n' "${BERKAS[@]}" | awk '!seen[$0]++')

if [ "${#BERKAS[@]}" -eq 0 ]; then
    echo "tak ada berkas cocok untuk tag '$TAG'"
    exit 1
fi

echo "=== akan dikirim (tag: $TAG) ==="
for f in "${BERKAS[@]}"; do
    printf '  %10s  %s\n' "$(du -h "$f" | cut -f1)" "$f"
done
TOTAL=$(du -ch "${BERKAS[@]}" | tail -1 | cut -f1)
echo "-----------------------------------------"
printf '  %10s  TOTAL (%d berkas)\n' "$TOTAL" "${#BERKAS[@]}"

if [ "$DENGAN_BOBOT" = "0" ]; then
    echo
    echo "catatan: checkpoint .pt TIDAK disertakan. Tambahkan --dengan-bobot bila"
    echo "         hasilnya perlu dievaluasi ulang di mesin lain."
fi

if [ "$LIHAT" = "1" ]; then
    echo
    echo "(mode --lihat: tidak ada yang di-commit)"
    exit 0
fi

echo
read -r -p "lanjut commit? [y/N] " jawab
case "$jawab" in
    [yY]) ;;
    *) echo "dibatalkan."; exit 0 ;;
esac

# -f wajib: pola ini diabaikan .gitignore secara sengaja, dan pengabaian itu tetap
# dipertahankan supaya run berikutnya tidak ikut terbawa tanpa disadari.
git add -f "${BERKAS[@]}"

if git diff --cached --quiet; then
    echo "tak ada perubahan untuk di-commit (berkas sudah identik dgn yang terlacak)."
    exit 0
fi

git commit -m "Hasil eksperimen $TAG dari server (${#BERKAS[@]} berkas)"

echo
echo "commit selesai. Kirim ke remote dengan:"
echo "    git push"
