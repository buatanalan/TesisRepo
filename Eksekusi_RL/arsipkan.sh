#!/usr/bin/env bash
# Pindahkan checkpoint & log usang ke outputs/arsip/, dengan manifes yang mencatat
# data apa, dalam tugas apa, konfigurasi bagaimana, dan mengapa diarsipkan.
#
# Tidak ada yang dihapus -- hanya dipindah. Klasifikasi lengkap ada di
# Dokumen_Penting/TRIASE_DATA.md
#
#   ./arsipkan.sh --lihat    tampilkan rencana, jangan pindahkan
#   ./arsipkan.sh            pindahkan setelah konfirmasi

set -euo pipefail
cd "$(dirname "$0")"

LIHAT=0
[ "${1:-}" = "--lihat" ] && LIHAT=1

ARSIP=outputs/arsip
MANIFES="$ARSIP/MANIFES.tsv"

# stem<TAB>tugas<TAB>konfigurasi<TAB>alasan
GOLONGAN=$(cat <<'EOF'
hppo_K1	Tahap A baseline kritik-tunggal	preset gabungan, trust beku 0,5, 300 iter, 30d	preset reward salah: suku Gini hanya 0,1% sinyal
hppo_K1_sb4x	Tahap A pasca-seimbang4x	sb4x, beku 0,5, 300 iter, 30d	dilatih sebelum perbaikan batas horizon
hppo_K1_sb4x_fixgini	Tahap Afix uji perbaikan delta-Gini	sb4x, beku 0,5, 300 iter, 30d	delta-Gini sudah diperbaiki, batas horizon belum
pppo_sb4x	P-PPO d=64 setia-paper Lin dkk.	sb4x, beku 0,5, 300 iter, 30d	pra-bnd; d=64 tak setara kapasitas dgn H-PPO
pppo_sb4x_d16	P-PPO d=16 pra-bnd	sb4x, beku 0,5, 300 iter, 30d	digantikan pppo_sb4x_d16_bnd
hppo_sb4x_ent0.05	kalibrasi entropi ent_coef=0,05	sb4x, beku 0,5, 300 iter, 30d	hipotesis kolaps-akibat-entropi gugur; penyebabnya panjang pelatihan
hppo_t3	Tahap 3 H-PPO trust dinamis	sb4x, dinamis, 300 iter, 30d	digantikan hppo_30d_abs (200 iter, konfigurasi identik)
pppo_t3	Tahap 3 P-PPO trust dinamis	sb4x, dinamis, 300 iter, 30d	digantikan pppo_30d_abs (200 iter, konfigurasi identik)
EOF
)

echo "=== rencana pengarsipan ==="
TOTAL=0
while IFS=$'\t' read -r stem tugas konfig alasan; do
    [ -z "$stem" ] && continue
    n=$(ls -1 outputs/t2_${stem}_seed*.pt 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && continue
    TOTAL=$((TOTAL + n))
    printf '  %-26s %d checkpoint  -- %s\n' "$stem" "$n" "$alasan"
done <<< "$GOLONGAN"
echo "  ------------------------------------------"
printf '  TOTAL %d checkpoint (beserta .jsonl & _meta.json)\n' "$TOTAL"

if [ "$LIHAT" = "1" ]; then
    echo; echo "(mode --lihat: tidak ada yang dipindahkan)"; exit 0
fi

echo
read -r -p "pindahkan ke $ARSIP/ ? [y/N] " j
case "$j" in [yY]) ;; *) echo "dibatalkan."; exit 0 ;; esac

mkdir -p "$ARSIP"
if [ ! -f "$MANIFES" ]; then
    printf 'stem\ttugas\tkonfigurasi\talasan_arsip\tdiarsipkan\n' > "$MANIFES"
fi

STAMP=$(date +%Y-%m-%d)
while IFS=$'\t' read -r stem tugas konfig alasan; do
    [ -z "$stem" ] && continue
    ls outputs/t2_${stem}_seed*.pt >/dev/null 2>&1 || continue
    mv -f outputs/t2_${stem}_seed*.pt        "$ARSIP"/ 2>/dev/null || true
    mv -f outputs/t2_${stem}_seed*.jsonl     "$ARSIP"/ 2>/dev/null || true
    mv -f outputs/t2_${stem}_seed*_meta.json "$ARSIP"/ 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\n' "$stem" "$tugas" "$konfig" "$alasan" "$STAMP" >> "$MANIFES"
    echo "  diarsipkan: $stem"
done <<< "$GOLONGAN"

echo
echo "selesai. Manifes: $MANIFES"
echo "Klasifikasi lengkap: Dokumen_Penting/TRIASE_DATA.md"
