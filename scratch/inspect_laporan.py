import json, sys

sys.stdout.reconfigure(encoding='utf-8')

for p, cells in [
    (r'Synthetic Model\Estimasi_Per_Tier\08_Laporan_Simulasi_Lengkap.ipynb', [27, 28, 29]),
    (r'Dokumen_Penting\Dokumentasi_Pemodelan_Simulasi_Lengkap.ipynb', [32, 33, 34]),
]:
    with open(p, 'r', encoding='utf-8') as fp:
        nb = json.load(fp)
    print(f"=== {p} ===")
    for c in cells:
        print(f"--- Cell {c} ---")
        src = ''.join(nb['cells'][c].get('source', []))
        print(src[:1000])
