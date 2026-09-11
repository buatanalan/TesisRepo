import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Dokumen_Penting\Pembuktian_Klaim_Empiris_2024.ipynb', 'r', encoding='utf-8') as fp:
    nb = json.load(fp)

for i in [0, 1, 2, 3, 54, 55, 87]:
    c = nb['cells'][i]
    print(f"=== Cell {i} ({c['cell_type']}) ===")
    src = ''.join(c.get('source', []))
    print(src[:1000])
