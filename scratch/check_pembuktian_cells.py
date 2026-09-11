import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Dokumen_Penting\Pembuktian_Klaim_Empiris_2024.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(45, 54):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    outs = []
    for out in cell.get('outputs', []):
        if 'text' in out:
            outs.append(''.join(out['text']))
        elif 'data' in out and 'text/plain' in out['data']:
            outs.append(''.join(out['data']['text/plain']))
    out_str = ''.join(outs)
    
    print(f"=== Cell {i} ({cell['cell_type']}) ===")
    print("[SOURCE]:")
    print(src[:600])
    if out_str:
        print("[OUTPUT]:")
        print(out_str[:800])
    print('-'*50)
