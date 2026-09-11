import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Eksekusi_RL\analisis_blok_A_klaim_utama.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(3, 10):
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        src = ''.join(cell.get('source', []))
        print(f"=== Cell {idx} ({cell['cell_type']}) ===")
        print("[SOURCE]:")
        print(src)
        outs = []
        for out in cell.get('outputs', []):
            if 'text' in out:
                outs.append(''.join(out['text']))
            elif 'data' in out and 'text/plain' in out['data']:
                outs.append(''.join(out['data']['text/plain']))
        if outs:
            print("[OUTPUT]:")
            print(''.join(outs))
        print('='*60)
