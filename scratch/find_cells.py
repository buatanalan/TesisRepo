import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Data_dan_Analisis\Analisis_Ketimpangan_SPKLU.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [31, 32]:
    cell = nb['cells'][idx]
    print(f"--- Cell {idx} ({cell['cell_type']}) ---")
    outs = []
    for out in cell.get('outputs', []):
        if 'text' in out:
            outs.append(''.join(out['text']))
        elif 'data' in out and 'text/plain' in out['data']:
            outs.append(''.join(out['data']['text/plain']))
    print(''.join(outs))
