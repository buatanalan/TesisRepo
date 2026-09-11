import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Data_dan_Analisis\Analisis_Ketimpangan_SPKLU.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

targets = [8, 9, 10, 11, 14, 19, 30, 31, 32, 52, 53, 54]

for idx in targets:
    if idx < len(nb['cells']):
        c = nb['cells'][idx]
        print(f"=== Cell {idx} ({c['cell_type']}) ===")
        src = ''.join(c.get('source', []))
        print("[SOURCE]:")
        print(src[:500])
        outs = []
        for out in c.get('outputs', []):
            if 'text' in out:
                outs.append(''.join(out['text']))
            elif 'data' in out and 'text/plain' in out['data']:
                outs.append(''.join(out['data']['text/plain']))
        if outs:
            print("[OUTPUT]:")
            print(''.join(outs)[:1000])
        print('-'*60)
