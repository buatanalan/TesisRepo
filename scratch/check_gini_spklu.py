import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Data_dan_Analisis\Analisis_Ketimpangan_SPKLU.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    outs = []
    for out in cell.get('outputs', []):
        if 'text' in out:
            outs.append(''.join(out['text']))
        elif 'data' in out and 'text/plain' in out['data']:
            outs.append(''.join(out['data']['text/plain']))
    out_str = ''.join(outs)
    
    # Check for Gini or Monthly or Annual
    if any(k in src.lower() or k in out_str.lower() for k in ['gini', 'bulanan', 'rentang', 'tahunan', '0.754', '0.802', '0.795', '0.851']):
        print(f"\n================ CELL {i} ({cell['cell_type']}) ================")
        lines_src = src.strip().split('\n')
        print(f"Source preview ({len(lines_src)} lines):")
        for line in lines_src[:15]:
            print("  ", line)
        if len(lines_src) > 15:
            print("   ...")
        if out_str:
            lines_out = out_str.strip().split('\n')
            print(f"Output preview ({len(lines_out)} lines):")
            for line in lines_out[:25]:
                print("  >", line)
            if len(lines_out) > 25:
                print("   > ...")
