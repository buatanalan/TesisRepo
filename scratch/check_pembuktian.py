import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'Dokumen_Penting\Pembuktian_Klaim_Empiris_2024.ipynb', 'r', encoding='utf-8') as f:
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
    
    # Check if cell mentions Gini, bulanan, klaim 1, klaim 2, 0.802, 0.754, 0.795, 0.851, dll
    keywords = ['gini', 'klaim 1', 'klaim 2', 'klaim 3', 'bulanan', '0.802', '0.754', '0.834', '0.795', '0.851', 'tahunan']
    if any(k in src.lower() for k in keywords) or any(k in out_str.lower() for k in ['0.802', '0.795', '0.851']):
        print(f"\n================ CELL {i} ({cell['cell_type']}) ================")
        lines_src = src.strip().split('\n')
        print(f"[SOURCE] ({len(lines_src)} lines):")
        for line in lines_src[:25]:
            print("  ", line)
        if len(lines_src) > 25:
            print("   ...")
        if out_str:
            lines_out = out_str.strip().split('\n')
            print(f"[OUTPUT] ({len(lines_out)} lines):")
            for line in lines_out[:30]:
                print("  >", line)
            if len(lines_out) > 30:
                print("   > ...")
