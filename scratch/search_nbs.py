import os, json, sys

sys.stdout.reconfigure(encoding='utf-8')

root = r'c:\Users\Lenovo\Documents\Tesis\Repo\Custom SImulation'
targets = {
    '0.802': 'Gini utilisasi 0.802',
    '0.795': 'Rentang bulanan 0.795',
    '0.851': 'Rentang bulanan 0.851',
    '2654': '2654 pasangan',
    '2.654': '2.654 pasangan',
    '501': '501 infra-setara',
    '4.55': 'median rasio 4.55',
    '10.3': 'median rasio 10.3',
    '61.2': 'Theil 61.2%',
    '0.754': 'Gini transaksi 0.754'
}

for dirpath, dirnames, filenames in os.walk(root):
    if any(x in dirpath for x in ['.git', '.gemini', 'node_modules', '.system_generated', 'outputs', '.venv']):
        continue
    for f in filenames:
        if f.endswith('.ipynb'):
            fpath = os.path.join(dirpath, f)
            try:
                with open(fpath, 'r', encoding='utf-8') as fp:
                    nb = json.load(fp)
                for i, cell in enumerate(nb.get('cells', [])):
                    src = ''.join(cell.get('source', []))
                    outs = []
                    for out in cell.get('outputs', []):
                        if 'text' in out:
                            outs.append(''.join(out['text']))
                        elif 'data' in out and 'text/plain' in out['data']:
                            outs.append(''.join(out['data']['text/plain']))
                    out_str = '\n'.join(outs)
                    
                    for t, desc in targets.items():
                        if t in src:
                            rel = os.path.relpath(fpath, root)
                            print(f"[SOURCE MATCH] {rel} Cell {i} ({cell['cell_type']}) -> target '{t}' ({desc})")
                        if t in out_str:
                            rel = os.path.relpath(fpath, root)
                            print(f"[OUTPUT MATCH] {rel} Cell {i} ({cell['cell_type']}) -> target '{t}' ({desc})")
            except Exception as e:
                pass
