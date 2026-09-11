import os, json, sys, re

sys.stdout.reconfigure(encoding='utf-8')

root = r'c:\Users\Lenovo\Documents\Tesis\Repo\Custom SImulation'
notebooks = []

for dirpath, dirnames, filenames in os.walk(root):
    if any(x in dirpath for x in ['.git', '.gemini', 'node_modules', '.system_generated', '.venv']):
        continue
    for f in filenames:
        if f.endswith('.ipynb'):
            notebooks.append(os.path.join(dirpath, f))

print(f"Total notebooks to scan: {len(notebooks)}")

results = []

for nb_path in notebooks:
    rel_path = os.path.relpath(nb_path, root)
    try:
        with open(nb_path, 'r', encoding='utf-8', errors='ignore') as fp:
            nb = json.load(fp)
    except Exception as e:
        print(f"Error reading {rel_path}: {e}")
        continue
        
    for cell_idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type', '')
        source_text = ''.join(cell.get('source', []))
        
        # Check source
        if '0.802' in source_text or '0,802' in source_text:
            results.append({
                'file': rel_path,
                'cell': cell_idx,
                'cell_type': cell_type,
                'location': 'SOURCE',
                'snippet': source_text[:400]
            })
            
        # Check outputs
        out_texts = []
        for out in cell.get('outputs', []):
            if 'text' in out:
                out_texts.append(''.join(out['text']))
            if 'data' in out:
                for mime, val in out['data'].items():
                    if isinstance(val, list):
                        out_texts.append(''.join(val))
                    elif isinstance(val, str):
                        out_texts.append(val)
        
        full_out = '\n'.join(out_texts)
        if '0.802' in full_out or '0,802' in full_out:
            # find surrounding lines
            lines = full_out.split('\n')
            matched_lines = [l for l in lines if '0.802' in l or '0,802' in l]
            results.append({
                'file': rel_path,
                'cell': cell_idx,
                'cell_type': cell_type,
                'location': 'OUTPUT',
                'matched_lines': matched_lines[:5],
                'snippet': full_out[:500]
            })

print(f"\nTotal matches found: {len(results)}\n")
for r in results:
    print(f"File: {r['file']} | Cell: {r['cell']} ({r['cell_type']}) | In: {r['location']}")
    if 'matched_lines' in r:
        print(f"  Matched line(s): {r['matched_lines']}")
    else:
        print(f"  Source snippet: {r['snippet'].strip()[:200]}")
    print("-" * 60)
