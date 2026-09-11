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

matches = []

for nb_path in notebooks:
    rel_path = os.path.relpath(nb_path, root)
    try:
        with open(nb_path, 'r', encoding='utf-8', errors='ignore') as fp:
            nb = json.load(fp)
    except Exception:
        continue
        
    for cell_idx, cell in enumerate(nb.get('cells', [])):
        # check all text in cell
        cell_str = json.dumps(cell)
        # find floats close to 0.802: 0.8015 to 0.8025
        float_matches = re.findall(r'0\.802\d*', cell_str)
        if float_matches:
            matches.append((rel_path, cell_idx, cell.get('cell_type'), float_matches, cell_str[:300]))

print(f"Total float matches in all {len(notebooks)} notebooks: {len(matches)}")
for m in matches:
    print(f"File: {m[0]} | Cell {m[1]} ({m[2]})")
    print(f"  Values: {set(m[3])}")
