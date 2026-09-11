import os, sys

sys.stdout.reconfigure(encoding='utf-8')

targets = ['2654', '2.654', '0.802', '0.795', '0.851', '4.55', '61.2%']

root = r'c:\Users\Lenovo\Documents\Tesis\Repo\Custom SImulation'

matches = {}

search_dirs = ['Data_dan_Analisis', 'Dokumen_Penting', 'archive', 'Synthetic Model', 'Dokumen tambahan', 'Eksekusi_Hipotesis', 'Eksekusi_RL', 'marl_spklu']

for sdir in search_dirs:
    dir_to_walk = os.path.join(root, sdir)
    for dirpath, dirnames, filenames in os.walk(dir_to_walk):
        if any(x in dirpath for x in ['.git', '.gemini', 'node_modules', '.system_generated', 'outputs', '.venv', '__pycache__']):
            continue
    for f in filenames:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.ipynb', '.py', '.md', '.txt']:
            filepath = os.path.join(dirpath, f)
            if os.path.getsize(filepath) > 10 * 1024 * 1024:
                continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    found = [t for t in targets if t in content]
                    if found:
                        matches[filepath] = found
            except Exception as e:
                pass

for k, v in matches.items():
    rel = os.path.relpath(k, root)
    print(f"{rel} -> {v}")
