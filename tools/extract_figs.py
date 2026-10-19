import json, sys, base64, os
nb = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
i = 0; k = 0
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    i += 1
    for o in c['outputs']:
        if 'image/png' in o.get('data', {}):
            k += 1
            open(f"{out}/c{i:02d}_{k}.png", 'wb').write(base64.b64decode(o['data']['image/png']))
print(sorted(os.listdir(out)))
