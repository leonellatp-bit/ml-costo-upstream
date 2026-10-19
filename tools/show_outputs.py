import json, sys
nb = json.load(open(sys.argv[1]))
i = 0
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    i += 1
    print(f'----- celda {i}')
    for o in c['outputs']:
        if o['output_type'] == 'stream':
            print(''.join(o['text']))
        elif 'image/png' in o.get('data', {}):
            print('[figura]')
        elif 'text/plain' in o.get('data', {}):
            print(''.join(o['data']['text/plain'])[:6000])
