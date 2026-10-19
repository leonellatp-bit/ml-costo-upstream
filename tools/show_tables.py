import json, sys, io
import pandas as pd
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 70)
nb = json.load(open(sys.argv[1]))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i, c in enumerate(cells, 1):
    for o in c['outputs']:
        d = o.get('data', {})
        if o['output_type'] == 'stream':
            print(f'== celda {i} (stdout)'); print(''.join(o['text']))
        elif 'text/html' in d:
            try:
                df = pd.read_html(io.StringIO(''.join(d['text/html'])))[0]
                print(f'== celda {i}'); print(df.head(40).to_string())
            except Exception as e:
                print('html?', e)
