from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
import os


def make_grid(script, params, gpus=[], bs=1, changer={}):
    scheduler = []
    n_gpu = len(gpus)
    grid = ParameterGrid(params)

    for i, p in enumerate(grid):
        base = 'python ' + script
        for k, _ in p.items():
            base += ' --{} {{{}}}'.format(k, changer.get(k, k))
        if len(gpus) > 0:
            base += ' --gpu_id {gpu}'

        p['gpu'] = gpus[i % n_gpu]
        run = ''
        run = base.format(**p)
        run += '\n' if (i + 1) % (n_gpu * bs) == 0 else ' &\n'
        scheduler.append(run)

    return scheduler


def get_logs(df, script):
    logs = pd.DataFrame()
    for _id, row in df.iterrows():
        if '{}-logs.csv'.format(script) not in os.listdir(row['root']):
            print('no logs for ', row['root'])
            continue

        tmp = pd.read_csv(row['root']/'{}-logs.csv'.format(script))
        for k, v in row.items():
            tmp[k] = str(v) if isinstance(v, list) else v
        logs = logs.append(tmp, ignore_index=True)
    return logs


def write_sh(fname, grid):
    with open(fname, 'w') as f:
        f.write("echo '' > pids.txt\n")
        f.write("echo $$ > pids.txt\n\n")
        for sh in grid:
            f.write(sh)
            f.write('FOO_PID=$!\n')
            f.write('echo $FOO_PID >> pids.txt\n\n')
