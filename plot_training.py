import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

CSVPATH = os.path.join(EXPERIMENTS, 'summary.h5')
HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

if not os.path.exists(CSVPATH):

    ds = unzip_good_exps(
        GEXPERIMENTS, EXPERIMENTS,
        exp_identifiers=[''], except_folders=[],
        unzip_what=['history.json', 'config']
    )

    histories = {}
    df = pd.DataFrame(columns=[])

    for d in tqdm(ds, desc='Creating pandas'):
        history_path = os.path.join(d, 'other_outputs', 'history.json')
        config_path = os.path.join(d, '1', 'config.json')
        # text_file = os.path.join(d, [l for l in os.listdir(d) if '.txt' in l][0])

        with open(config_path) as f:
            config = json.load(f)

        with open(history_path) as f:
            history = json.load(f)

        results = {}
        results.update({k: v for k, v in config.items()})
        what = lambda k, v: max(v) if 'acc' in k else min(v)
        results.update({k: what(k, np.array(v)[~np.isnan(np.array(v))]) for k, v in history.items()})

        small_df = pd.DataFrame([results])

        df = df.append(small_df)
        print(history['val_sparse_mode_accuracy'])
        history = {k.replace('val_', ''): v for k, v in history.items() if 'val' in k}
        histories[d] = history


    df = df.sort_values(by='comments')

    df.to_hdf(CSVPATH, key='df', mode='w')
    json.dump(histories, open(HSITORIESPATH, "w"))
    # print(df.to_string())
else:
    # mdf = pd.read_csv(CSVPATH)
    df = pd.read_hdf(CSVPATH, 'df')  # load it
    with open(HSITORIESPATH) as f:
        histories = json.load(f)

print(df.to_string())