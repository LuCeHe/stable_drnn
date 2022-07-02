import os
from tqdm import tqdm
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

ds = unzip_good_exps(
    GEXPERIMENTS, EXPERIMENTS,
    exp_identifiers=[''], except_folders=[],
    unzip_what=['history.json', 'config']
)


# for d in tqdm(ds, desc='Creating pandas'):
#     text_file = os.path.join(d, [l for l in os.listdir(d) if '.txt' in l][0])
