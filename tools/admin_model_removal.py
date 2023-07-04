import os, shutil
from tqdm import tqdm

from pyaromatics.stay_organized.utils import str2val
from sg_design_lif.config.config import default_config


FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
GEXPERIMENTS = os.path.abspath(os.path.join(CDIR, '..', 'good_experiments'))

def get_lsctype(comments):
    if 'supnpsd' in comments:
        lsctype = 'supnpsd'

    elif 'supsubnpsd' in comments:
        lsctype = 'supsubnpsd'

    elif 'logradius' in comments:
        lsctype = 'logradius'

    elif 'radius' in comments:
        lsctype = 'radius'

    elif 'entrywise' in comments:
        lsctype = 'entrywise'

    elif 'lscvar' in comments:
        lsctype = 'lscvar'
    else:
        lsctype = 'other'
    return lsctype


def get_pretrained_file(comments, s, net_name, task_name, ostack):
    target_norm = str2val(comments, 'targetnorm', float, default=1)
    if ostack == 'None':
        ostack = None
    elif ostack in ['1', '3', '5', '7']:
        ostack = int(ostack)

    stack, batch_size, embedding, n_neurons, lr = default_config(
        ostack, None, None, None, .1, task_name, net_name, setting='LSC'
    )

    c = ''
    if 'targetnorm' in comments:
        c += f'_tn{str(target_norm).replace(".", "p")}'

    if 'randlsc' in comments:
        c += '_randlsc'

    if 'lscshuffw' in comments:
        c += '_lscshuffw'

    if 'gausslsc' in comments:
        c += '_gausslsc'

    if 'learnsharp' in comments:
        c += '_ls'

    if 'learndamp' in comments:
        c += '_ld'

    lsct = get_lsctype(comments)
    return f"pretrained_s{s}_{net_name}_{lsct}_{task_name}_stack{str(stack).replace(':', 'c')}{c}.h5"




def remove_nonrec_pretrained_extra(experiments, remove_opposite=True, folder=None, net_name='ffn'):
    files = []
    print('Desired:')
    for exp in experiments:
        if 'onlypretrain' in exp['comments'][0]:
            lsct = get_lsctype(exp['comments'][0])
            file = f"pretrained_s{exp['seed'][0]}_{net_name}" \
                   f"_{exp['dataset'][0]}_{exp['activation'][0]}_{lsct}.h5"
            print(file)
            files.append(file)

    if folder is None:
        folder = GEXPERIMENTS

    safety_folder = os.path.abspath(os.path.join(folder, '..', 'safety'))
    os.makedirs(safety_folder, exist_ok=True)

    existing_pretrained = [d
                           for d in os.listdir(folder)
                           if 'pretrained_' in d and '.h5' in d and f'_{net_name}_' in d]

    which_is_missing = [f for f in files if not f in existing_pretrained]
    print('Missing:')
    for f in which_is_missing:
        print(f)

    pbar = tqdm(total=len(existing_pretrained))
    removed = 0
    print('\nRemoving:')
    for d in existing_pretrained:
        # copy d file to safety folder
        # print(d)
        # print(os.path.join(folder, d))
        # print(os.path.join(safety_folder, d))

        if os.path.exists(os.path.join(folder, d)):
            if os.path.exists(os.path.join(safety_folder, d)):
                os.remove(os.path.join(safety_folder, d))
                pass
            shutil.copy(os.path.join(folder, d), os.path.join(safety_folder, d))

        if not d in files and remove_opposite:
            os.remove(os.path.join(folder, d))
            removed += 1

        if d in files and not remove_opposite:
            os.remove(os.path.join(folder, d))
            removed += 1

        pbar.update(1)
        pbar.set_description(f"Removed {removed} of {len(existing_pretrained)}")



def remove_pretrained_extra(experiments, remove_opposite=True, folder=None, erase_safety=False, truely_remove=True):
    files = []
    print('Desired:')
    for exp in experiments:
        if 'onlypretrain' in exp['comments'][0]:
            file = get_pretrained_file(
                comments=exp['comments'][0],
                s=exp['seed'][0],
                net_name=exp['net'][0],
                task_name=exp['task'][0],
                ostack=exp['stack'][0]
            )
            print(file)
            files.append(file)

    if folder is None:
        folder = GEXPERIMENTS

    safety_folder = os.path.abspath(os.path.join(folder, '..', 'safety'))
    os.makedirs(safety_folder, exist_ok=True)

    existing_pretrained = [
        d for d in os.listdir(folder)
        if 'pretrained_' in d
           and '.h5' in d
           and not '_ffn_' in d
           and not '_effnet_' in d
    ]

    which_is_missing = [f for f in files if not f in existing_pretrained]
    print('Missing:')
    for f in which_is_missing:
        print(f)

    if truely_remove:

        pbar = tqdm(total=len(existing_pretrained))
        removed = 0
        for d in existing_pretrained:
            # copy d file to safety folder
            shutil.copy(os.path.join(folder, d), os.path.join(safety_folder, d))

            if not d in files and remove_opposite and truely_remove:
                os.remove(os.path.join(folder, d))
                removed += 1

            if d in files and not remove_opposite and truely_remove:
                os.remove(os.path.join(folder, d))
                if erase_safety:
                    os.remove(os.path.join(safety_folder, d))
                removed += 1

            pbar.update(1)
            pbar.set_description(f"Removed {removed} of {len(existing_pretrained)}")
