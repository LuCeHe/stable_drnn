import os
from alif_sg.generate_data.heidelberg_generator import SpokenHeidelbergDigits
from alif_sg.generate_data.mnist_generators import SeqMNIST
from alif_sg.generate_data.ptb_generator import PTBGenerator

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
STATSPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'task_stats.csv'))


def Task(timerepeat=1, batch_size=64, steps_per_epoch=None, epochs=1, name='time_ae', train_val_test='train',
         neutral_phase_length=0, category_coding='onehot', inherit_from_gen=False, maxlen=100, output_type='[io]',
         lr=1e-4, comments=''):
    if 'ptb' == name:
        gen = PTBGenerator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='',
            config=comments,
            lr=lr)

    elif 'wordptb' == name:
        gen = PTBGenerator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='',
            char_or_word='word',
            config=comments,
            lr=lr)


    elif 'wordptb_oh' == name:
        gen = PTBGenerator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='onehot',
            char_or_word='word',
            lr=lr)

    elif name == 'heidelberg':
        gen = SpokenHeidelbergDigits(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            lr=lr)
    elif name == 'sl_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=False,
            spike_latency=True)

    elif name == 's_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=False)

    elif name == 'ps_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=True)

    elif name == 'ss_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=False,
            poisson_input=True)

    elif name == 'sps_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=True,
            poisson_input=True)

    else:
        raise NotImplementedError

    if not hasattr(gen, 'in_len'): gen.in_len = maxlen
    if not hasattr(gen, 'out_len'): gen.out_len = maxlen
    if not hasattr(gen, 'name'): gen.name = name
    if not hasattr(gen, 'timerepeat'): gen.timerepeat = timerepeat
    if not hasattr(gen, 'n_regularizations'): gen.n_regularizations = 0
    gen.output_type = output_type

    return gen


def checkTaskMeanVariance(task_name):
    import pandas as pd
    from tqdm import tqdm
    import numpy as np

    if not os.path.exists(STATSPATH):
        data = {'task_name': [], 'mean': [], 'var': []}

        # create dataframe
        df = pd.DataFrame(data)
        df.to_csv(STATSPATH)
    else:
        df = pd.read_csv(STATSPATH)

    if not task_name in df['task_name'].values:
        gen = Task(batch_size=64, name=task_name, train_val_test='train', steps_per_epoch=None)
        spe = gen.steps_per_epoch

        full_mean = 0
        full_var = 0
        for i in tqdm(range(spe)):
            # idx = 1 if 'wordptb' in task_name else 0
            idx = 0
            batch = gen.__getitem__(i)[0][idx]
            mean_batch = np.mean(np.mean(np.mean(batch, axis=2), axis=1), axis=0)
            var_batch = np.mean(np.mean(np.std(batch, axis=2) ** 2, axis=1), axis=0)

            full_mean += mean_batch
            full_var += var_batch

        full_mean /= spe
        full_var /= spe
        new_row = {'task_name': task_name, 'mean': full_mean, 'var': full_var}
        df = df.append(new_row, ignore_index=True)
        df.to_csv(STATSPATH)
    else:
        full_mean = df.loc[df.task_name == task_name, 'mean'].values[0]
        full_var = df.loc[df.task_name == task_name, 'var'].values[0]

    return full_mean, full_var


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm

    tasks = ['sl_mnist', 'heidelberg']
    for task_name in tasks:
        print('-' * 50)
        print(task_name)
        full_mean, full_var = checkTaskMeanVariance(task_name)
        print('mean {}, var {}'.format(full_mean, full_var))

    # heidelberg
    # mean 0.04388437186716791, var 0.04195531663882064
    # sl_mnist
    # mean 0.0036727774199098347, var 0.0036592709210085173
    # wordptb
    # mean 0.00010001000191550702, var 0.00010000323345397739
