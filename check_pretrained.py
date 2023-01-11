import os
import tensorflow as tf
from tqdm import tqdm
from GenericTools.keras_tools.esoteric_optimizers.AdamW import AdamW as AdamW2
from GenericTools.keras_tools.esoteric_layers import AddLossLayer, AddMetricsLayer
from GenericTools.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization
from GenericTools.keras_tools.learning_rate_schedules import DummyConstantSchedule
from sg_design_lif.neural_models import maLSNN, maLSNNb

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')


ds = [d for d in os.listdir(GEXPERIMENTS)]

print(ds)

for d in tqdm(ds):
    dpath = os.path.join(GEXPERIMENTS, d)

    model = tf.keras.models.load_model(
        dpath,
        custom_objects={
            'maLSNN': maLSNN, 'maLSNNb': maLSNNb, 'RateVoltageRegularization': RateVoltageRegularization,
            'AddLossLayer': AddLossLayer, 'AddMetricsLayer': AddMetricsLayer,
            'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
            'AdamW': AdamW2, 'DummyConstantSchedule': DummyConstantSchedule

        }
    )

    if 'wordptb' in d:
        stack = '1300c300'
    else:
        weights = model.get_weights()
        if 'maLSNN' in d:
            if len(weights) == 9:
                stack = '1'
            elif len(weights) == 51:
                stack = '7'
            elif len(weights) == 16:
                stack = '2'
            else:
                print('Unknown stack')
        else:

            if len(weights) == 5:
                stack = '1'
            elif len(weights) == 23:
                stack = '7'
            elif len(weights) == 8:
                stack = '2'
            else:
                print('Unknown stack')

    path_pretrained = dpath.replace('.h5', f'_stack{stack}.h5')
    model.save(path_pretrained)
