import os
import tensorflow as tf
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

for d in ds:
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
    model.summary()
    if 'wordptb' in d:
        stack = '1300:300'
        print('wordptb')
        model.predict(tf.random.uniform((1, 1, 1, 1), dtype=tf.float32))
    else:
        weights = model.get_weights()