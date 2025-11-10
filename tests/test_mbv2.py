from torq.testing.tensorflow import *
import pytest

# make sure tests use reproducible weights
tf.keras.utils.set_random_seed(21321)

inputs = tf.keras.Input(shape=(320, 320, 3), batch_size=1)
model = tf.keras.applications.MobileNetV2(weights=None, input_tensor=inputs, include_top=False)

generate_tests_from_model(model, globals(), pytest.mark.ci)