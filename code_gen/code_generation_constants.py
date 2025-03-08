from fcntl import F_GETLEASE

CONV_LAYER_TYPES = ['s', 'pw', 'dw']
MODEL_NAME = 'gprox_3'

DEFAULT_DATA_LAYOUT = 'CHW'
MODELS_HWC_LAYERS = {}
ADD_LAYER_ID_IN_CPP = 3
AVG_POOL_LAYER_ID_IN_CPP = 4
# MODELS_HWC_LAYERS = {'mob_v1': [2, 3, 4, 5], 'mob_v2': [
#     2, 3, 4], 'gprox_3': [3, 4], 'xce_r': [2, 3, 4]}
# uniform_mobilenetv2_25
