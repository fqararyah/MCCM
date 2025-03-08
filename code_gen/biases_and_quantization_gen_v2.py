from random import uniform
import numpy as np
import utils
import math
import code_generation_constants as cgc
import os

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

from_files = True
#########################################################################
biases_files_location = '../extract_tflite_model_metadata/{}/biases/'.format(
        cgc.MODEL_NAME)
weights_files_location = '../extract_tflite_model_metadata/{}/weights/'.format(
        cgc.MODEL_NAME)
weights_scales_files_location = weights_files_location
fms_scales_files_location = '../extract_tflite_model_metadata/{}/fms/'.format(
        cgc.MODEL_NAME)

#########################################################################

#########################################################################
model_dag = utils.read_model_dag()
#########################################################################

#########################################################################
secondary_layer_fms_scales_files_formats = {}
secondary_layer_fms_zero_points_files_formats = {}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


weights_file_format = 'weights_{}.txt'
weights_scales_file_format = weights_scales_files_location + \
    'weights_{}_scales.txt'
weights_zero_points_file_format = weights_scales_files_location + \
    'weights_{}_zps.txt'
biases_file_format = biases_files_location + 'biases_{}.txt'
#########################################################################


# './out/dw_weights.h'
weights_dir = '../{}_weights'.format(cgc.MODEL_NAME)

fused_scales_file = weights_dir + '/fused_scales.txt'
fused_zps_file = weights_dir + '/fused_zps.txt'
fused_params_count_file = weights_dir + '/num_of_fused_params.txt'
layers_fused_parameters_offsets_file = weights_dir + '/fused_params_offsets.txt'

layers_fused_parameters_offsets = [0] * (len(model_dag) + 1)

fused_zero_points = []
fused_scales = []
fused_params_so_far = 0

first_conv_layer = True
for layer_index in range(len(model_dag)):

    biases = []
    layer_specs = model_dag[layer_index]
    layer_type = ''

    if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
        layer_type = layer_specs['type']
        if 'activation' in layer_specs and layer_specs['activation'] not in ['', '0']:
            model_activation = layer_specs['activation']
    else:
        continue

    layers_fused_parameters_offsets[layer_index] = fused_params_so_far

    weights_file = weights_file_format.format(layer_index)

    weights = np.loadtxt(weights_files_location +
                         weights_file).astype(np.int8)

    if os.path.isfile(biases_file_format.format(layer_index)):
        with open(biases_file_format.format(layer_index), 'r') as f:
            for line in f:
                bias = line.replace(' ', '').replace('\n', '')
                assert(int(bias) < 2**31-1)
                biases.append(int(bias))
    else:
        print(bcolors.WARNING +
              biases_file_format.format(layer_index) + ' does not exist!!!')

    ifms_zero_point = layer_specs['ifms_zero_points']
    layer_weight_shape = layer_specs['weights_shape']

    if layer_type == 'pw':
        weights = np.reshape(
            weights, (layer_weight_shape[0], layer_weight_shape[1]))
    elif layer_type == 'dw':
        weights = np.reshape(
            weights, (layer_weight_shape[0], layer_weight_shape[1], layer_weight_shape[2]))
    else:
        weights = np.reshape(
            weights, (layer_weight_shape[0], layer_weight_shape[1], layer_weight_shape[2], layer_weight_shape[3]))

    for i in range(layer_weight_shape[0]):
        filter_weights_sum = 0
        if layer_type == 'pw':
            filter_weights_sum = np.sum(weights[i, :])
        elif layer_type == 'dw':
            filter_weights_sum = np.sum(weights[i, :, :])
        else:
            filter_weights_sum = np.sum(weights[i, :, :, :])

        fused_zero_point = filter_weights_sum * -ifms_zero_point \
            + biases[i]
        # if layer_index == 3:
        #     print(filter_weights_sum, ifms_zero_point,
        #         filter_weights_sum * ifms_zero_point, biases[i])
        assert(fused_zero_point < 2 ** 31 -
               1 and fused_zero_point > - 2**31)
        fused_zero_points.append(fused_zero_point)

    fused_params_so_far += layer_weight_shape[0]

    with open(weights_scales_file_format.format(layer_index), 'r') as f:
        ifms_scale = layer_specs['ifms_scales']
        ofms_scale = layer_specs['ofms_scales']
        for line in f:
            weight_scale = float(line.replace(' ', '').replace('\n', ''))
            ifm_weight_fused_scale = weight_scale * ifms_scale
            assert(ifm_weight_fused_scale < 0.5)
            ofm_ifm_weigh_fused_scale = ifm_weight_fused_scale / \
                ofms_scale
            fused_scales.append(ofm_ifm_weigh_fused_scale)
            assert(ofm_ifm_weigh_fused_scale <
                   1) or 'mob_v1' in cgc.MODEL_NAME or 'mob_v2_0_' in cgc.MODEL_NAME \
                or 'uniform' in cgc.MODEL_NAME
            assert(ofm_ifm_weigh_fused_scale > 0)


with open(fused_scales_file, 'w') as f:
    for i in fused_scales:
        f.write(str(i) + '\n')

with open(fused_zps_file, 'w') as f:
    for i in fused_zero_points:
        f.write(str(i) + '\n')

with open(layers_fused_parameters_offsets_file, 'w') as f:
    for i in layers_fused_parameters_offsets:
        f.write(str(i) + '\n')

with open(fused_params_count_file, 'w') as f:
    f.write(str(fused_params_so_far))
