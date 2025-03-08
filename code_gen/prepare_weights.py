import numpy as np
import utils
import code_generation_constants as cgc
import os
import math

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

weights_files_location = '../extract_tflite_model_metadata/{}/weights/'.format(
        cgc.MODEL_NAME)

biases_files_location = '../extract_tflite_model_metadata/{}/biases/'.format(
        cgc.MODEL_NAME)

constant_weights_file = '../const_weights.h'

weights_file_format = weights_files_location + 'weights_{}.txt'
fc_weights_file_format = weights_files_location + 'weights_{}.txt'
fc_biases_file_format = biases_files_location + 'biases_{}.txt'

weights_dir = '../{}_weights'.format(cgc.MODEL_NAME)

print(weights_dir)

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

dst_fc_weights_file = weights_dir + '/fc_weights.txt'
dst_fc_weight_sums_file = weights_dir + '/fc_weight_sums.txt'
dst_fc_biases_file = weights_dir + '/fc_biases.txt'
conv_pw_weights_file = weights_dir + '/conv_pw_weights.txt'
dw_weights_file = weights_dir + '/dw_weights.txt'
conv_pw_weights_offsets_file = weights_dir + '/conv_pw_weights_offsets.txt'
dw_weights_offsets_file = weights_dir + '/dw_weights_offsets.txt'
num_of_conv_pw_weights_file = weights_dir + '/num_of_conv_pw_weights.txt'
num_of_dw_weights_file = weights_dir + '/num_of_dw_weights.txt'

first_layer_weights_def_string = '__constant__ weights_dt first_layer_weights[] = {\n'

model_dag = utils.read_model_dag()

CONSTANT_MEMORY_SIZE = 0  # 32 * 1024  # half of 64

layers_weights = {}
dw_layers_weights = {}
dw_layers_weights_count = 0
pw_layers_weights_count = 0
c_layers_weights = {}
num_s_pw_layers_so_far = 0
num_conv_layers_so_far = 0
fc_layer_index = 0
fc_weights_shape = []
all_pw_s_weights = 0

dw_layers_weights_offsets = [0] * len(model_dag)
pw_layers_weights_offsets = [0] * len(model_dag)

first_layer = True
for layer_index in range(len(model_dag)):
    layer_specs = model_dag[layer_index]

    if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
        num_conv_layers_so_far += 1
        weights_file = weights_file_format.format(layer_index)
        weights = np.loadtxt(weights_file).astype(np.int8)
        if 'type' in layer_specs and (layer_specs['type'] == 'pw' or layer_specs['type'] == 's'):
            all_pw_s_weights += weights.size

            if layer_specs['type'] == 's' and not first_layer:
                ws = layer_specs['weights_shape']
                weights = np.reshape(weights, (ws[0], ws[1], ws[2] * ws[3]))
                weights = np.transpose(weights, (0, 2, 1))
                weights = np.reshape(weights, weights.size)

            if first_layer:
                first_layer = False
                first_layer_weights_def_string += str(
                    weights).replace('[', '').replace(']', '') + '};\n'
            elif all_pw_s_weights < CONSTANT_MEMORY_SIZE:
                c_layers_weights[layer_index] = weights
            else:
                pw_layers_weights_offsets[layer_index] = pw_layers_weights_count
                pw_layers_weights_count += weights.size
                layers_weights[layer_index] = weights
            
            num_s_pw_layers_so_far += 1
        elif layer_specs['type'] == 'dw':
            ws = layer_specs['weights_shape']
            filter_w_h = ws[1] * ws[2]
            # weights = np.reshape(weights, (ws[0], filter_w_h))
            # weights = np.resize(weights, (ws[0], int(
            #     math.pow(
            #         2, math.ceil(math.log2(filter_w_h))
            #     )
            # )
            # )
            # )
            # weights = np.reshape(weights, weights.size)
            padded_weights_size = int(math.pow(
                2, math.ceil(math.log2(filter_w_h))
            ) * weights.size / filter_w_h)

            dw_layers_weights_offsets[layer_index] = dw_layers_weights_count
            dw_layers_weights_count += padded_weights_size
            dw_layers_weights[layer_index] = weights

    elif 'type' in layer_specs and layer_specs['type'] == 'fc':
        fc_layer_index = layer_index
        if 'layer_specs' not in layer_specs:
            print('ISSUE in FC LAYER')
            continue
        fc_weights_shape = layer_specs['weights_shape']
        fc_weights_file = fc_weights_file_format.format(layer_index)
        fc_biases_file = fc_biases_file_format.format(layer_index)
        weights = np.loadtxt(fc_weights_file).astype(np.int8)
        biases = np.loadtxt(fc_biases_file).astype(np.int32)
        np.savetxt(dst_fc_weights_file.format(
            cgc.MODEL_NAME), weights, fmt='%i')
        np.savetxt(dst_fc_biases_file.format(cgc.MODEL_NAME), biases, fmt='%i')
        weights = np.reshape(weights, fc_weights_shape)
        weight_sums = np.sum(weights, axis=1)
        np.savetxt(dst_fc_weight_sums_file.format(
            cgc.MODEL_NAME), weight_sums, fmt='%i')


def write_weights(layers_weights, weights_file, weights_offsets_file, num_of_weights_file, layers_weights_offsets,
                  layers_weights_count, constant_weights):
    layers_indices = list(layers_weights.keys())
    layers_indices.sort()
    combined_weights = []

    constant_weights_def_string = '__constant__ weights_dt c_weights[] = {\n'
    constant_weights_offsets_def_string = '__constant__ int c_weights_offsets[] = {\n'

    for layer_index in layers_indices:
        combined_weights.append(layers_weights[layer_index])

    if constant_weights:
        constant_weights_def_string += \
            str(list(np.concatenate(combined_weights, 0))).replace(
                '[', '').replace(']', '') + '};\n'
        constant_weights_offsets_def_string += str(
            layers_weights_offsets).replace('[', '').replace(']', '') + '};\n'
        with open(weights_file, 'w') as f:
            f.write('#include "cuda_runtime.h"\n')
            f.write('#include "dtype_defs.h"\n\n')
            f.write('#ifndef CONSTANT_WEIGHTS\n')
            f.write('#define CONSTANT_WEIGHTS\n\n')
            f.write(constant_weights_def_string)
            f.write(constant_weights_offsets_def_string)
            f.write('\n#endif\n')

    else:
        np.savetxt(weights_file.format(cgc.MODEL_NAME),
                   np.concatenate(combined_weights, 0), fmt='%i')
        np.savetxt(weights_offsets_file.format(cgc.MODEL_NAME),
                   np.array(layers_weights_offsets), fmt='%i')
        with open(num_of_weights_file.format(cgc.MODEL_NAME), 'w') as f:
            if dw_layers_weights_count != 0:
                f.write(str(layers_weights_count))


write_weights(layers_weights, conv_pw_weights_file,
              conv_pw_weights_offsets_file, num_of_conv_pw_weights_file, pw_layers_weights_offsets, pw_layers_weights_count, False)

if utils.has_dw_layers(model_dag):
    write_weights(dw_layers_weights, dw_weights_file,
                  dw_weights_offsets_file, num_of_dw_weights_file, dw_layers_weights_offsets, dw_layers_weights_count, False)

if CONSTANT_MEMORY_SIZE != 0:
    write_weights(c_layers_weights, constant_weights_file,
                  '', '', dw_layers_weights_count, True)
