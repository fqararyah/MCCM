import math
import utils
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

# './out/layers_specs.h'
out_file_h_name = '{}_layers_specs.h'
out_file_h = '../model_specs/' + out_file_h_name
out_file_cpp = '../model_specs/{}_layers_specs.cpp'

specs_struct_seq = '    layer_specs_seq[{}].layer_index = {};\n\
    layer_specs_seq[{}].conv_layer_type = {}; \n\
    layer_specs_seq[{}].layer_num_fils = {};\n\
    layer_specs_seq[{}].strides = {};\n\
    layer_specs_seq[{}].filter_size = {};\n\
    layer_specs_seq[{}].padding_left = {};\n\
    layer_specs_seq[{}].padding_right = {};\n\
    layer_specs_seq[{}].padding_top = {};\n\
    layer_specs_seq[{}].padding_bottom = {};\n\
    layer_specs_seq[{}].layer_depth = {};\n\
    layer_specs_seq[{}].layer_ifm_height = {};\n\
    layer_specs_seq[{}].layer_ifm_width = {};\n\
    layer_specs_seq[{}].layer_ofm_height = {};\n\
    layer_specs_seq[{}].layer_ofm_width = {};\n\
    layer_specs_seq[{}].layer_activation = {};\n\
    layer_specs_seq[{}].layer_num_of_ifm_tiles_h = {};\n\
    layer_specs_seq[{}].layer_num_of_ifm_tiles_w = {};\n\
    layer_specs_seq[{}].layer_num_of_ofm_tiles_h = {};\n\
    layer_specs_seq[{}].layer_num_of_ofm_tiles_w = {};\n\
    layer_specs_seq[{}].layer_weights_offset = {};\n\
    layer_specs_seq[{}].write_to_result_or_channels = {};\n\
    layer_specs_seq[{}].write_to_tmp = {};\n\
    layer_specs_seq[{}].followed_by = {};\n\
    layer_specs_seq[{}].layer_ifms_zero_point = {};\n\
    layer_specs_seq[{}].layer_ofms_scale = {};\n\
    layer_specs_seq[{}].relu_threshold = {} ;\n\
    layer_specs_seq[{}].layer_ofms_zero_point = {};\n\
    layer_specs_seq[{}].add_layer_scale_reciprocal = {};\n\
    layer_specs_seq[{}].add_layer_zero_point = {};\n\
    layer_specs_seq[{}].skip_connection_other_layer_scale = {};\n\
    layer_specs_seq[{}].skip_connection_other_layer_zero_point = {};\n\n\
    layer_specs_seq[{}].data_layout = {};\n\n'


pooling_specs_struct_seq = 'layer_specs_seq[{}].layer_index = {};\n\
                pooling_layer_specs_seq[{}].ifm_depth = {};\n\
                pooling_layer_specs_seq[{}].ifm_height = {};\n\
                pooling_layer_specs_seq[{}].ifm_width = {};\n\
                pooling_layer_specs_seq[{}].ofm_depth = {};\n\
                pooling_layer_specs_seq[{}].ofm_height = {};\n\
                pooling_layer_specs_seq[{}].ofm_width = {};\n\
                pooling_layer_specs_seq[{}].full_hw = {};\n\
                pooling_layer_specs_seq[{}].fused_scale = {};\n\
                pooling_layer_specs_seq[{}].ifms_zero_point = {};\n\
                pooling_layer_specs_seq[{}].ofms_zero_point = {};\n\n'

fc_specs_struct = 'const fc_layer_specs layer_{}_specs = {}\n\
                {},//const fms_dt ifm_zero_point\n\
                {};\n'

specs_block = "//****************************\n \
const int layer_{}_{}_num_fils = {};\n\
const int layer_{}_{}_depth = {};\n\
const int layer_{}_{}_filter_dim = {};\n \
const int layer_{}_{}_ifm_width = {};\n \
//****************************\n"

first_layer_specs_block = "//****************************\n \
const int first_conv_layer_num_fils = {};\n\
const int first_conv_layer_depth = {};\n\
const int first_conv_layer_filter_dim = {};\n \
const int first_conv_layer_strides = {};\n \
const int first_conv_layer_padding_left = {};\n \
const int first_conv_layer_padding_right = {};\n \
const int first_conv_layer_ifm_width = {};\n \
//****************************\n"

max_layer_d_and_w = (0, 0)

weights_dir = '../{}_weights'.format(cgc.MODEL_NAME)
conv_pw_weights_offsets_file = weights_dir + '/conv_pw_weights_offsets.txt'
dw_weights_offsets_file = weights_dir + '/dw_weights_offsets.txt'

conv_pw_weights_offsets = []
dw_weights_offsets = []
model_dag = utils.read_model_dag()

with open(conv_pw_weights_offsets_file, 'r') as f:
    for line in f:
        conv_pw_weights_offsets.append(int(line))

if utils.has_dw_layers(model_dag):
    with open(dw_weights_offsets_file, 'r') as f:
        for line in f:
            dw_weights_offsets.append(int(line))

model_configs_list = [0] * 2 * len(model_dag)
current_block_indx = 0
cumulative_s_pw_weights = 0
cumulative_s_pw_weights_on_chip = 0
cumulative_dw_weights = 0
dw_ifms_cumulative_width_offset = 0
num_conv_layers_so_far = 0
first_conv_layer = True

to_write_specs_block = ''
conv_layers_indices = [0] * len(model_dag)
conv_layers_indices_declaration_str = 'const int conv_layers_indices[{}] = {}'

with open(out_file_cpp.format(cgc.MODEL_NAME), 'w') as f:
    f.write('#include "./model_specs/layers_specs.h"\n\n')

    f.write("#if MODEL_ID == {} \n\n".format(cgc.MODEL_NAME.upper()))
    f.write("void layer_specs_init(layer_specs *layer_specs_seq, pooling_layer_specs * pooling_layer_specs_seq)\n{\n")

    for layer_index in range(len(model_dag)):
        layer_specs = model_dag[layer_index]
        layer_type = ''
        replacement_list = []
        if 'type' in layer_specs:
            layer_type = layer_specs['type']
        if layer_type in cgc.CONV_LAYER_TYPES:
            conv_layers_indices[layer_index] = 1
            num_conv_layers_so_far += 1
            layer_weights_shape = layer_specs['weights_shape']
            layer_weights_size = 1
            for i in layer_weights_shape:
                layer_weights_size *= i
            layer_filter_dim = 1
            layer_ifms_depth = layer_weights_shape[1]
            layer_num_fils = layer_weights_shape[0]
            if layer_type != 'pw':
                layer_filter_dim = layer_weights_shape[-1]
            if layer_type == 'dw':
                layer_ifms_depth = layer_num_fils

            # replacement_dic['*PREV*'] = layers_types[i-1]
            strides = layer_specs['strides']
            filter_dim = layer_filter_dim
            num_of_filters = layer_num_fils

            replacement_list.append(layer_index)
            replacement_list.append(layer_index)

            replacement_list.append(layer_index)
            if layer_type == 'pw':
                replacement_list.append('PW_CONV')
            elif layer_type == 'dw':
                replacement_list.append('DW_CONV')
            else:
                replacement_list.append('S_CONV')

            replacement_list.append(layer_index)
            replacement_list.append(num_of_filters)

            replacement_list.append(layer_index)
            replacement_list.append(strides)

            replacement_list.append(layer_index)
            replacement_list.append(filter_dim)

            padding_left = 0
            padding_right = 0
            padding_top = 0
            padding_bottom = 0
            if layer_type != 'pw':
                padding = int(filter_dim - strides)
                if strides == 1:
                    padding_left = int(padding / 2)
                    padding_right = int(padding / 2)
                    padding_top = int(padding / 2)
                    padding_bottom = int(padding / 2)
                else:
                    padding_right = padding
                    padding_bottom = padding

            replacement_list.append(layer_index)
            replacement_list.append(padding_left)

            replacement_list.append(layer_index)
            replacement_list.append(padding_right)

            replacement_list.append(layer_index)
            replacement_list.append(padding_top)

            replacement_list.append(layer_index)
            replacement_list.append(padding_bottom)

            layer_ifms_shape = layer_specs['ifms_shape']
            layer_depth = layer_ifms_shape[0]
            layer_height = layer_ifms_shape[1]
            layer_width = layer_ifms_shape[2]

            if model_dag[layer_index - 1]['name'] == 'pad':
                layer_width -= padding_right
                layer_height -= padding_bottom

            replacement_list.append(layer_index)
            replacement_list.append(layer_depth)

            replacement_list.append(layer_index)
            replacement_list.append(layer_height)

            replacement_list.append(layer_index)
            replacement_list.append(layer_width)

            layer_ofm_height = int(layer_height / strides)
            replacement_list.append(layer_index)
            replacement_list.append(layer_ofm_height)

            layer_ofm_width = int(layer_width / strides)
            replacement_list.append(layer_index)
            replacement_list.append(layer_ofm_width)

            if layer_ofm_width * layer_num_fils >= max_layer_d_and_w[0] * max_layer_d_and_w[1]:
                max_layer_d_and_w = (layer_ofm_width, layer_num_fils)

            layer_activation = ''
            layer_activation_val = -1
            if layer_specs['activation'] != '':
                layer_activation = layer_specs['activation']
                if layer_activation == 'relu6' or layer_activation == 'RELU6':
                    layer_activation_val = 6
            else:
                layer_activation = '0'

            replacement_list.append(layer_index)
            replacement_list.append(layer_activation)

            replacement_list.append(layer_index)
            replacement_list.append(
                '(' + str(layer_height) + ' + TILE_H - 1) / TILE_H')

            replacement_list.append(layer_index)
            replacement_list.append(
                '(' + str(layer_width) + ' + TILE_W - 1) / TILE_W')

            replacement_list.append(layer_index)
            replacement_list.append(
                '(' + str(int(layer_height / strides)) + ' + TILE_H - 1) / TILE_H')

            replacement_list.append(
                '(' + str(int(layer_width / strides)) + ' + TILE_W - 1) / TILE_W')
            replacement_list.append(layer_index)

            layer_weights_offset = 0
            if (layer_type == 'pw' or layer_type == 's'):
                layer_weights_offset = conv_pw_weights_offsets[layer_index]
            elif layer_type == 'dw':
                layer_weights_offset = dw_weights_offsets[layer_index]

            replacement_list.append(layer_index)
            replacement_list.append(layer_weights_offset)

            write_to_tmp = 0
            write_to_result_or_channels = 1
            followed_by = 0
            add_layer_scale_reciprocal = 1
            add_layer_zero_point = 0
            skip_connection_other_layer_scale = 1
            skip_connection_other_layer_zero_point = 0

            layer_children = layer_specs['children']
            for i in range(len(layer_children)):
                if layer_children[i] - i != layer_index + 1:
                    write_to_tmp = 1

            if model_dag[layer_children[0]]['name'] == 'add' and model_dag[layer_children[0]]['id'] == layer_index + 1:
                add_layer_specs = model_dag[layer_children[0]]
                the_other_conv_layer_specs = model_dag[add_layer_specs['parents'][0]]
                followed_by = cgc.ADD_LAYER_ID_IN_CPP
                add_layer_scale_reciprocal = 1 / add_layer_specs['ofms_scales']
                add_layer_zero_point = add_layer_specs['ofms_zero_points']
                skip_connection_other_layer_scale = the_other_conv_layer_specs['ofms_scales']
                skip_connection_other_layer_zero_point = the_other_conv_layer_specs[
                    'ofms_zero_points']
                if len(add_layer_specs['children']) > 1:
                    # if the fused add layer is a beginning of a branch
                    write_to_tmp = 1
            
            if 'type' in model_dag[layer_children[0]] and model_dag[layer_children[0]]['type'] == 'avgpool':
                followed_by = cgc.AVG_POOL_LAYER_ID_IN_CPP

            if write_to_tmp == 1 and followed_by == 0 and len(layer_children) <= 1:
                write_to_result_or_channels = 0

            replacement_list.append(layer_index)
            replacement_list.append(write_to_result_or_channels)

            replacement_list.append(layer_index)
            replacement_list.append(write_to_tmp)

            replacement_list.append(layer_index)
            replacement_list.append(followed_by)

            replacement_list.append(layer_index)
            replacement_list.append(layer_specs['ifms_zero_points'])

            replacement_list.append(layer_index)
            replacement_list.append(layer_specs['ofms_scales'])

            assert(layer_activation_val == -1 or
                   round(layer_activation_val / layer_specs['ofms_scales']) > 100)

            replacement_list.append(layer_index)
            replacement_list.append(round(layer_activation_val / layer_specs['ofms_scales']) if
                                    layer_activation_val != -1 else 0)  # relu_threshold

            replacement_list.append(layer_index)
            replacement_list.append(layer_specs['ofms_zero_points'])

            replacement_list.append(layer_index)
            replacement_list.append(add_layer_scale_reciprocal)

            replacement_list.append(layer_index)
            replacement_list.append(add_layer_zero_point)

            replacement_list.append(layer_index)
            replacement_list.append(skip_connection_other_layer_scale)

            replacement_list.append(layer_index)
            replacement_list.append(skip_connection_other_layer_zero_point)
            
            layer_data_layout = cgc.DEFAULT_DATA_LAYOUT
            if cgc.MODEL_NAME in cgc.MODELS_HWC_LAYERS and layer_index in cgc.MODELS_HWC_LAYERS[cgc.MODEL_NAME]:
                layer_data_layout = 'HWC'

            replacement_list.append(layer_index)
            replacement_list.append(layer_data_layout)

            if current_block_indx > 0:
                to_write_specs_block += specs_block.format(layer_index, layer_type, layer_num_fils,
                                                           layer_index, layer_type, layer_ifms_depth,
                                                           layer_index, layer_type, layer_filter_dim,
                                                           layer_index, layer_type, layer_width)
            else:
                current_block_indx += 1
                to_write_specs_block += first_layer_specs_block.format(layer_num_fils, layer_ifms_depth, layer_filter_dim,
                                                                       strides, padding_left, padding_right, layer_width)
            f.write(specs_struct_seq.format(*replacement_list))

            model_configs_list[2 * layer_index] = layer_depth
            model_configs_list[2 * layer_index + 1] = num_of_filters

            first_conv_layer = False

        elif 'type' in layer_specs:
            layer_type = layer_specs['type']
            layer_specs_struct_str = '\nstruct{}\n{}{}{};\n'
            struct_var_name = 'layer_' + \
                str(layer_specs['id']) + '_' + layer_type + '_specs'
            struct_var_body = ''

            if layer_type == 'avgpool':
                replacement_list.append(str(layer_index))
                replacement_list.append(str(layer_index))
                layer_ifms_shape = layer_specs['ifms_shape']
                ifms_depth = layer_ifms_shape[-1]
                ifm_height = layer_ifms_shape[-3]
                ifm_width = layer_ifms_shape[-2]
                replacement_list.append(str(layer_index))
                replacement_list.append(ifms_depth)
                replacement_list.append(str(layer_index))
                replacement_list.append(ifm_height)
                replacement_list.append(str(layer_index))
                replacement_list.append(ifm_width)
                layer_ofms_shape = layer_specs['ofms_shape']
                ofms_depth = layer_ofms_shape[-1]
                if len(layer_ofms_shape) >= 4:#the fourth is batch
                    ofm_height = layer_ofms_shape[-3]
                else:
                    ofm_height = 1
                if len(layer_ofms_shape) >= 4:
                    ofm_width = layer_ofms_shape[-2]
                else:
                    ofm_width = 1
                replacement_list.append(layer_index)
                replacement_list.append(ofms_depth)
                replacement_list.append(layer_index)
                replacement_list.append(ofm_height)
                replacement_list.append(layer_index)
                replacement_list.append(ofm_width)
                parent_layer_specs = model_dag[layer_specs['parents'][0]]
                pooling_ofms_scale = layer_specs['ofms_scales']
                pooling_ifms_scale = parent_layer_specs['ofms_scales']

                replacement_list.append(layer_index)
                if ofm_height == 1 and ofm_width == 1:
                    replacement_list.append('true')
                else:
                    replacement_list.append('false')

                replacement_list.append(layer_index)
                replacement_list.append(pooling_ifms_scale / pooling_ofms_scale)
                replacement_list.append(layer_index)
                replacement_list.append(layer_specs['ifms_zero_points'])
                replacement_list.append(layer_index)
                replacement_list.append(layer_specs['ofms_zero_points'])
                f.write(pooling_specs_struct_seq.format(*replacement_list))

            elif layer_type == 'fc':
                replacement_list.append(str(layer_index) + '_' + layer_type)
                replacement_list.append('{')
                replacement_list.append(layer_specs['ifms_zero_points'])
                replacement_list.append('}')
                to_write_specs_block += (fc_specs_struct.format(*
                                         replacement_list))

        f.write("\n")

    f.write("}\n")
    f.write("#endif\n")


max_layer_d_and_w = (2 ** math.ceil(math.log2(max_layer_d_and_w[0])),
                     2 ** math.ceil(math.log2(max_layer_d_and_w[1]))
                     )

with open(out_file_h.format(cgc.MODEL_NAME), 'w') as f:
    f.write("#ifndef LAYERS_SPECS\n")
    f.write("#define LAYERS_SPECS\n\n")
    f.write("const int MODEL_NUM_LAYERS = " + str(len(model_dag)) + ";\n")
    f.write("const int MAX_LAYER_DW = " +
            str(max_layer_d_and_w[0]) + ' * ' + str(max_layer_d_and_w[1]) + ";\n")
    f.write(conv_layers_indices_declaration_str.format(len(model_dag),
                                                       str(conv_layers_indices).replace('[', '{').replace(']', '}')) + ';\n')
    f.write(to_write_specs_block)

    f.write('#endif\n')
