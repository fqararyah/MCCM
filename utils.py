import classes
from multiprocessing.dummy import active_children
import sys
import pathlib
import json
import math

current_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(current_dir)
print(current_dir)

DELIMITER = '::'

NET_PREFIX = 'mob_v2'
NET_FULL_NAME = 'mobilenet_v2'
input_folder = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/models_archs/models/'\
    + NET_FULL_NAME + '/'
IFMS_FILE = input_folder + 'layers_inputs.txt'
OFMS_FILE = input_folder + 'layers_outputs.txt'
LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
SECONDARY_LAYERS_TYPES_FILE = input_folder + 'secondary_layers_types.txt'
LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'
LAYERS_SKIP_CONNECTIONS_FILE = input_folder + 'skip_connections_indices.txt'
LAYERS_ACTIVATIONS_FILE = input_folder + 'layers_activations.txt'
LAYERS_EXECUTION_SEQUENCE = input_folder + 'layers_execution_sequence.txt'
MODEL_DAG_FILE = input_folder + 'model_dag.json'


def set_globals(prefix, full_name):
    global NET_PREFIX, NET_FULL_NAME, input_folder, IFMS_FILE, OFMS_FILE, LAYERS_TYPES_FILE, LAYERS_WEIGHTS_FILE, LAYERS_STRIDES_FILE, EXPANSION_PROJECTION_FILE, LAYERS_RELUS_FILE, LAYERS_SKIP_CONNECTIONS_FILE, SECONDARY_LAYERS_TYPES_FILE, LAYERS_ACTIVATIONS_FILE, \
        LAYERS_EXECUTION_SEQUENCE, MODEL_DAG_FILE
    NET_PREFIX = prefix
    NET_FULL_NAME = full_name
    input_folder = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/models_archs/models/'\
        + NET_FULL_NAME + '/'
    IFMS_FILE = input_folder + 'layers_inputs.txt'
    OFMS_FILE = input_folder + 'layers_outputs.txt'
    LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
    SECONDARY_LAYERS_TYPES_FILE = input_folder + 'secondary_layers_types.txt'
    LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
    LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
    EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
    LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'
    LAYERS_ACTIVATIONS_FILE = input_folder + 'layers_activations.txt'
    LAYERS_SKIP_CONNECTIONS_FILE = input_folder + 'skip_connections_indices.txt'
    LAYERS_EXECUTION_SEQUENCE = input_folder + 'layers_execution_sequence.txt'
    MODEL_DAG_FILE = input_folder + 'model_dag.json'


def clean_line(line):
    return line.replace(' ', '').replace('\n', '').replace('\n', '').replace('\t', '')


def clean_line_keep_spaces(line):
    return line.replace('\r', '').replace('\n', '').replace('\t', '')


def read_layers_input_shapes():
    layers_inputs = []
    with open(IFMS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                if len(splits) == 3:
                    layers_inputs.append(classes.feature_map(
                        int(splits[0]), int(splits[1]), int(splits[2])))
                elif len(splits) == 1:
                    layers_inputs.append(classes.feature_map(
                        int(splits[0]), 1, 1))

    return layers_inputs


def read_layers_output_shapes():
    layers_outputs = []
    with open(OFMS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                splits = line.split('x')
                if len(splits) == 3:
                    layers_outputs.append(classes.feature_map(
                        int(splits[0]), int(splits[1]), int(splits[2])))
                elif len(splits) == 1:
                    layers_outputs.append(classes.feature_map(
                        int(splits[0]), 1, 1))

    return layers_outputs


def read_layers_weight_shapes(layers_types):
    layers_weights = []
    count = 0
    with open(LAYERS_WEIGHTS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                if layers_types[count] == 'dw':
                    layers_weights.append(classes.weights(int(splits[0]), 1,
                                                          int(splits[1]) if len(splits) > 1 else 1, int(splits[2]) if len(splits) > 2 else 1))
                else:
                    layers_weights.append(classes.weights(int(splits[0]), int(splits[1]),
                                                          int(splits[2]) if len(splits) > 2 else 1, int(splits[3]) if len(splits) > 3 else 1))
                count += 1

    return layers_weights


def read_layers_strides():
    layers_strides = []
    with open(LAYERS_STRIDES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_strides.append(int(line))

    return layers_strides


def read_layers_types():
    layers_types = []
    with open(LAYERS_TYPES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_types.append(line)

    return layers_types


def read_secondary_layers_types():
    layers_types = []
    with open(SECONDARY_LAYERS_TYPES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_types.append(line)

    return layers_types


def read_expansion_projection():
    expansion_projection = []
    with open(EXPANSION_PROJECTION_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            expansion_projection.append(int(line))

    return expansion_projection


def read_layers_relus():
    layers_relus = []
    with open(LAYERS_RELUS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_relus.append(int(line))

    return layers_relus


def read_layers_activations():
    layers_activations = []
    with open(LAYERS_ACTIVATIONS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_activations.append(line.replace('\n', '').replace(' ', ''))

    return layers_activations


def read_skip_connections_indices():
    skip_connections_indices = {}
    with open(LAYERS_SKIP_CONNECTIONS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            skip_connections_indices[int(line)] = 1

    return skip_connections_indices


def read_layers_execution_sequence():
    layers_execution_sequence = []
    with open(LAYERS_EXECUTION_SEQUENCE, 'r') as f:
        for line in f:
            layers_execution_sequence.append(
                line.replace('\n', '').replace(' ', ''))

    return layers_execution_sequence


def read_model_dag():
    f = open(MODEL_DAG_FILE)
    return json.load(f)


def read_model_dag_v2(dag_file):
    f = open(dag_file)
    return json.load(f)


def is_padding_layer(layer_specs):
    return 'name' in layer_specs and layer_specs['name'] == 'pad'


def get_filter_dim(layer_specs):
    if is_pw_layer(layer_specs):
        return 1
    layer_weights_shape = layer_specs['weights_shape']
    return layer_weights_shape[-1]


def get_strides(layer_specs):
    if is_conv_layer(layer_specs):
        return layer_specs['strides']
    return 1


def get_layer_single_filter_size(layer_specs):
    layer_weights_shape = layer_specs['weights_shape']
    if is_pw_layer(layer_specs):
        return layer_weights_shape[1]
    if is_pw_layer(layer_specs):
        return layer_weights_shape[1]
    elif is_dw_layer(layer_specs):
        return layer_weights_shape[-2] * layer_weights_shape[-1]
    else:
        return layer_weights_shape[1] * layer_weights_shape[2] * layer_weights_shape[3]


def get_layer_paddings(layer_specs, prev_layer_specs):
    paddings = [0, 0, 0, 0]
    if is_pw_layer(layer_specs):
        return paddings

    strides = layer_specs['strides']
    if is_padding_layer(prev_layer_specs):
        return [prev_layer_specs['padding_top'], prev_layer_specs['padding_right'], prev_layer_specs['padding_bottom'], prev_layer_specs['padding_left']]
    else:
        padding = get_filter_dim(layer_specs) - strides
        if strides == 1:
            all_paddings = int(padding / 2)
            return [all_paddings, all_paddings, all_paddings, all_paddings]
        else:
            return [0, padding, padding, 0]


def get_conv_layer_index_from_offset(model_dag, anchor_layer_index, layer_offset, layer_type=''):

    num_conv_layers = 0
    if layer_offset >= 0:
        layer_index = anchor_layer_index
        while num_conv_layers <= layer_offset and layer_index < len(model_dag):
            layer_specs = model_dag[layer_index]
            if is_conv_layer(layer_specs) and (layer_type == '' or layer_specs['type']):
                num_conv_layers += 1

            layer_index += 1

        layer_index -= 1

    else:
        layer_offset = abs(layer_offset)
        layer_index = anchor_layer_index - 1
        while num_conv_layers < layer_offset and layer_index > 0:
            layer_specs = model_dag[layer_index]
            if is_conv_layer(layer_specs) and (layer_type == '' or layer_specs['type']):
                num_conv_layers += 1

            layer_index -= 1

        if num_conv_layers != layer_offset:
            return -1
        layer_index += 1

    return layer_index


def get_num_conv_layer_count_in_range(model_dag, from_layer_index, to_layer_index):

    num_conv_layers = 0
    for i in range(from_layer_index, to_layer_index + 1):
        if i >= len(model_dag):
            break
        if is_conv_layer(model_dag[i]):
            num_conv_layers += 1

    return num_conv_layers


def get_conv_layer_indices_in_range(model_dag, from_layer_index, num_layers):

    conv_layers_indices = []
    i = from_layer_index
    conv_layers_so_far = 0
    while i < len(model_dag) and conv_layers_so_far < num_layers:
        if is_conv_layer(model_dag[i]):
            conv_layers_so_far += 1
            conv_layers_indices.append(i)
        i += 1

    return conv_layers_indices


def get_conv_layer_indices_in_range_inc(model_dag, from_layer_index, to_layer_index):

    conv_layers_indices = []
    i = from_layer_index
    conv_layers_so_far = 0
    while i < len(model_dag) and i <= to_layer_index:
        if is_conv_layer(model_dag[i]):
            conv_layers_so_far += 1
            conv_layers_indices.append(i)
        i += 1

    return conv_layers_indices


def layer_index_to_conv_rank_mapping(model_dag):
    current_conv_layer_rank = 0
    index_to_rank_map = {}
    for i in range(len(model_dag)):
        if is_conv_layer(model_dag[i]):
            index_to_rank_map[i] = current_conv_layer_rank
            current_conv_layer_rank += 1

    return index_to_rank_map


def is_conv_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['s', 'pw', 'dw']


def is_fc_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['fc']


def is_dw_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['dw']


def is_pw_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['pw']


def has_dw_layers(model_dag, starting_layer_index=0, num_layers=1000):
    current_layer_index = starting_layer_index
    conv_layers_so_far = 0
    while current_layer_index < len(model_dag) and conv_layers_so_far < num_layers:
        if is_dw_layer(model_dag[current_layer_index]):
            return True
        if is_conv_layer(model_dag[current_layer_index]):
            conv_layers_so_far += 1
        current_layer_index += 1
    return False


def get_fms_sizes(model_dag):

    fms_sizes = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            fms_sizes.append(layer_specs['ifms_shape'][0] * layer_specs['ifms_shape'][1] * layer_specs['ifms_shape'][2] +
                             layer_specs['ofms_shape'][0] * layer_specs['ofms_shape'][1] * layer_specs['ofms_shape'][2])

    return fms_sizes

def get_fms_sizes_by_indices(model_dag, layer_indices):

    fms_sizes = []
    for layer_index in layer_indices:
        layer_specs = model_dag[layer_index]
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            fms_sizes.append(layer_specs['ifms_shape'][0] * layer_specs['ifms_shape'][1] * layer_specs['ifms_shape'][2] +
                             layer_specs['ofms_shape'][0] * layer_specs['ofms_shape'][1] * layer_specs['ofms_shape'][2])

    return fms_sizes

def get_weights_sizes(model_dag):

    weights_sizes = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            weights_sizes.append(get_layer_weights_size(layer_specs))

    return weights_sizes

def get_weights_sizes_by_indices(model_dag, layer_indices):

    weights_sizes = []
    for layer_index in layer_indices:
        layer_specs = model_dag[layer_index]
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            weights_sizes.append(get_layer_weights_size(layer_specs))

    return weights_sizes


def get_layer_weights_size(layer_specs):
    weights_size = 0
    if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
        weights_shape = layer_specs['weights_shape']
        if layer_specs['type'] in ['s']:
            return (
                weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3])
        elif layer_specs['type'] in ['dw']:
            return (
                weights_shape[0] * weights_shape[1] * weights_shape[2])
        elif layer_specs['type'] in ['pw']:
            return (weights_shape[0] * weights_shape[1])


def get_layer_weights_shape(layer_specs):
    return layer_specs['weights_shape']


def get_layer_ifms_size(layer_specs):
    return layer_specs['ifms_shape'][0] * layer_specs['ifms_shape'][1] * layer_specs['ifms_shape'][2]


def get_layer_ifms_shape(layer_specs):
    return [layer_specs['ifms_shape'][0], layer_specs['ifms_shape'][1], layer_specs['ifms_shape'][2]]


def get_layer_ofms_size(layer_specs):
    ofms_shape = layer_specs['ofms_shape']
    ofms_size = 1
    for dim in ofms_shape:
        ofms_size *= dim

    return ofms_size


def get_layer_ofms_shape(layer_specs):
    return [layer_specs['ofms_shape'][0], layer_specs['ofms_shape'][1], layer_specs['ofms_shape'][2]]


def get_ifms_sizes(model_dag):

    fms_sizes = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            fms_sizes.append(get_layer_ifms_size(layer_specs))

    return fms_sizes


def get_ofms_sizes(model_dag):

    fms_sizes = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            fms_sizes.append(get_layer_ofms_size(layer_specs))

    return fms_sizes


def get_ofms_sizes_with_fusion(model_dag):
    fms_sizes = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            if is_fused_with_pooling(model_dag, layer_specs):
                fms_sizes.append(get_layer_ofms_size(
                    model_dag[layer_specs['children'][0]]))
            else:
                fms_sizes.append(get_layer_ofms_size(layer_specs))

    return fms_sizes


def get_layer_children(layer_specs):
    return layer_specs['children']


def get_layer_children_with_fusion(model_dag, layer_specs):
    layer_chidren = layer_specs['children']
    if len(layer_chidren) == 1 and is_fused_with_add(model_dag, layer_specs):
        layer_chidren = get_layer_children(
            model_dag[layer_chidren[0]])

    return layer_chidren


def get_layer_num_children(layer_specs):
    return len(layer_specs['children'])


def get_max_tmp_channels(model_dag, starting_layer=0):

    for i in range(starting_layer, len(model_dag)):
        layer_specs = model_dag[i]
        if is_conv_layer(layer_specs):
            if len(layer_specs['children']) > 1:
                return layer_specs['ofms_shape'][0] * layer_specs['ofms_shape'][1] * layer_specs['ofms_shape'][2]

    return 0


def get_layers_op_counts(model_dag, starting_offset=0):

    layers_num_of_ops = []
    conv_layers_so_far = 0
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            if conv_layers_so_far < starting_offset:
                conv_layers_so_far += 1
                continue
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']

            layer_num_of_ops = 1
            for i in layers_weights_shape:
                layer_num_of_ops *= i

            layer_num_of_ops *= layers_ofms_shape[1] * layers_ofms_shape[2]

            layers_num_of_ops.append(layer_num_of_ops)
            # if is_fused_with_add(model_dag, layer_specs):
            #    layers_num_of_ops[-1] += layers_ofms_shape[0] * layers_ofms_shape[1] * layers_ofms_shape[2]

    return layers_num_of_ops


def get_layer_type_sequence(model_dag, starting_offset=0):
    layer_types = []
    conv_layers_so_far = 0
    for layer_specs in model_dag:
        if is_conv_layer(layer_specs):
            if conv_layers_so_far < starting_offset:
                conv_layers_so_far += 1
                continue
            layer_types.append(layer_specs['type'])

    return layer_types


def get_dw_layers_op_counts(model_dag, starting_offset=0):
    conv_layers_so_far = 0
    layers_num_of_ops = []
    for layer_specs in model_dag:
        if is_conv_layer(layer_specs) and conv_layers_so_far < starting_offset:
            conv_layers_so_far += 1
            continue
        if 'type' in layer_specs and layer_specs['type'] in ['dw']:
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']

            layer_num_of_ops = 1
            for i in layers_weights_shape:
                layer_num_of_ops *= i

            layer_num_of_ops *= layers_ofms_shape[1] * layers_ofms_shape[2]

            layers_num_of_ops.append(layer_num_of_ops)
            # if is_fused_with_add(model_dag, layer_specs):
            #    layers_num_of_ops[-1] += layers_ofms_shape[0] * layers_ofms_shape[1] * layers_ofms_shape[2]

    return layers_num_of_ops


def get_layers_op_counts_by_indices(model_dag, layer_indices):

    layers_num_of_ops = []
    for i in range(len(layer_indices)):
        layer_specs = model_dag[layer_indices[i]]
        if is_conv_layer(layer_specs):
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']

            layer_num_of_ops = 1
            for i in layers_weights_shape:
                layer_num_of_ops *= i

            layer_num_of_ops *= layers_ofms_shape[1] * layers_ofms_shape[2]

            layers_num_of_ops.append(layer_num_of_ops)
            # if is_fused_with_add(model_dag, layer_specs):
            #    layers_num_of_ops[-1] += layers_ofms_shape[0] * layers_ofms_shape[1] * layers_ofms_shape[2]

    return layers_num_of_ops


def sum_layer_ops_in_range(layer_ops, first_layer, last_layer):
    sum_ops = 0
    for i in range(first_layer, last_layer + 1):
        sum_ops += layer_ops[i]

    return sum_ops


def get_last_conv_layer(model_dag):
    last_conv_layer_index = len(model_dag) - 1
    while last_conv_layer_index >= 0:
        l_specs = model_dag[last_conv_layer_index]
        if is_conv_layer(l_specs):
            return l_specs
        last_conv_layer_index -= 1

    return -1


def get_skip_connection_early_layer(model_dag, layer_specs):
    layer_children = layer_specs['children']
    layer_index = layer_specs['id']
    if model_dag[layer_children[0]]['name'] in ['add', 'concat'] and model_dag[layer_children[0]]['id'] == layer_index + 1:
        add_layer_specs = model_dag[layer_children[0]]
        return model_dag[add_layer_specs['parents'][0]]

    return -1


def get_layer_index(layer_specs):
    return layer_specs['id']


def is_fused_with_add(model_dag, layer_specs):
    layer_children = layer_specs['children']
    layer_index = layer_specs['id']
    return model_dag[layer_children[0]]['name'] == 'add' and model_dag[layer_children[0]]['id'] == layer_index + 1


def is_fused_with_pooling(model_dag, layer_specs):
    layer_children = layer_specs['children']
    layer_index = layer_specs['id']
    return 'pool' in model_dag[layer_children[0]]['name'] and model_dag[layer_children[0]]['id'] == layer_index + 1


def is_pooling_layer(layer_specs):
    return 'pool' in layer_specs['name'] or ('type' in layer_specs and 'pool' in layer_specs['type'])


def pow_of_2_geq(num):
    return int(math.pow(2, int(math.ceil(math.log2(num)))))


def pow_of_2_leq(num):
    return int(math.pow(2, int(math.floor(math.log2(num)))))


def closest_pow_of_2(num):
    leq_pow_2 = pow_of_2_leq(num)
    geq_pow_2 = pow_of_2_geq(num)
    if (num - leq_pow_2 < geq_pow_2 - num):
        return leq_pow_2
    return geq_pow_2


def read_int_list_from_line(line):
    result_list = []
    line = clean_line(line)
    line = line.replace('[', '').replace(']', '')
    splits = line.split(',')
    for split in splits:
        if split.isdigit():
            result_list.append(int(split))
        elif split.lstrip('-').isdigit():
            result_list.append(int(split))

    return result_list


def read_bool_list_from_line(line):
    result_list = []
    line = clean_line(line)
    line = line.replace('[', '').replace(']', '')
    splits = line.split(',')
    for split in splits:
        result_list.append(split == 'True' or split ==
                           'true' or split == 'T' or split == 't')

    return result_list


def get_unique_pw_and_conv_layers_and_frequencies(model_dag, starting_layer=0):
    unique_layers = {}
    uniqueness_map = {}
    unique_layer_replicas = {}

    for layer_index in range(len(model_dag)):
        if layer_index < starting_layer:
            continue
        layer_specs = model_dag[layer_index]
        if is_dw_layer(layer_specs) or not is_conv_layer(layer_specs):
            continue
        ifms_shape = get_layer_ifms_shape(layer_specs)
        weightss_shape = get_layer_weights_shape(layer_specs)
        strides = get_strides(layer_specs)
        layer_key = ''
        for dim in ifms_shape:
            layer_key += str(dim)
        for dim in weightss_shape:
            layer_key += str(dim)

        layer_key += str(strides)

        if layer_key not in uniqueness_map:
            uniqueness_map[layer_key] = layer_index
            unique_layers[layer_index] = 0

        unique_layers[uniqueness_map[layer_key]] += 1
        unique_layer_replicas[layer_index] = uniqueness_map[layer_key]

    return unique_layers, unique_layer_replicas


def get_unique_dw_layers_and_frequencies(model_dag, starting_layer=0):
    unique_layers = {}
    uniqueness_map = {}
    unique_layer_replicas = {}

    for layer_index in range(len(model_dag)):
        if layer_index < starting_layer:
            continue
        layer_specs = model_dag[layer_index]
        if not is_dw_layer(layer_specs):
            continue
        ifms_shape = get_layer_ifms_shape(layer_specs)
        weightss_shape = get_layer_weights_shape(layer_specs)
        strides = get_strides(layer_specs)
        layer_key = ''
        for dim in ifms_shape:
            layer_key += str(dim)
        for dim in weightss_shape:
            layer_key += str(dim)

        layer_key += str(strides)

        if layer_key not in uniqueness_map:
            uniqueness_map[layer_key] = layer_index
            unique_layers[layer_index] = 0

        unique_layers[uniqueness_map[layer_key]] += 1
        unique_layer_replicas[layer_index] = uniqueness_map[layer_key]

    return unique_layers, unique_layer_replicas


def get_config_parallelism(mapping_config_parallelism_file):
    layers_parallelism_config = []
    with open(mapping_config_parallelism_file, 'r') as f:
        line_num = 0
        for line in f:
            line = line.replace('(', '').replace(')', '').replace(
                ' ', '').replace('\n', '')
            splits = line.split(',')
            if len(splits) > 1:
                # par_ofms), self.par_ifms, self.par_in_filter, self.par_height, int(self.par_width
                layers_parallelism_config.append({})
                layers_parallelism_config[line_num]['par_ofms'] = int(
                    float(splits[0]))
                layers_parallelism_config[line_num]['par_ifms'] = int(
                    float(splits[1]))
                layers_parallelism_config[line_num]['par_in_filter'] = int(
                    float(splits[2]))
                layers_parallelism_config[line_num]['par_h'] = int(
                    float(splits[3]))
                layers_parallelism_config[line_num]['par_w'] = int(
                    float(splits[4]))
                layers_parallelism_config[line_num]['all'] = 1
                for i in range(len(splits)):
                    layers_parallelism_config[line_num]['all'] *= int(
                        float(splits[i]))

                line_num += 1

    return layers_parallelism_config


def segrr_get_inter_segment_buffer_shape(mapping_config_inter_seg_buffers_file, pipe_len, model_dag):

    inter_segment_buffers_sz = 0
    inter_segment_buffers_shape = [1, 1, 1]
    inter_segment_buffer_locations = []
    with open(mapping_config_inter_seg_buffers_file, 'r') as f:
        for line in f:
            line = clean_line(line)
            if len(line) > 2:
                inter_segment_buffer_locations = read_int_list_from_line(line)

    index = 0
    for segment_buffer_location in inter_segment_buffer_locations:
        if segment_buffer_location == 1:
            first_layer_offset = index * pipe_len
            first_layer_index = get_conv_layer_index_from_offset(
                model_dag, 0, first_layer_offset)
            first_layer_specs = model_dag[first_layer_index]
            if inter_segment_buffers_sz < get_layer_ifms_size(first_layer_specs):
                inter_segment_buffers_sz = get_layer_ifms_size(
                    first_layer_specs)
                inter_segment_buffers_shape = get_layer_ifms_shape(
                    first_layer_specs)

        index += 1

    return inter_segment_buffers_shape


def list_to_file_name_str(a_list):
    ret_str = ''
    for i in range(len(a_list) - 1):
        ret_str += str(a_list[i]) + '_'

    ret_str += a_list[-1]

    return ret_str

def ideal_weight_reuse(model_dag):

    num_ops = sum(get_layers_op_counts(model_dag, 0))
    num_weights = sum(get_weights_sizes(model_dag))

    return int(num_ops / num_weights)

def ideal_fms_reuse(model_dag):

    num_ops = sum(get_layers_op_counts(model_dag, 0))
    num_fms = sum(get_ifms_sizes(model_dag)) + sum(get_ofms_sizes(model_dag))

    return int(num_ops / num_fms)