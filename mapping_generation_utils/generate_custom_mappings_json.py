import __init__
import json
import mapping_utils.mapping_general_utils as mapping_general_utils
import constants
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from mapping_types.hybrid_mapping import *
from mapping_types.custom_mapping import *
import os
from code_gen import code_generation_constants as cgc
import random

random.seed(5)


def generate_hybrid(board_name, model_name, min_engines, max_engines):
    mappings_list = []

    for num_engines in range(min_engines, max_engines + 1):
        mapping_dict = {}
        mapping_dict["board_name"] = board_name
        mapping_dict["model_name"] = model_name
        mapping_dict["layer_engine_mapping"] = {"{}-{}".format(0, num_engines): "{}-{}".format(0, num_engines),
                                                "{}-last".format(num_engines): str(num_engines)}
        mappings_list.append(mapping_dict)

    json_obj = json.dumps(mappings_list)
    out_dir = constants.CUSTOM_MAPPINGS_JSON_DIR + '/' + HybridMapping.MAPPING_LABEL
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + "/{}_{}.json".format(board_name, model_name), "w") as f:
        f.write(json_obj)


def generate_segrr(board_name, model_name, min_engines, max_engines):
    mappings_list = []

    for num_engines in range(min_engines, max_engines + 1):
        mapping_dict = {}
        mapping_dict["board_name"] = board_name
        mapping_dict["model_name"] = model_name
        mapping_dict["layer_engine_mapping"] = {
            "{}-last".format(0, num_engines): "{}-{}".format(0, num_engines)}
        mappings_list.append(mapping_dict)

    json_obj = json.dumps(mappings_list)
    out_dir = constants.CUSTOM_MAPPINGS_JSON_DIR + '/' + SegmentMappingRR.MAPPING_LABEL
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + "/{}_{}.json".format(board_name, model_name), "w") as f:
        f.write(json_obj)


def generate_seg(board_name, model_name, min_engines, max_engines):
    mappings_list = []

    model_dag = utils.read_model_dag_v2(
        cgc.MODEL_ARCH_DIR + model_name + '/model_dag.json')

    for num_engines in range(min_engines, max_engines + 1):
        layer_cluster_map = mapping_general_utils.cluster_layers_using_kmeans(
            model_dag, num_engines)
        print(board_name, model_name, num_engines)
        num_layers_so_far = 0
        mapping_config_dict = {}
        for i in range(num_engines):
            current_engine_num_layers = layer_cluster_map[i]
            last_layer_offset = num_layers_so_far + current_engine_num_layers
            mapping_config_dict["{}-{}".format(num_layers_so_far,
                                               last_layer_offset)] = "{}".format(i)

            num_layers_so_far += current_engine_num_layers

        mapping_dict = {}
        mapping_dict["board_name"] = board_name
        mapping_dict["model_name"] = model_name
        mapping_dict["layer_engine_mapping"] = mapping_config_dict
        mappings_list.append(mapping_dict)

    json_obj = json.dumps(mappings_list)
    out_dir = constants.CUSTOM_MAPPINGS_JSON_DIR + '/' + SegmentMapping.MAPPING_LABEL
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + "/{}_{}.json".format(board_name, model_name), "w") as f:
        f.write(json_obj)


def generate_hetero_sesl_seg(board_name, model_name, min_engines, max_engines, num_instances):
    mappings_list = []
    model_dag = utils.read_model_dag_v2(
        constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
    num_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))
    possibilities_count = 0
    out_dir = constants.CUSTOM_MAPPINGS_JSON_DIR + '/' + \
        'hetero_sesl_seg_{}'.format(num_instances)
    out_file = out_dir + "/{}_{}.json".format(board_name, model_name)

    if os.path.exists(out_file):
        print('ALREADY GENERATED')
    else:
        for first_part_num_engines in range(max_engines[0], min_engines[0] - 1, -1):
            first_layer_in_second_part = first_part_num_engines
            for second_part_num_engines in range(min_engines[1], max_engines[1] + 1):
                if first_part_num_engines + second_part_num_engines >= constants.MAX_ENGINES or \
                        len(mappings_list) >= num_instances:
                    break
                second_part_num_layers = num_layers - first_part_num_engines

                second_part_possibilities = mapping_general_utils.generate_distributions(
                    second_part_num_layers, second_part_num_engines)
                for possibility in second_part_possibilities:
                    if len(mappings_list) >= num_instances:
                        break
                    mapping_dict = {}
                    mapping_dict["board_name"] = board_name
                    mapping_dict["model_name"] = model_name
                    current_layer = first_layer_in_second_part
                    current_engine = first_part_num_engines
                    possibilities_count += 1
                    mapping_dict["layer_engine_mapping"] = {"{}-{}".format(
                        0, first_part_num_engines): "{}-{}".format(0, first_part_num_engines)}
                    for engine_layer_count in possibility:
                        mapping_dict["layer_engine_mapping"]["{}-{}".format(str(current_layer),
                                                                            str(current_layer + engine_layer_count))] = str(current_engine)
                        current_engine += 1
                        current_layer += engine_layer_count

                    mappings_list.append(mapping_dict)

        json_obj = json.dumps(mappings_list)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(out_file, "w") as f:
            f.write(json_obj)

        print(possibilities_count, 'possibilities!')

board_name_list = mapping_general_utils.read_board_names(file_name=constants.HW_CONFIGS_FILE_v1)
model_name_list = constants.model_names

do_generate_hybrid = False
do_generate_seg = False
do_generate_segrr = False
do_generate_hetero = False

for board_name in board_name_list:
    for model_name in model_name_list:
        if do_generate_hybrid:
            generate_hybrid(board_name, model_name,
                            constants.MIN_ENGINES, constants.MAX_ENGINES)
        if do_generate_segrr:
            generate_segrr(board_name, model_name,
                           constants.MIN_ENGINES, constants.MAX_ENGINES)
        if do_generate_seg:
            generate_seg(board_name, model_name,
                         constants.MIN_ENGINES, constants.MAX_ENGINES)

num_instances = 100000
if do_generate_hetero:
    generate_hetero_sesl_seg('vcu110', 'xce_r', [constants.MIN_ENGINES, constants.MIN_ENGINES],
                             [constants.MAX_ENGINES - constants.MIN_ENGINES,
                                 constants.MAX_ENGINES - constants.MIN_ENGINES],
                             num_instances)
