
import __init__
import utils
from engines.engine import *
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.sesl_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from hw_config import *
import random
from mapping_types.custom_mapping import *
from preformance_record import *


def prepare_custom_mapping_desc(segment_layers_list, segment_blocks_list, block_engines_list):
    mapping_desc = {}
    for segment_index in range(len(segment_layers_list)):
        segment_layers = segment_layers_list[segment_index]
        segment_engines = block_engines_list[segment_blocks_list[segment_index]]

        layers_str = str(segment_layers[0])
        if len(segment_layers) == 2:
            layers_str += '-' + str(segment_layers[1])
        engines_str = str(segment_engines[0])
        if len(segment_engines) == 2:
            engines_str += '-' + str(segment_engines[1])

        mapping_desc[layers_str] = engines_str

    return mapping_desc


def custom_mapping_from_desc_dict(board_name, model_dag, mapping_desc_dict):
    layers = utils.get_conv_layer_indices_in_range(
        model_dag, 0, len(model_dag))

    mappings_segments_config_list = infer_mapping_types(
        mapping_desc_dict, model_dag)
    hw_config = HWConfig(board_name)

    return CustomMapping(hw_config, model_dag, layers, mappings_segments_config_list)


def infer_mapping_types(mapping_details, model_dag):
    mappings_segments_list = []
    for key, val in mapping_details.items():
        engine_list = mapping_general_utils.clean_engine_range(val)
        layer_lits = mapping_general_utils.clean_layer_range(key, model_dag)
        if len(engine_list) > 1:
            if len(layer_lits) == len(engine_list):
                mapping_type = SESLMapping.MAPPING_LABEL
            else:
                mapping_type = SegmentMappingRR.MAPPING_LABEL
        else:
            mapping_type = SEMLMapping_LBL.MAPPING_LABEL

        mappings_segments_list.append(
            {'mapping': mapping_type, 'engine_list': engine_list, 'layer_list': layer_lits})

    return mappings_segments_list


def copy_segment_layers_or_engines(segment_src_list, segment_dst_list,
                                   starting_segment_in_src, starting_segment_in_dst,
                                   end_segment, add_offset=0):

    index_in_dst = starting_segment_in_dst
    assert index_in_dst <= len(segment_dst_list)
    # print(starting_segment_in_src, end_segment, len(segment_src_list))
    for i in range(starting_segment_in_src, end_segment):
        if index_in_dst < len(segment_dst_list):
            segment_dst_list[index_in_dst] = []
        else:
            segment_dst_list.append([])
        for layer in segment_src_list[i]:
            segment_dst_list[index_in_dst].append(layer + add_offset)

        index_in_dst += 1

def proportional_allocation(num_pes, op_counts_list):

    engines_pes = []
    alpha = 0
    sum_roots = 0
    for i in range(len(op_counts_list)):
        sum_roots += math.sqrt(op_counts_list[i])

    alpha = (sum_roots / num_pes) ** 2

    for i in range(len(op_counts_list)):
        engines_pes.append(max(1, int(math.sqrt(op_counts_list[i] / alpha))))

    return engines_pes
