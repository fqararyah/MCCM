import __init__
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from preformance_record import *
import constants as consts
import mapping_utils.custom_mapping_utils as custom_mapping_utils
from datetime import datetime


def get_file_prefix_from_metric_board_or_model_list(a_list):
    if set(a_list).issubset(set(constants.model_names)):
        if len(a_list) == len(constants.model_names):
            ret_str = 'all_models'
        else:
            model_name_list = [model_name.lower()
                               for model_name in constants.model_name]
            ret_str = utils.list_to_file_name_str(model_name_list)
    elif set(a_list).issubset(set(constants.metric_display_names)) or \
            set(a_list).issubset(set(constants.metric_list)) or \
            set(a_list).issubset(set(constants.metric_list)):
        if len(a_list) == len(constants.metric_list):
            ret_str = 'all_metrics'
        else:
            metric_file_names = [constants.metric_file_names[metric].lower() for metric in a_list]
            ret_str = utils.list_to_file_name_str(metric_file_names)
    else:
        board_name_list = [board_name.lower() for board_name in a_list]
        ret_str = utils.list_to_file_name_str(board_name_list)

    return ret_str


def initialized_base_mapping(board_name, model_name, mapping_label, num_engines):
    hw_cfg = HWConfig(board_name)
    model_dag = utils.read_model_dag_v2(
        constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
    layers = utils.get_conv_layer_indices_in_range(
        model_dag, 0, len(model_dag))

    if HybridMapping.MAPPING_LABEL == mapping_label:
        mapping = HybridMapping(
            hw_cfg, model_dag, layers, num_engines)
    if SegmentMapping.MAPPING_LABEL == mapping_label:
        mapping = SegmentMapping(
            hw_cfg, model_dag, layers, num_engines)
    if SegmentMappingRR.MAPPING_LABEL == mapping_label:
        mapping = SegmentMappingRR(
            hw_cfg, model_dag, layers, num_engines)

    return mapping


def run_mapping(board_name, model_name, mapping):
    estimated_exec_time = round(1000 * mapping.calc_exec_time(), 2)
    estimated_throughput = mapping.calc_throughput()
    on_chip_fms_buffer_sz = round(
        mapping.calc_fms_buffer_sz() / constants.MiB, 2)
    on_chip_weights_buffer_sz = round(
        mapping.calc_weights_buffer_sz() / constants.MiB, 2)
    on_chip_buffer_sz = round(
        mapping.calc_on_chip_buffer_sz() / constants.MiB, 2)

    off_chip_weight_access = round(
        mapping.calc_off_chip_weights_access() / constants.MiB, 2)
    off_chip_fms_access = round(
        mapping.calc_off_chip_fms_access() / constants.MiB, 2)
    
    record = PerformanceRecord(board_name, model_name, mapping.get_label(),
                               mapping.get_num_engines(),
                               estimated_exec_time,
                               estimated_throughput,
                               on_chip_fms_buffer_sz, on_chip_weights_buffer_sz,
                               on_chip_buffer_sz,
                               off_chip_fms_access, off_chip_weight_access)

    return record


def run_base_mapping(board_name, model_name, mapping_label, num_engines):
    mapping = initialized_base_mapping(
        board_name, model_name, mapping_label, num_engines)
    performance_record = run_mapping(board_name, model_name, mapping)

    return performance_record, mapping


def run_a_base_mapping_engine_range(board_name, model_name, mapping_label,
                                    min_engines=constants.MIN_ENGINES, max_engines=constants.MAX_ENGINES,
                                    serialize_records = False):
    perf_records = []
    for num_engines in range(min_engines, max_engines + 1):
        mapping = initialized_base_mapping(
            board_name, model_name, mapping_label, num_engines)
        if serialize_records:
            perf_records.append(run_mapping(board_name, model_name, mapping).__dict__)
        else:
            perf_records.append(run_mapping(board_name, model_name, mapping))

    return perf_records


def run_base_mappings_engine_range(board_name, model_name,
                                   max_engines_bounded_by_layers = False,
                                   min_engines=constants.MIN_ENGINES, max_engines=constants.MAX_ENGINES,
                                   serialize_records = False):
    perf_records = {}
    mapping_labels = [HybridMapping.MAPPING_LABEL,
                      SegmentMapping.MAPPING_LABEL, SegmentMappingRR.MAPPING_LABEL]
    
    if max_engines_bounded_by_layers:
        model_dag = utils.read_model_dag_v2(
                        constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
        num_conv_layers = utils.get_num_conv_layer_count_in_range(
            model_dag, 0, len(model_dag))
        max_engines = num_conv_layers
    for mapping_label in mapping_labels:
        if mapping_label == HybridMapping.MAPPING_LABEL:
            max_engines -= 1
        perf_records[mapping_label] = run_a_base_mapping_engine_range(board_name, model_name, mapping_label,
                                                                      min_engines, max_engines, serialize_records)

    return perf_records


def get_best_of_a_mapping(board_name, model_name, mapping_label, min_engines, max_engines, metric):

    best_val = -1
    best_mapping = None
    for num_engines in range(min_engines, max_engines + 1):
        perf_record, mapping = run_base_mapping(
            board_name, model_name, mapping_label, num_engines)
        if best_val == -1 or not perf_record.is_better(metric, best_val):
            best_val = perf_record.get_metric_val(metric)
            best_mapping = mapping

    return best_val, best_mapping


def get_bests_of_mappings(board_name_list, model_name_list, mapping_label_list, min_engines, max_engines, metric_list):

    best_in_metric_board_model_mapping_dict = {}
    for metric in metric_list:
        metric_label = consts.metric_display_names[metric]
        best_in_metric_board_model_mapping_dict[metric_label] = {}
        for board_name in board_name_list:
            best_in_metric_board_model_mapping_dict[metric_label][board_name] = {
            }
            for model_name in model_name_list:
                print(board_name, model_name)
                best_in_metric_board_model_mapping_dict[metric_label][board_name][model_name] = {
                }
                for mapping_label in mapping_label_list:
                    best_val, best_mapping = get_best_of_a_mapping(
                        board_name, model_name, mapping_label, min_engines, max_engines, metric)
                    best_mapping_desc_dict = best_mapping.get_dict_representation()
                    best_in_metric_board_model_mapping_dict[metric_label][board_name][model_name][mapping_label] = \
                        [best_val, best_mapping_desc_dict]

    return best_in_metric_board_model_mapping_dict

def validate_optimized_mapping_dicts(performance_dict):
    norm_dict = {}
    for metric_label, metric_dict in performance_dict.items():
        metric = consts.display_names_metrics_dict[metric_label]
        norm_dict[metric_label] = {}
        for board_name, board_name_dict in metric_dict.items():
            norm_dict[metric_label][board_name] = {}
            for model_name, model_name_dict in board_name_dict.items():
                mapping_dict = performance_dict[metric_label][board_name][model_name][1]
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                if not mapping_general_utils.validate_mapping_dict(mapping_dict, model_dag):
                    print('INVALID MAPPING:', mapping_dict)