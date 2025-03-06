import __init__
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from preformance_record import *
import constants as consts
import mapping_utils.custom_mapping_utils as custom_mapping_utils
import optimizers.simulated_annealing as sa
import optimizers.genetic_algorithm as ga
from datetime import datetime
import optimizers.nsga as nsga2


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
    
    energy = mapping.calc_energy()

    record = PerformanceRecord(board_name, model_name, mapping.get_label(),
                               mapping.get_num_engines(),
                               estimated_exec_time,
                               estimated_throughput,
                               on_chip_fms_buffer_sz, on_chip_weights_buffer_sz,
                               on_chip_buffer_sz,
                               off_chip_fms_access, off_chip_weight_access, energy=energy)

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


def run_nsga2_experiments(board_name_list, model_name_list, metric_list,
                          number_of_generations, population_size,
                          min_segments=constants.MIN_SEGMENTS,
                          max_segments=constants.MAX_ENGINES_V2,
                          print_timing=False):

    best_in_metric_board_model = {}
    num_models = len(model_name_list)
    num_boards = len(board_name_list)
    metrics_str = ''
    for metric in metric_list:
        timing_metric = Metrics.THROUGHPUT
        metrics_str += constants.metric_display_names[metric].lower() + '_'
        if metric == Metrics.LATENCY:
            timing_metric = metric

    for board_name in board_name_list:
        best_in_metric_board_model[board_name] = {}
        boards_str = board_name.lower()
        for model_name in model_name_list:
            models_str = model_name.lower()
            model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
            best_in_metric_board_model[board_name][model_name] = {}
            print(board_name, model_name)
            custom_mapping, mapping_desc =\
                custom_mapping_utils.generate_random_mapping(
                    board_name, model_name, model_dag, 2, timing_metric)
            population = nsga2.run_nsga2(
                mapping_desc, metric_list, number_of_generations, population_size, min_segments=min_segments,
                max_segments=max_segments, print_timing=print_timing)

            population_dict_list = []
            for mapping_desc in population:
                population_dict_list.append(custom_mapping_utils.prepare_custom_mapping_desc(
                    mapping_desc.segment_layers_list,
                    mapping_desc.segment_block_list,
                    mapping_desc.block_engines_list))

            best_in_metric_board_model[board_name][model_name] = population_dict_list
            
            mapping_general_utils.save_dict_to_json({model_name: best_in_metric_board_model[board_name][model_name]},
                                                constants.FIGURES_DATA_DIR_P2 + '/nsga2/models/',
                                                ('nsga_{}{}_{}_' +
                                                    'population_{}_generations_{}.json').format(metrics_str, boards_str, models_str,
                                                                                                population_size,
                                                                                                number_of_generations))

        mapping_general_utils.save_dict_to_json({board_name: best_in_metric_board_model[board_name]},
                                                constants.FIGURES_DATA_DIR_P2 + '/nsga2/boards/',
                                                ('nsga_{}{}_' +
                                                    'population_{}_generations_{}.json').format(metrics_str, boards_str,
                                                                                                population_size,
                                                                                                number_of_generations))
    mapping_general_utils.save_dict_to_json(best_in_metric_board_model,
                                            constants.FIGURES_DATA_DIR_P2 + '/nsga2/metrics/',
                                            ('nsga_{}boards_{}_models_{}_' +
                                                'population_{}_generations_{}.json').format(metrics_str,
                                                                                            num_boards, num_models,
                                                                                            population_size,
                                                                                            number_of_generations))

    return best_in_metric_board_model


def run_genetic_algorithm_experiments(board_name_list, model_name_list, metric_list,
                                      number_of_generations, population_size,
                                      min_segments=constants.MIN_SEGMENTS,
                                      max_segments=constants.MAX_ENGINES_V2):

    best_in_metric_board_model = {}
    metrics_str = get_file_prefix_from_metric_board_or_model_list(metric_list)
    boards_str = get_file_prefix_from_metric_board_or_model_list(board_name_list)
    models_str = get_file_prefix_from_metric_board_or_model_list(model_name_list)
    
    metrics_str_list = metrics_str.split('_')
    boards_str_list = boards_str.split('_')
    
    metric_index = -1
    for metric in metric_list:
        metric_index += 1
        timing_metric = Metrics.THROUGHPUT
        if metric == Metrics.LATENCY:
            timing_metric = metric
        metric_label = consts.metric_display_names[metric]
        best_in_metric_board_model[metric_label] = {}
        board_index = -1
        for board_name in board_name_list:
            board_index += 1
            best_in_metric_board_model[metric_label][board_name] = {}
            model_index = -1
            for model_name in model_name_list:
                model_index += 1
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                best_in_metric_board_model[metric_label][board_name][model_name] = {
                }
                best_perf_across_engines = -1
                best_mapping_across_engines = None
                print(board_name, model_name)
                custom_mapping, mapping_desc =\
                    custom_mapping_utils.generate_random_mapping(
                        board_name, model_name, model_dag, 2, timing_metric)
                orig_best, best_mapping_desc = ga.run_genetic_algorithm(
                    mapping_desc, metric, number_of_generations, population_size, min_segments=min_segments,
                    max_segments=max_segments)
                mapping_desc_dict = custom_mapping_utils.prepare_custom_mapping_desc(
                    best_mapping_desc.segment_layers_list,
                    best_mapping_desc.segment_block_list,
                    best_mapping_desc.block_engines_list)
                best_sa_mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                    board_name, model_dag, mapping_desc_dict, timing_metric)
                perf_record = run_mapping(
                    board_name, model_name, best_sa_mapping)
                best = perf_record.get_metric_val(metric)
                # print((metric, orig_best, best, best_perf_across_engines, str(best_mapping_desc)))
                if best_perf_across_engines == -1 or \
                        first_better_than_second(metric, best, best_perf_across_engines):
                    best_perf_across_engines = best
                    best_mapping_across_engines = copy.deepcopy(
                        best_mapping_desc)

                best_mapping_across_engines_dicts = custom_mapping_utils.prepare_custom_mapping_desc(
                    best_mapping_across_engines.segment_layers_list,
                    best_mapping_across_engines.segment_block_list,
                    best_mapping_across_engines.block_engines_list)
                best_in_metric_board_model[metric_label][board_name][model_name] = [
                    best_perf_across_engines, best_mapping_across_engines_dicts]

                mapping_general_utils.save_dict_to_json({model_name: best_in_metric_board_model[metric_label][board_name][model_name]},
                                                        constants.FIGURES_DATA_DIR_P2 + '/genetic/models/',
                                                        ('genetic_algorithm_bests_in_{}_boards_{}_models_{}_' +
                                                        'population_{}_generations_{}.json').format(metrics_str_list[metric_index], 
                                                                                                    boards_str_list[board_index],
                                                                                                    model_name.lower(),
                                                                                                    population_size,
                                                                                                    number_of_generations))

            mapping_general_utils.save_dict_to_json({metric_label: best_in_metric_board_model[metric_label]},
                                                    constants.FIGURES_DATA_DIR_P2 + '/genetic/boards/',
                                                    ('genetic_algorithm_bests_in_{}_boards_{}_models_{}_' +
                                                        'population_{}_generations_{}.json').format(metrics_str_list[metric_index],
                                                                                                    boards_str_list[board_index],
                                                                                                    models_str,
                                                                                                    population_size,
                                                                                                    number_of_generations))
            
        mapping_general_utils.save_dict_to_json({metric_label: best_in_metric_board_model[metric_label]},
                                                    constants.FIGURES_DATA_DIR_P2 + '/genetic/boards/',
                                                    ('genetic_algorithm_bests_in_{}_boards_{}_models_{}_' +
                                                        'population_{}_generations_{}.json').format(metrics_str_list[metric_index],
                                                                                                    boards_str, 
                                                                                                    models_str,
                                                                                                    population_size,
                                                                                                    number_of_generations))

    return best_in_metric_board_model


def run_simulated_annealing_experiments(board_name_list, model_name_list, metric_list, num_iterations,
                                        min_segments=constants.MIN_SEGMENTS,
                                        max_segments=constants.MAX_ENGINES_V2):
    best_in_metric_board_model = {}
    metrics_str = get_file_prefix_from_metric_board_or_model_list(metric_list)
    boards_str = get_file_prefix_from_metric_board_or_model_list(board_name_list)
    models_str = get_file_prefix_from_metric_board_or_model_list(model_name_list)
    metrics_str_list = metrics_str.split('_')
    boards_str_list = boards_str.split('_')
    metric_index = -1

    for metric in metric_list:
        metric_index += 1
        timing_metric = Metrics.THROUGHPUT
        if metric == Metrics.LATENCY:
            timing_metric = metric
        metric_label = consts.metric_display_names[metric]
        best_in_metric_board_model[metric_label] = {}
        board_index = -1
        for board_name in board_name_list:
            board_index += 1
            best_in_metric_board_model[metric_label][board_name] = {}
            model_index = -1
            for model_name in model_name_list:
                model_index += 1
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                best_in_metric_board_model[metric_label][board_name][model_name] = {
                }
                best_perf_across_engines = -1
                best_mapping_across_engines = None
                #for num_segments in range(min_segments, max_segments + 1):
                #print(board_name, model_name, num_segments)
                custom_mapping, mapping_desc =\
                    custom_mapping_utils.generate_random_mapping(
                        board_name, model_name, model_dag, num_segments= -1, timing_metric= timing_metric)
                _, best_mapping_desc = sa.run_simulated_annealing(
                    custom_mapping, mapping_desc, metric, num_iterations)
                mapping_desc_dict = custom_mapping_utils.prepare_custom_mapping_desc(
                    best_mapping_desc.segment_layers_list,
                    best_mapping_desc.segment_block_list,
                    best_mapping_desc.block_engines_list)
                best_sa_mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                   board_name, model_dag, mapping_desc_dict, timing_metric)
                best_mapping_across_engines = copy.deepcopy(best_mapping_desc)
                perf_record = run_mapping(
                    board_name, model_name, best_sa_mapping)
                best_perf_across_engines = perf_record.get_metric_val(metric)
                # best = perf_record.get_metric_val(metric)
                # if best_perf_across_engines == -1 or \
                #         first_better_than_second(metric, best, best_perf_across_engines):
                #     best_perf_across_engines = best
                #     best_mapping_across_engines = copy.deepcopy(
                #         best_mapping_desc)

                best_mapping_across_engines_dicts = custom_mapping_utils.prepare_custom_mapping_desc(
                    best_mapping_across_engines.segment_layers_list,
                    best_mapping_across_engines.segment_block_list,
                    best_mapping_across_engines.block_engines_list)
                best_in_metric_board_model[metric_label][board_name][model_name] = [
                    best_perf_across_engines, best_mapping_across_engines_dicts]

                mapping_general_utils.save_dict_to_json({model_name: best_in_metric_board_model[metric_label][board_name][model_name]},
                                                        constants.FIGURES_DATA_DIR_P2 + '/genetic/models/',
                                                        ('genetic_algorithm_bests_in_{}_boards_{}_models_{}_' +
                                                        'population_{}_generations_{}.json').format(metrics_str_list[metric_index], 
                                                                                                    boards_str_list[board_index],
                                                                                                    model_name.lower(),
                                                                                                    num_iterations))

            mapping_general_utils.save_dict_to_json({metric_label: best_in_metric_board_model[metric_label]},
                                                    constants.FIGURES_DATA_DIR_P2 + '/genetic/boards/',
                                                    ('genetic_algorithm_bests_in_{}_boards_{}_models_{}_' +
                                                        'population_{}_generations_{}.json').format(metrics_str_list[metric_index],
                                                                                                    boards_str_list[board_index], 
                                                                                                    models_str,
                                                                                                    num_iterations))
            
        mapping_general_utils.save_dict_to_json({metric_label: best_in_metric_board_model[metric_label]},
                                                    constants.FIGURES_DATA_DIR_P2 + '/genetic/boards/',
                                                    ('genetic_algorithm_bests_in_{}_boards_{}_models_{}_' +
                                                        'population_{}_generations_{}.json').format(metrics_str_list[metric_index],
                                                                                                    boards_str, 
                                                                                                    models_str,
                                                                                                    num_iterations))

    return best_in_metric_board_model

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