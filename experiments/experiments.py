
from mapping_types.generic_mapping import *
from mapping_types.sesl_mapping import *
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from mapping_types.hybrid_mapping import *
from mapping_types.hybrid_rr_mapping import *
from mapping_types.custom_mapping import *
from mapping_utils.custom_mapping_utils import *
import __init__
import utils
import constants
from plot import plot_results
import mapping_utils.mapping_general_utils as mapping_general_utils
from hw_config import *
from preformance_record import *
import os


def plot_engines_and_metrics(engine_list, estimated_exec_times, exec_time_unit,
                             on_chip_fms_buffer_szs, on_chip_weights_buffer_szs, on_chip_buffer_szs,
                             buffer_sz_unit,
                             off_chip_fms_access_list, off_chip_weight_access_list, off_chip_access_list,
                             access_unit,
                             mappings_labels, board_name, model_name,
                             plot_mode='sep', normalized=True, normalize=True):

    x_axis_label = 'No of engines'

    plot_results.bar_mapping_groups(engine_list, estimated_exec_times, 'latency', mappings_labels,
                                    [x_axis_label,
                                     'latency'], 'engines_latency', board_name, model_name, exec_time_unit,
                                    plot_mode, normalized, normalize)

    plot_results.bar_mapping_groups(engine_list, off_chip_fms_access_list, 'access', mappings_labels,
                                    [x_axis_label, 'off-chip access FMs'], 'engines_access_fms', board_name, model_name, access_unit,
                                    plot_mode, normalized, normalize)
    plot_results.bar_mapping_groups(engine_list, off_chip_weight_access_list, 'access', mappings_labels,
                                    [x_axis_label, 'off-chip access weights'], 'engines_access_weights', board_name, model_name, access_unit,
                                    plot_mode, normalized, normalize)
    plot_results.bar_mapping_groups(engine_list, off_chip_access_list, 'access', mappings_labels,
                                    [x_axis_label, 'off-chip access'], 'engines_access', board_name, model_name, access_unit,
                                    plot_mode, normalized, normalize)

    plot_results.bar_mapping_groups(engine_list, on_chip_fms_buffer_szs, 'buffers', mappings_labels,
                                    [x_axis_label, 'on-chip FMs buffer'], 'engines_on_chip_fms', board_name, model_name, buffer_sz_unit,
                                    plot_mode, normalized, normalize)
    plot_results.bar_mapping_groups(engine_list, on_chip_weights_buffer_szs, 'buffers', mappings_labels,
                                    [x_axis_label, 'on-chip weights buffer'], 'engines_on_chip_weights', board_name, model_name, buffer_sz_unit,
                                    plot_mode, normalized, normalize)
    plot_results.bar_mapping_groups(engine_list, on_chip_buffer_szs, 'buffers', mappings_labels,
                                    [x_axis_label, 'on-chip buffer'], 'engines_on_chip', board_name, model_name, buffer_sz_unit,
                                    plot_mode, normalized, normalize)


def plot_time_on_chip_buffers(estimated_exec_times,
                              on_chip_fms_buffer_szs, on_chip_weights_buffer_szs, on_chip_buffer_szs,
                              buffer_sz_unit,
                              mappings_labels, board_name, model_name, latency=True,
                              custom_engine_counts=None,
                              plot_line=None):
    latency_throughput = 'Latency'
    latency_throughput_label = latency_throughput
    if not latency:
        latency_throughput = 'Throughput'
        latency_throughput_label = latency_throughput + ' (FPS)'
    plot_results.scatter(estimated_exec_times, on_chip_fms_buffer_szs, 'buffers', mappings_labels,
                         [latency_throughput_label,
                             'on-chip weights buffer ({})'.format(buffer_sz_unit)], latency_throughput + '_on_chip_fms',
                         board_name, model_name, custom_engine_counts=custom_engine_counts, plot_line=plot_line,
                         annotate_max_min=True)
    plot_results.scatter(estimated_exec_times, on_chip_weights_buffer_szs, 'buffers', mappings_labels,
                         [latency_throughput_label,
                             'on-chip FMs buffer ({})'.format(buffer_sz_unit)], latency_throughput + '_on_chip_weights',
                         board_name, model_name, custom_engine_counts=custom_engine_counts, plot_line=plot_line,
                         annotate_max_min=True)
    plot_results.scatter(estimated_exec_times, on_chip_buffer_szs, 'buffers', mappings_labels,
                         [latency_throughput_label, 'Buffer ({})'.format(
                             buffer_sz_unit)],
                         latency_throughput + '_on_chip', board_name, model_name, custom_engine_counts=custom_engine_counts, 
                         plot_line=plot_line, annotate_max_min=True)


def plot_time_off_chip_access(estimated_exec_times,
                              off_chip_fms_access_list, off_chip_weight_access_list, off_chip_access_list,
                              access_unit,
                              mappings_labels, board_name, model_name, latency=True,
                              custom_engine_counts=None,
                              plot_line=None):
    latency_throughput = 'Latency'
    latency_throughput_label = latency_throughput
    if not latency:
        latency_throughput = 'Throughput'
        latency_throughput_label = latency_throughput + ' (FPS)'
    plot_results.scatter(estimated_exec_times, off_chip_fms_access_list, 'access', mappings_labels,
                         [latency_throughput_label,
                             'off-chip access FMs (MiB)'], latency_throughput + '_access_fms',
                         board_name, model_name, custom_engine_counts=custom_engine_counts, plot_line=plot_line,
                         annotate_max_min=True)
    plot_results.scatter(estimated_exec_times, off_chip_weight_access_list, 'access', mappings_labels,
                         [latency_throughput_label,
                             'off-chip access weights (MiB)'], latency_throughput + '_access_weights',
                         board_name, model_name, custom_engine_counts=custom_engine_counts, plot_line=plot_line,
                         annotate_max_min=True)
    plot_results.scatter(estimated_exec_times, off_chip_access_list, 'access', mappings_labels,
                         [latency_throughput_label, 'Access (MiB)'], latency_throughput +
                         '_access', board_name, model_name,
                         custom_engine_counts=custom_engine_counts, plot_line=plot_line, annotate_max_min=True)


def best_access_and_board(board_names, model_names,
                          min_engines, max_engines, from_file=False):
    normalized_bests_lat = {}
    normalized_bests_thr = {}
    save_path = mapping_general_utils.prepare_save_path(
        constants.FIGURES_DATA_DIR, 'bests')

    if from_file:
        with open(save_path + 'best_access_normalized_to_first.json', 'r') as f:
            content = json.load(f)
        normalized_bests_lat = content[0]
        normalized_bests_thr = content[1]
    else:
        _, _, bests = engines_and_metrics(board_names, model_names,
                                       min_engines, max_engines, plot=False)
        for board_name in board_names:
            for model_name in model_names:
                for _, val in bests[board_name][model_name].items():
                    first_latency = val['latency']
                    first_throughput = val['throughput']
                    break
                    # if val['latency'] > max_latency:
                    #     max_latency = val['latency']
                for key, val in bests[board_name][model_name].items():
                    if key not in normalized_bests_lat:
                        normalized_bests_lat[key] = []
                        normalized_bests_thr[key] = []

                    normalized_bests_lat[key].append(
                        val['latency'] / first_latency)  # max_latency)
                    normalized_bests_thr[key].append(
                        val['throughput'] / first_throughput)  # min_throughput)

        json_obj = json.dumps([normalized_bests_lat, normalized_bests_thr])
        with open(save_path + 'best_access_normalized_to_first.json', 'w') as f:
            f.write(json_obj)

    x_ticks_dict = {len(board_names): mapping_general_utils.get_model_display_name_list(model_names),
                    len(model_names): mapping_general_utils.get_board_display_name_list(board_names)}
    plot_results.plot_bar_groups(normalized_bests_lat, 'best_latancies_normalized',
                                 x_title=None, y_title=None, x_ticks_dict=x_ticks_dict, relative_save_path='bests')
    plot_results.plot_bar_groups(normalized_bests_thr, 'best_throughputs_normalized',
                                 x_title=None, y_title=None, x_ticks_dict=x_ticks_dict, relative_save_path='bests')


def get_best_mappings(board_names, model_names, min_engines, max_engines, metrics, exec_v2 = True):
    _, bests_engines, bests = engines_and_metrics(board_names, model_names,
                                       min_engines, max_engines, plot=False, exec_v2 = exec_v2)
    best_mappings = {}
    for board_name in board_names:
        best_mappings[board_name] = {}
        for model_name in model_names:
            best_mappings[board_name][model_name] = {}
            for mapping_key, val in bests[board_name][model_name].items():
                for metric in metrics:
                    perf_value = val[metric]
                    num_engines = bests_engines[board_name][model_name][mapping_key][metric]
                    if metric not in best_mappings[board_name][model_name]:
                        best_mappings[board_name][model_name][metric] = [perf_value, num_engines, mapping_key]
                    else:
                        if 'thr' in metric.lower():
                            best_val = best_mappings[board_name][model_name][metric][0]
                            if perf_value >= 1.1 * best_val:
                                best_mappings[board_name][model_name][metric] = [perf_value, num_engines, mapping_key]
                            elif perf_value > 0.9 * best_val and \
                                perf_value < 1.1 * best_val:
                                if perf_value > best_val:
                                    best_mappings[board_name][model_name][metric][0] = perf_value
                                best_mappings[board_name][model_name][metric].append(num_engines)
                                best_mappings[board_name][model_name][metric].append(mapping_key)
                        else:
                            best_val = best_mappings[board_name][model_name][metric][0]
                            if perf_value <= 0.9 * best_val:
                                best_mappings[board_name][model_name][metric] = [perf_value, num_engines, mapping_key]
                            elif perf_value > 0.9 * best_val and \
                                perf_value < 1.1 * best_val:
                                if perf_value < best_val:
                                    best_mappings[board_name][model_name][metric][0] = perf_value
                                best_mappings[board_name][model_name][metric].append(num_engines)
                                best_mappings[board_name][model_name][metric].append(mapping_key)
                                
    return best_mappings


def best_perf_across_model_and_board(board_names, model_names,
                                     min_engines, max_engines, metrics, plot_the_normalization_base_as_bars=True,
                                     from_file=False):

    normalized_bests = {}
    save_path = mapping_general_utils.prepare_save_path(
        constants.FIGURES_DATA_DIR, 'bests')
    file_name_template = 'best_{}_norm_to_first{}.json'
    figure_name_template = 'best_{}_normalized'
    first_mapping_keys = []
    for metric in metrics:
        normalized_bests[metric] = {}

    boards_models_str = ''
    for board_name in board_names:
        boards_models_str += '_' + board_name
    for model_name in model_names:
        boards_models_str += '_' + model_name

    if from_file:
        for metric in metrics:
            file_name = file_name_template.format(
                metric.lower(), boards_models_str)
            with open(save_path + file_name, 'r') as f:
                content = json.load(f)
                normalized_bests[metric] = content
    else:
        _, _, bests = engines_and_metrics(board_names, model_names,
                                       min_engines, max_engines, plot=False)
        for board_name in board_names:
            for model_name in model_names:
                firsts = {}
                first_mapping_key = ''
                for mapping_key, val in bests[board_name][model_name].items():
                    first_mapping_key = mapping_key
                    first_mapping_keys.append(first_mapping_key)
                    for metric in metrics:
                        firsts[metric] = val[metric]
                    break
                for mapping_key, val in bests[board_name][model_name].items():
                    for metric in metrics:
                        if plot_the_normalization_base_as_bars or mapping_key != first_mapping_key:
                            if mapping_key not in normalized_bests[metric]:
                                normalized_bests[metric][mapping_key] = []
                            normalized_bests[metric][mapping_key].append(
                                val[metric] / firsts[metric])  # max_latency)

        for metric in metrics:
            json_obj = json.dumps(normalized_bests[metric])
            file_name = file_name_template.format(
                metric.lower(), boards_models_str)
            with open(save_path + file_name, 'w') as f:
                f.write(json_obj)

    x_ticks_dict = {len(board_names): mapping_general_utils.get_model_display_name_list(model_names),
                    len(model_names): mapping_general_utils.get_board_display_name_list(board_names)}

    assert(first_mapping_keys[-1] == first_mapping_keys[0])
    seperate_baseline_label= None
    if not plot_the_normalization_base_as_bars:
        seperate_baseline_label= first_mapping_keys[0]
    for metric in metrics:
        plot_results.plot_bar_groups(normalized_bests[metric], figure_name_template.format(metric.lower()),
                                     x_title=None, y_title=metric + '\n(normalized)',
                                     x_ticks_dict=x_ticks_dict, relative_save_path='bests',
                                     seperate_baseline_label = seperate_baseline_label)


def best_access_breakdown_across_model_and_board(board_names, model_names,
                                                 min_engines, max_engines, metrics):

    normalized_bests = {}
    figure_name_template = 'best_{}_normalized'
    mapping_keys = []
    metrics_display_labels = {}
    for metric in metrics:
        metrics_display_labels[metric] = metric.split(' ')[0]
        normalized_bests[metrics_display_labels[metric]] = []

    _, _, bests = engines_and_metrics(board_names, model_names,
                                   min_engines, max_engines, plot=False)
    for board_name in board_names:
        for model_name in model_names:
            first = 0
            for _, val in bests[board_name][model_name].items():
                for metric in metrics:
                    first += val[metric]
                break
            for mapping_key, val in bests[board_name][model_name].items():
                for metric in metrics:
                    if mapping_key not in mapping_keys:
                        mapping_keys.append(mapping_key)
                    normalized_bests[metrics_display_labels[metric]].append(
                        val[metric] / first)  # max_latency)

    x_ticks_dict = {len(model_names): mapping_keys,
                    len(mapping_key): mapping_general_utils.get_model_display_name_list(model_names)}

    x_title = ''
    if len(metrics[0]) > 1:
        x_title = metrics[0].split(' ')[1].capitalize()
    plot_results.plot_bar_groups(normalized_bests, figure_name_template.format(metric.lower().replace(' ', '_')),
                                 x_title=x_title + ' (normalized)', y_title=None, x_ticks_dict=x_ticks_dict,
                                 relative_save_path='bests', breakdown=True,
                                 horizontal_bars=True)

def eval_custom_mappings(board_name, model_name, custom_mapping_configs_list):

    custom_engine_count_configs = []
    performance_records = []

    valid_mappings = 0
    invalid_mappings = 0

    model_dag = utils.read_model_dag_v2(
        constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')

    layers = utils.get_conv_layer_indices_in_range(
        model_dag, 0, len(model_dag))

    save_path = mapping_general_utils.prepare_save_path(constants.FIGURES_DATA_DIR,
                                                'custom_mappings')
    num_mappings = len(custom_mapping_configs_list)
    out_perf_file = save_path + \
        'performance_{}_{}_{}.json'.format(
            board_name, model_name, num_mappings)
    if os.path.exists(out_perf_file):
        print('eval_custom_mappings: ALREADY EXISTS!')
    else:
        for current_mapping_configs in custom_mapping_configs_list:
            mapping_details = current_mapping_configs['layer_engine_mapping']
            custom_model_name = current_mapping_configs['model_name']
            custom_board_name = current_mapping_configs['board_name']
            if custom_board_name != board_name or custom_model_name != model_name:
                continue
            hw_config = HWConfig(board_name)

            mappings_segments_config_list = infer_mapping_types(
                mapping_details, model_dag)
            try:
                mapping = CustomMapping(hw_config, model_dag, layers, mappings_segments_config_list,
                                        first_layer_ifms_are_on_chip=False,
                                        last_layer_ofms_are_on_chip=False,
                                        apply_fusion=False)
                valid_mappings += 1
            except:
                invalid_mappings += 1
                continue
            if max(valid_mappings, invalid_mappings) % 1000 == 0:
                print(valid_mappings, invalid_mappings)
            custom_engine_count_configs.append(mapping_details)

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

            performance_records.append(PerformanceRecord(board_name, model_name, mapping.get_label(),
                                                         mapping.get_num_engines(),
                                                         estimated_exec_time,
                                                         estimated_throughput,
                                                         on_chip_fms_buffer_sz, on_chip_weights_buffer_sz,
                                                         on_chip_buffer_sz,
                                                         off_chip_fms_access, off_chip_weight_access).__dict__)

        json_obj = json.dumps(performance_records)
        with open(out_perf_file, 'w') as f:
            f.write(json_obj)
        json_obj = json.dumps(custom_engine_count_configs)
        with open(save_path + 'mapping_configs_{}_{}_{}.json'.format(board_name, model_name, num_mappings), 'w') as f:
            f.write(json_obj)


def metric_pairs(board_names, model_names, min_engines, max_engines, plot, custom_perf_data=None, plot_line = None):

    for board_name in board_names:
        hw_cfg = HWConfig(board_name)
        for model_name in model_names:
            model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
            first_layer = 0
            layers = utils.get_conv_layer_indices_in_range(
                model_dag, first_layer, len(model_dag))

            sum_ops = sum(utils.get_layers_op_counts_by_indices(
                model_dag, layers))
            print(board_name, model_name)
            # print('FLOPs:', 2 * sum_ops / 1000000000)
            # print('ideal:', (sum_ops / (hw_cfg.num_pes * hw_cfg.frequency)) * (1000))

            # mappings.append(SEMLMapping_LBL(model_dag, layers, num_pes))
            # mappings.append(SEMLMapping_FUSED(model_dag, layers, num_pes))
            # mappings.append(SESLMapping(model_dag, layers, num_pes))
            estimated_exec_times = []
            estimated_throughputs = []
            on_chip_fms_buffer_szs = []
            on_chip_weights_buffer_szs = []
            on_chip_buffer_szs = []
            off_chip_fms_access_list = []
            off_chip_weight_access_list = []
            off_chip_access_list = []
            for num_engines in range(min_engines, max_engines + 1):
                mappings = []
                mappings_labels = []
                spliting_point = num_engines
                mappings.append(SegmentMapping(
                    hw_cfg, model_dag, layers, num_engines))
                mappings.append(SegmentMappingRR(
                    hw_cfg, model_dag, layers, num_engines))
                mappings.append(HybridMapping(
                    hw_cfg, model_dag, layers, spliting_point))
                # mappings.append(HybridRRMapping(hw_cfg, model_dag, layers, num_engines))

                if num_engines == min_engines:
                    for i in range(len(mappings)):
                        estimated_exec_times.append([])
                        on_chip_fms_buffer_szs.append([])
                        on_chip_weights_buffer_szs.append([])
                        on_chip_buffer_szs.append([])
                        off_chip_fms_access_list.append([])
                        off_chip_weight_access_list.append([])
                        off_chip_access_list.append([])
                        estimated_throughputs.append([])

                mapping_index = 0
                for mapping in mappings:
                    mappings_labels.append(mapping.get_label())
                    estimated_exec_times[mapping_index].append(
                        round(1000 * mapping.calc_exec_time(), 2))
                    estimated_throughputs[mapping_index].append(
                        mapping.calc_throughput())
                    on_chip_fms_buffer_sz = round(
                        mapping.calc_fms_buffer_sz() / constants.MiB, 2)
                    on_chip_fms_buffer_szs[mapping_index].append(
                        on_chip_fms_buffer_sz)
                    on_chip_weights_buffer_sz = round(
                        mapping.calc_weights_buffer_sz() / constants.MiB, 2)
                    on_chip_weights_buffer_szs[mapping_index].append(
                        on_chip_weights_buffer_sz)
                    on_chip_buffer_szs[mapping_index].append(round(
                        mapping.calc_on_chip_buffer_sz() / constants.MiB, 2))

                    off_chip_weight_access = round(
                        mapping.calc_off_chip_weights_access() / constants.MiB, 2)
                    off_chip_fms_access = round(
                        mapping.calc_off_chip_fms_access() / constants.MiB, 2)
                    off_chip_weight_access_list[mapping_index].append(
                        off_chip_weight_access)
                    off_chip_fms_access_list[mapping_index].append(
                        off_chip_fms_access)
                    off_chip_access_list[mapping_index].append(
                        off_chip_weight_access + off_chip_fms_access)

                    mapping_index += 1

            custom_engine_counts = None
            if custom_perf_data is not None:
                mappings_labels.append(CustomMapping.MAPPING_LABEL)
                custom_engine_counts = []
                estimated_exec_times.append([])
                on_chip_fms_buffer_szs.append([])
                on_chip_weights_buffer_szs.append([])
                on_chip_buffer_szs.append([])
                off_chip_fms_access_list.append([])
                off_chip_weight_access_list.append([])
                off_chip_access_list.append([])
                estimated_throughputs.append([])
                for perf_dict in custom_perf_data:
                    perf_record = PerformanceRecord()
                    perf_record.init_from_dict(perf_dict)
                    custom_engine_counts.append(perf_record.num_engines)
                    estimated_exec_times[-1].append(perf_record.latency)
                    on_chip_fms_buffer_szs[-1].append(
                        perf_record.on_chip_buffer_fms)
                    on_chip_weights_buffer_szs[-1].append(
                        perf_record.on_chip_buffer_weights)
                    on_chip_buffer_szs[-1].append(perf_record.on_chip_buffer)
                    off_chip_fms_access_list[-1].append(
                        perf_record.off_chip_access_fms)
                    off_chip_weight_access_list[-1].append(
                        perf_record.off_chip_access_weights)
                    off_chip_access_list[-1].append(
                        perf_record.off_chip_access)
                    estimated_throughputs[-1].append(perf_record.throughput)

            if plot:
                plot_time_on_chip_buffers(estimated_exec_times, on_chip_fms_buffer_szs, on_chip_weights_buffer_szs,
                                          on_chip_buffer_szs,
                                          'MiB',
                                          mappings_labels, board_name, model_name,
                                          custom_engine_counts=custom_engine_counts,
                                          plot_line=plot_line)

                plot_time_off_chip_access(estimated_exec_times, off_chip_fms_access_list, off_chip_weight_access_list,
                                          off_chip_access_list,
                                          'MiB',
                                          mappings_labels, board_name, model_name,
                                          custom_engine_counts=custom_engine_counts,plot_line=plot_line)

                plot_time_on_chip_buffers(estimated_throughputs, on_chip_fms_buffer_szs, on_chip_weights_buffer_szs,
                                          on_chip_buffer_szs,
                                          'MiB',
                                          mappings_labels, board_name, model_name, False,
                                          custom_engine_counts=custom_engine_counts, plot_line=plot_line)

                plot_time_off_chip_access(estimated_throughputs, off_chip_fms_access_list, off_chip_weight_access_list,
                                          off_chip_access_list,
                                          'MiB',
                                          mappings_labels, board_name, model_name, False,
                                          custom_engine_counts=custom_engine_counts, plot_line=plot_line)


def engines_and_metrics(board_names, model_names, min_engines, max_engines, plot, print_results=False, exec_v2 = True):
    performance_records = []
    bests = {}
    bests_engine = {}
    for board_name in board_names:
        hw_cfg = HWConfig(board_name)
        bests[board_name] = {}
        bests_engine[board_name] = {}
        for model_name in model_names:
            bests[board_name][model_name] = {}
            bests_engine[board_name][model_name] = {}
            print(board_name, model_name)
            model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
            first_layer = 0
            layers = utils.get_conv_layer_indices_in_range(
                model_dag, first_layer, len(model_dag))

            estimated_exec_times = []
            estimated_throughputs = []
            on_chip_fms_buffer_szs = []
            on_chip_weights_buffer_szs = []
            on_chip_buffer_szs = []
            off_chip_fms_access_list = []
            off_chip_weight_access_list = []
            off_chip_access_list = []
            engine_counts = []
            for num_engines in range(min_engines, max_engines + 1):
                mappings = []
                mappings_labels = []
                spliting_point = num_engines
                mappings.append(SegmentMapping(
                    hw_cfg, model_dag, layers, num_engines, exec_v2 = exec_v2))
                mappings.append(SegmentMappingRR(
                    hw_cfg, model_dag, layers, num_engines))
                mappings.append(HybridMapping(
                    hw_cfg, model_dag, layers, spliting_point, exec_v2 = exec_v2))

                if num_engines == min_engines:
                    for i in range(len(mappings)):
                        estimated_exec_times.append([])
                        estimated_throughputs.append([])
                        on_chip_fms_buffer_szs.append([])
                        on_chip_weights_buffer_szs.append([])
                        on_chip_buffer_szs.append([])
                        off_chip_fms_access_list.append([])
                        off_chip_weight_access_list.append([])
                        off_chip_access_list.append([])
                        engine_counts.append([])

                mapping_index = 0
                for mapping in mappings:
                    mappings_labels.append(mapping.get_label())
                    if num_engines == min_engines:
                        bests[board_name][model_name][mapping.get_label()] = {}
                        bests_engine[board_name][model_name][mapping.get_label()] = {}

                    estimated_exec_times[mapping_index].append(
                        round(1000 * mapping.calc_exec_time(), 2))
                    estimated_throughputs[mapping_index].append(
                        mapping.calc_throughput())
                    on_chip_fms_buffer_sz = round(
                        mapping.calc_fms_buffer_sz() / constants.MiB, 2)
                    on_chip_fms_buffer_szs[mapping_index].append(
                        on_chip_fms_buffer_sz)
                    on_chip_weights_buffer_sz = round(
                        mapping.calc_weights_buffer_sz() / constants.MiB, 2)
                    on_chip_weights_buffer_szs[mapping_index].append(
                        on_chip_weights_buffer_sz)
                    on_chip_buffer_szs[mapping_index].append(round(
                        mapping.calc_on_chip_buffer_sz() / constants.MiB, 2))

                    off_chip_weight_access = round(
                        mapping.calc_off_chip_weights_access() / constants.MiB, 2)
                    off_chip_fms_access = round(
                        mapping.calc_off_chip_fms_access() / constants.MiB, 2)
                    off_chip_weight_access_list[mapping_index].append(
                        off_chip_weight_access)
                    off_chip_fms_access_list[mapping_index].append(
                        off_chip_fms_access)
                    off_chip_access_list[mapping_index].append(
                        off_chip_weight_access + off_chip_fms_access)

                    engine_counts[mapping_index].append(num_engines)

                    performance_records.append(PerformanceRecord(board_name, model_name, mapping.get_label(), num_engines,
                                                                 estimated_exec_times[mapping_index][-1],
                                                                 estimated_throughputs[mapping_index][-1],
                                                                 on_chip_fms_buffer_sz, on_chip_weights_buffer_sz,
                                                                 on_chip_buffer_szs[mapping_index][-1],
                                                                 off_chip_fms_access, off_chip_weight_access))

                    latency_label = constants.metric_display_names[Metrics.LATENCY]
                    throughput_label = constants.metric_display_names[Metrics.THROUGHPUT]
                    access_label = constants.metric_display_names[Metrics.ACCESS]
                    buffer_label = constants.metric_display_names[Metrics.BUFFER]

                    if latency_label not in bests[board_name][model_name][mapping.get_label()]:
                        bests_engine[board_name][model_name][mapping.get_label()][latency_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()][latency_label] = \
                            estimated_exec_times[mapping_index][-1]

                    if estimated_exec_times[mapping_index][-1] < bests[board_name][model_name][mapping.get_label()][latency_label]:
                        bests_engine[board_name][model_name][mapping.get_label()][latency_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()][latency_label] = \
                            estimated_exec_times[mapping_index][-1]

                    if throughput_label not in bests[board_name][model_name][mapping.get_label()]:
                        bests_engine[board_name][model_name][mapping.get_label()][throughput_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()][throughput_label] = \
                            estimated_throughputs[mapping_index][-1]

                    if estimated_throughputs[mapping_index][-1] > bests[board_name][model_name][mapping.get_label()][throughput_label]:
                        bests_engine[board_name][model_name][mapping.get_label()][throughput_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()][throughput_label] = \
                            estimated_throughputs[mapping_index][-1]

                    total_access = off_chip_fms_access + off_chip_weight_access
                    if access_label not in bests[board_name][model_name][mapping.get_label()]:
                        bests_engine[board_name][model_name][mapping.get_label()][access_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()
                                                      ][access_label] = total_access
                        bests[board_name][model_name][mapping.get_label()]['FMs access'] = \
                            off_chip_fms_access
                        bests[board_name][model_name][mapping.get_label()]['Weights access'] = \
                            off_chip_weight_access

                    if total_access < bests[board_name][model_name][mapping.get_label()][access_label]:
                        bests_engine[board_name][model_name][mapping.get_label()][access_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()
                                                      ][access_label] = total_access
                        bests[board_name][model_name][mapping.get_label()]['FMs access'] = \
                            off_chip_fms_access
                        bests[board_name][model_name][mapping.get_label()]['Weights access'] = \
                            off_chip_weight_access

                    total_buffer = on_chip_fms_buffer_sz + on_chip_weights_buffer_sz
                    if buffer_label not in bests[board_name][model_name][mapping.get_label()]:
                        bests_engine[board_name][model_name][mapping.get_label()][buffer_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()
                                                      ][buffer_label] = total_buffer
                        bests[board_name][model_name][mapping.get_label()]['FMs buffers'] = \
                            on_chip_fms_buffer_sz
                        bests[board_name][model_name][mapping.get_label()]['Weights buffers'] = \
                            on_chip_weights_buffer_sz

                    # the buffers that guarantee minimum access
                    if total_access <= bests[board_name][model_name][mapping.get_label()][access_label] and \
                        total_buffer < bests[board_name][model_name][mapping.get_label()
                                                      ][buffer_label]:
                        bests_engine[board_name][model_name][mapping.get_label()][buffer_label] = num_engines
                        bests[board_name][model_name][mapping.get_label()
                                                      ][buffer_label] = total_buffer
                        bests[board_name][model_name][mapping.get_label()]['FMs buffers'] = \
                            on_chip_fms_buffer_sz
                        bests[board_name][model_name][mapping.get_label()]['Weights buffers'] = \
                            on_chip_weights_buffer_sz

                    if print_results:
                        print(
                            '************{} {}****************'.format(model_name, num_engines))
                        print(mapping.get_label())
                        print('estimated exec time:', 1000 *
                              mapping.calc_exec_time(), 'ms')
                        print('fms buffers sz:',
                              mapping.calc_fms_buffer_sz() / constants.MiB, 'MiB')
                        print('weights buffer sz:', mapping.calc_weights_buffer_sz(
                        ) / constants.MiB, 'MiB')
                        print('off chip weight access:', mapping.calc_off_chip_weights_access(
                        ) / constants.MiB, 'MiB')
                        print('off chip fms access:', mapping.calc_off_chip_fms_access(
                        ) / constants.MiB, 'MiB')

                    mapping_index += 1

            if plot:
                plot_engines_and_metrics(engine_counts,
                                         estimated_exec_times, 'ms',
                                         on_chip_fms_buffer_szs, on_chip_weights_buffer_szs, on_chip_buffer_szs, 'MiB',
                                         off_chip_fms_access_list, off_chip_weight_access_list, off_chip_access_list, 'MiB',
                                         mappings_labels, board_name, model_name, plot_mode='inter', normalized=False)

                plot_engines_and_metrics(engine_counts,
                                         estimated_exec_times, 'ms',
                                         on_chip_fms_buffer_szs, on_chip_weights_buffer_szs, on_chip_buffer_szs, 'MiB',
                                         off_chip_fms_access_list, off_chip_weight_access_list, off_chip_access_list, 'MiB',
                                         mappings_labels, board_name, model_name, plot_mode='sep', normalized=True)

    return performance_records, bests_engine, bests


def engines_and_metrics_mapping(board_names, model_names, min_engines, max_engines, mapping_labels, plot, print_results=True,
                                print_details=False,
                                custom_metrics=None):
    performance_records = []
    for board_name in board_names:
        hw_cfg = HWConfig(board_name)
        for model_name in model_names:
            print(board_name, model_name)
            model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
            first_layer = 0
            layers = utils.get_conv_layer_indices_in_range(
                model_dag, first_layer, len(model_dag))

            estimated_exec_times = []
            estimated_throughputs = []
            on_chip_fms_buffer_szs = []
            on_chip_weights_buffer_szs = []
            on_chip_buffer_szs = []
            off_chip_fms_access_list = []
            off_chip_weight_access_list = []
            off_chip_access_list = []
            engine_counts = []
            for num_engines in range(min_engines, max_engines + 1):
                mappings = []
                mappings_labels = []
                spliting_point = num_engines
                if HybridMapping.MAPPING_LABEL in mapping_labels:
                    mappings.append(HybridMapping(
                        hw_cfg, model_dag, layers, spliting_point))
                if SegmentMapping.MAPPING_LABEL in mapping_labels:
                    mappings.append(SegmentMapping(
                        hw_cfg, model_dag, layers, num_engines))
                if SegmentMappingRR.MAPPING_LABEL in mapping_labels:
                    mappings.append(SegmentMappingRR(
                        hw_cfg, model_dag, layers, num_engines))

                if num_engines == min_engines:
                    for i in range(len(mappings)):
                        estimated_exec_times.append([])
                        estimated_throughputs.append([])
                        on_chip_fms_buffer_szs.append([])
                        on_chip_weights_buffer_szs.append([])
                        on_chip_buffer_szs.append([])
                        off_chip_fms_access_list.append([])
                        off_chip_weight_access_list.append([])
                        off_chip_access_list.append([])
                        engine_counts.append([])

                mapping_index = 0
                for mapping in mappings:
                    mappings_labels.append(mapping.get_label())
                    segment_exec_times = mapping.get_segment_exec_times()

                    estimated_exec_times[mapping_index].append(
                        round(1000 * mapping.calc_exec_time(), 2))
                    estimated_throughputs[mapping_index].append(
                        mapping.calc_throughput())
                    on_chip_fms_buffer_sz = round(
                        mapping.calc_fms_buffer_sz() / constants.MiB, 2)
                    on_chip_fms_buffer_szs[mapping_index].append(
                        on_chip_fms_buffer_sz)
                    on_chip_weights_buffer_sz = round(
                        mapping.calc_weights_buffer_sz() / constants.MiB, 2)
                    on_chip_weights_buffer_szs[mapping_index].append(
                        on_chip_weights_buffer_sz)
                    on_chip_buffer_szs[mapping_index].append(round(
                        mapping.calc_on_chip_buffer_sz() / constants.MiB, 2))

                    off_chip_weight_access = round(
                        mapping.calc_off_chip_weights_access() / constants.MiB, 2)
                    off_chip_fms_access = round(
                        mapping.calc_off_chip_fms_access() / constants.MiB, 2)
                    off_chip_weight_access_list[mapping_index].append(
                        off_chip_weight_access)
                    off_chip_fms_access_list[mapping_index].append(
                        off_chip_fms_access)
                    off_chip_access_list[mapping_index].append(
                        off_chip_weight_access + off_chip_fms_access)

                    engine_counts[mapping_index].append(num_engines)

                    performance_records.append(PerformanceRecord(board_name, model_name, mapping.get_label(), num_engines,
                                                                 estimated_exec_times[mapping_index][-1],
                                                                 estimated_throughputs[mapping_index][-1],
                                                                 on_chip_fms_buffer_sz, on_chip_weights_buffer_sz,
                                                                 on_chip_buffer_szs[mapping_index][-1],
                                                                 off_chip_fms_access, off_chip_weight_access, segment_exec_times=segment_exec_times))

                    if print_results:
                        print(
                            '************{} {}****************'.format(model_name, num_engines))
                        print(mapping.get_label())
                        if custom_metrics == None or Metrics.LATENCY in custom_metrics:
                            if SegmentMappingRR.MAPPING_LABEL in mapping_labels:
                                print(mapping.calc_segments_exec_times())
                            print('estimated exec time:', 1000 *
                                  mapping.calc_exec_time(print_details), 'ms')
                        fms_buffer_sz = mapping.calc_fms_buffer_sz() / constants.MiB
                        if custom_metrics == None or Metrics.BUFFER in custom_metrics:
                            print('fms buffers sz:', fms_buffer_sz, 'MiB')
                        weights_buffer_sz = mapping.calc_weights_buffer_sz() / constants.MiB
                        if custom_metrics == None or Metrics.BUFFER in custom_metrics:
                            print('weights buffer sz:',
                                  weights_buffer_sz, 'MiB')
                            print(
                                'on-chip buffer sz:', round(mapping.calc_on_chip_buffer_sz() / constants.MiB, 2), 'MiB')
                        if custom_metrics == None or Metrics.ACCESS in custom_metrics:
                            print('off chip weight access:', mapping.calc_off_chip_weights_access(
                            ) / constants.MiB, 'MiB')
                            print('off chip fms access:', mapping.calc_off_chip_fms_access(
                            ) / constants.MiB, 'MiB')

    return performance_records


def overleaf_table_out(board_names, model_names, min_engines, max_engines):
    plot = False
    results_list, _, _ = engines_and_metrics(board_names, model_names,
                                          min_engines, max_engines, plot)

    results_table = []
    normalized_table = []
    normalized_table.append(
        ['mapping', 'latency', 'throughput', 'on-chip buffets', 'off-chip access'])
    results_list.sort(key=lambda x: x.latency)
    results_table.append(results_list[0].totals_as_list())
    results_list.sort(key=lambda x: x.throughput, reverse=True)
    results_table.append(results_list[0].totals_as_list())
    results_list.sort(key=lambda x: x.on_chip_buffer)
    results_table.append(results_list[0].totals_as_list())
    results_list.sort(key=lambda x: x.off_chip_access)
    results_table.append(results_list[0].totals_as_list())

    best_results_of_all = [0]
    for i in range(len(results_table)):
        for j in range(1, len(results_table[0])):
            if i == 0:
                best_results_of_all.append(results_table[i][j])
            else:
                if normalized_table[0][j] == 'throughput':
                    best_results_of_all[j] = max(
                        best_results_of_all[j], results_table[i][j])
                else:
                    best_results_of_all[j] = min(
                        best_results_of_all[j], results_table[i][j])

    for i in range(len(results_table)):
        normalized_table.append([results_table[i][0]])
        for j in range(1, len(results_table[0])):
            normalized_table[-1].append(round(results_table[i]
                                        [j] / best_results_of_all[j], 2))

    print(mapping_general_utils.overleaf_table_format(normalized_table))
