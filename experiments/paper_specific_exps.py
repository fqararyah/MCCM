

from plot import plot_results
from mapping_types.generic_mapping import *
from mapping_types.sesl_mapping import *
from mapping_types.seml_mapping_lbl import *
from mapping_types.seml_mapping_fused import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from mapping_types.hybrid_mapping import *
from mapping_types.hybrid_rr_mapping import *
import __init__
import mapping_utils.mapping_general_utils as mapping_general_utils
from hw_config import *
import experiments.experiments as exps
from preformance_record import *
import constants
import mapping_generation_utils.generate_custom_mappings_json as gen_mappings
import os
import time


def fine_grained():
    board_name = 'zc706'
    hw_config = HWConfig(board_name)
    compute_mem = {'Compute time': [], 'Memory access time': []}
    for model_name in ['resnet50']:
        model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
        layers = utils.get_conv_layer_indices_in_range(
                model_dag, 0, len(model_dag))
        sum_diff = 0
        for num_engines in range(2, 2+ 1):
            mapping = SegmentMappingRR(hw_config, model_dag, layers, num_engines)
            exec_time = mapping.calc_exec_time()
            mapping_1_seg_exec_times = mapping.get_segment_comp_and_access_times()
            for segment_index in range(len(mapping_1_seg_exec_times)):
                seg_time = mapping_1_seg_exec_times[segment_index]
                ctc = seg_time[1] / seg_time[0]
                compute_mem['Compute time'].append(seg_time[0] / exec_time)
                compute_mem['Memory access time'].append(seg_time[1] / exec_time)
                if ctc > 1:
                    sum_diff += seg_time[1] - seg_time[0]
                    print(seg_time[0] / exec_time, seg_time[1] / exec_time, model_name, num_engines, segment_index)
            print(sum_diff / exec_time)

        engine_layer_map = mapping.engine_layer_map
        mapping_engines = mapping.get_engines()
        segments_underutilization = []
        segment_comnp_times = mapping.get_segment_comp_and_access_times()
        all_engines_pes = 0
        op_counts = utils.get_layers_op_counts(model_dag, 0)
        for engine in mapping_engines:
            all_engines_pes += engine.get_parallelism()
        for layer_group_index in range(mapping.num_layer_groups):
            ideal_time = sum(op_counts[layer_group_index * num_engines: (layer_group_index + 1) * num_engines])/ \
                (all_engines_pes * hw_config.frequency)
            segments_underutilization.append((segment_comnp_times[layer_group_index][0] - ideal_time)/ ideal_time)

        print('>>', segments_underutilization)

        min_underutilization = min(segments_underutilization)
        segments_underutilization = [segments_underutilization[i] / min_underutilization for i in range(len(segments_underutilization))]

    plot_results.plot_bar_groups(compute_mem, 'segrr_resnet50',
                                            x_title='Segment', x_ticks_dict={1: [*range(1, len(mapping_1_seg_exec_times) + 1)]},
                                            y_title= 'Time \n (% Overall)',
                                            relative_save_path='comp_mem/underutilization', percentage=True)

    weight_access = mapping.calc_off_chip_weights_access()
    fms_access = mapping.calc_off_chip_fms_access()
    all_access = weight_access + fms_access
    x_ticks_dict = {1:[mapping.get_label()]}
    accesses = {'Weights':[ weight_access / all_access], 'FMs': [ fms_access / all_access]}


    layers = utils.get_conv_layer_indices_in_range(
                model_dag, 0, len(model_dag))
    mapping = SegmentMapping(hw_config, model_dag, layers, 7, exec_v2=True)
    weight_access = mapping.calc_off_chip_weights_access()
    fms_access = mapping.calc_off_chip_fms_access()
    all_access = weight_access + fms_access
    x_ticks_dict[1].append(mapping.get_label())
    accesses['Weights'].append( weight_access / all_access)
    accesses['FMs'].append( fms_access / all_access)

    segment_comp_access_times = mapping.get_segment_comp_and_access_times()
    segment_exec_times = mapping.get_segment_exec_times()
    exec_time = mapping.calc_exec_time()
    compute_mem = {'Compute time': [], 'Memory access time': []}
    for segment in range(mapping.num_engines):
        seg_time = segment_comp_access_times[segment]
        compute_mem['Compute time'].append(seg_time[0] / exec_time)
        compute_mem['Memory access time'].append(seg_time[1] / exec_time)

    plot_results.plot_bar_groups(compute_mem, 'seg_resnet50',
                                            x_title='Segment', x_ticks_dict={1: [*range(1, mapping.num_engines + 1)]},
                                            y_title= 'Time \n (% Overall)',
                                            relative_save_path='comp_mem/underutilization', percentage=True)

    mapping = HybridMapping(hw_config, model_dag, layers, 9, exec_v2= True)
    weight_access = mapping.calc_off_chip_weights_access()
    fms_access = mapping.calc_off_chip_fms_access()
    all_access = weight_access + fms_access
    x_ticks_dict[1].append(mapping.get_label())
    accesses['Weights'].append( weight_access / all_access)
    accesses['FMs'].append( fms_access / all_access)

    plot_results.plot_bar_groups(accesses, 'seg_segrr_res50_zc706', x_ticks_dict=x_ticks_dict,
                                            x_title= 'Accesses breakdown',
                                            relative_save_path='access_breakdown', breakdown=True, 
                                            horizontal_bars=True, percentage=True)

def bottlenecks():
    board_name = 'inf_mem'
    model_name = 'mob_v2'
    hw_config = HWConfig(board_name)
    num_engines = 11
    model_dag = utils.read_model_dag_v2(
            constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
    layers = utils.get_conv_layer_indices_in_range(
            model_dag, 0, len(model_dag))
    mapping1 = SegmentMapping(hw_config, model_dag, layers, num_engines)
    segment_weights_on_chip = mapping1.get_segment_weights_buffer_sizes()
    print(segment_weights_on_chip)
    segment_fms_on_chip = mapping1.get_segment_fms_buffer_sizes()
    print(segment_weights_on_chip)
    segment_on_chip = mapping1.get_segment_buffer_sizes()
    print(segment_on_chip)
    #sum_on_chip = sum(segment_on_chip)
    #segment_weights_on_chip_normalized = [segment_weights_on_chip[i] / sum_on_chip for i in range(num_engines)]
    #segment_fms_on_chip_normalized = [segment_fms_on_chip[i] / sum_on_chip for i in range(num_engines)]

    segment_weights_on_chip = [segment_weights_on_chip[i] / constants.MiB for i in range(num_engines)]
    segment_fms_on_chip = [segment_fms_on_chip[i] / constants.MiB for i in range(num_engines)]
    
    plot_results.plot_bar_groups({'FMs': segment_fms_on_chip, 'Weights': segment_weights_on_chip}, 'segmented_buffers_11',
                                        x_title='Segment', x_ticks_dict={1: [*range(1, 12)]},  y_title= 'Buffers (MiB)',
                                        relative_save_path='bottlenecks/buffer', breakdown=True)
    
    exps.metric_pairs([board_name], [model_name], constants.MIN_ENGINES,
                      constants.MAX_ENGINES, plot=True, custom_perf_data=None, plot_line={'On-chip': 4})
    
def des_section_bottlenecks():
    board_name = 'vcu110'
    model_name = 'xce_r'
    hw_config = HWConfig(board_name)
    num_engines = 4
    model_dag = utils.read_model_dag_v2(
            constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
    layers = utils.get_conv_layer_indices_in_range(
            model_dag, 0, len(model_dag))
    mapping1 = SegmentMapping(hw_config, model_dag, layers, num_engines)
    mapping_1_seg_exec_times = mapping1.get_segment_exec_times()
    max_exec_time = max(mapping_1_seg_exec_times)
    mapping1_engines = mapping1.get_engines()

    engine_layer_map = mapping1.engine_layer_map
    segments_op_counts = []
    segments_underutilization = []
    for engine, layers in engine_layer_map.items():
        segments_op_counts.append(sum(utils.get_layers_op_counts_by_indices(model_dag, layers)))
        num_pes = mapping1_engines[2 * engine].num_pes + mapping1_engines[2 * engine + 1].num_pes
        ideal_time = segments_op_counts[engine] / (num_pes * hw_config.frequency)
        segments_underutilization.append((mapping_1_seg_exec_times[engine] - ideal_time)/ ideal_time)

    min_underutilization = min(segments_underutilization)
    segments_underutilization = [segments_underutilization[i] / min_underutilization for i in range(len(segments_underutilization))]

    plot_results.plot_bar_groups({ 1: segments_underutilization}, 'segmented_underutilization',
                                        x_title='Segment', x_ticks_dict={1: [1,2, 3, 4]},  y_title= 'Underutilization\n(normalized)',
                                        relative_save_path='custom/underutilization')

    segment_weights_on_chip = mapping1.get_segment_weights_buffer_sizes()
    segment_fms_on_chip = mapping1.get_segment_fms_buffer_sizes()
    segment_on_chip = mapping1.get_segment_buffer_sizes()
    sum_on_chip = sum(segment_on_chip)
    segment_weights_on_chip_normalized = [segment_weights_on_chip[i] / sum_on_chip for i in range(num_engines)]
    segment_fms_on_chip_normalized = [segment_fms_on_chip[i] / sum_on_chip for i in range(num_engines)]
    plot_results.plot_bar_groups({'FMs': segment_fms_on_chip_normalized, 'Weights': segment_weights_on_chip_normalized}, 'segmented_buffers',
                                        x_title='Segment', x_ticks_dict={1: [1, 2, 3, 4]},  y_title= 'Buffers\n(normalized)',
                                        relative_save_path='custom/underutilization', breakdown=True)

    layers = utils.get_conv_layer_indices_in_range(
            model_dag, 0, len(model_dag))
    num_engines = 7
    mapping1 = HybridMapping(hw_config, model_dag, layers, num_engines)
    mapping_1_seg_exec_times = mapping1.get_segment_exec_times()
    mapping1_engines = mapping1.get_engines()

    engine_layer_map = {}
    layer_op_counts = utils.get_layers_op_counts(model_dag, 0)
    part_1_pes = mapping1.first_part_hw_config.num_pes
    part_2_pes = mapping1.second_part_hw_config.num_pes
    part_1_ops = sum(layer_op_counts[:num_engines])
    part_2_ops = sum(layer_op_counts[num_engines:])

    segments_underutilization = []
    ideal_time = part_1_ops / (part_1_pes * hw_config.frequency)
    segments_underutilization.append((mapping_1_seg_exec_times[0] - ideal_time)/ ideal_time)
    ideal_time = part_2_ops / (part_2_pes * hw_config.frequency)
    segments_underutilization.append((mapping_1_seg_exec_times[1] - ideal_time)/ ideal_time)
    min_underutilization = min(segments_underutilization)
    normalized_under = []
    for i in range(len(segments_underutilization)):
        normalized_under.append(segments_underutilization[i] / min_underutilization)
        

    plot_results.plot_bar_groups({ 1: normalized_under}, 'hybrid_underutilization',
                                        x_title='Segment', x_ticks_dict={1: [1,2]},  y_title= 'Underutilization\n(normalized)',
                                        relative_save_path='custom/underutilization')

    segment_weights_on_chip = mapping1.get_segment_weights_buffer_sizes()
    segment_fms_on_chip = mapping1.get_segment_fms_buffer_sizes()
    segment_on_chip = mapping1.get_segment_buffer_sizes()
    sum_on_chip = sum(segment_on_chip)
    segment_weights_on_chip_normalized = [segment_weights_on_chip[i] / sum_on_chip for i in range(2)]
    segment_fms_on_chip_normalized = [segment_fms_on_chip[i] / sum_on_chip for i in range(2)]

    plot_results.plot_bar_groups({'FMs': segment_fms_on_chip_normalized, 'Weights': segment_weights_on_chip_normalized}, 'hybrid_buffers',
                                        x_title='Segment', x_ticks_dict={1: [1,2]},  y_title= 'Buffers\n(normalized)',
                                        relative_save_path='custom/underutilization', breakdown=True)
    
    
def des_section_bottlenecks_v2():
    board_name = 'vcu110'
    model_name = 'xce_r'
    hw_config = HWConfig(board_name)
    mapping_1_num_engines = 4
    model_dag = utils.read_model_dag_v2(
            constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
    layers = utils.get_conv_layer_indices_in_range(
            model_dag, 0, len(model_dag))
    mapping1 = SegmentMapping(hw_config, model_dag, layers, mapping_1_num_engines)
    mapping_1_seg_exec_times = mapping1.get_segment_exec_times()
    mapping1_engines = mapping1.get_engines()

    engine_layer_map = mapping1.engine_layer_map
    segments_op_counts = []
    mapping_1_segments_underutilization = []
    for engine, layers in engine_layer_map.items():
        segments_op_counts.append(sum(utils.get_layers_op_counts_by_indices(model_dag, layers)))
        num_pes = mapping1_engines[2 * engine].num_pes + mapping1_engines[2 * engine + 1].num_pes
        ideal_time = segments_op_counts[engine] / (num_pes * hw_config.frequency)
        print(mapping_1_seg_exec_times[engine], ideal_time)
        mapping_1_segments_underutilization.append((mapping_1_seg_exec_times[engine] - ideal_time)/ ideal_time)


    mapping_1_segment_weights_on_chip = mapping1.get_segment_weights_buffer_sizes()
    mapping_1_segment_fms_on_chip = mapping1.get_segment_fms_buffer_sizes()
    mapping_1_segment_on_chip = mapping1.get_segment_buffer_sizes()

    layers = utils.get_conv_layer_indices_in_range(
            model_dag, 0, len(model_dag))
    mapping_2_num_engines = 7
    mapping2 = HybridMapping(hw_config, model_dag, layers, mapping_2_num_engines)
    mapping_2_seg_exec_times = mapping2.get_segment_exec_times()
    mapping_1_label = mapping1.get_label()
    mapping_2_label = mapping2.get_label()
    
    engine_layer_map = {}
    layer_op_counts = utils.get_layers_op_counts(model_dag, 0)
    part_1_pes = mapping2.first_part_hw_config.num_pes
    part_2_pes = mapping2.second_part_hw_config.num_pes
    part_1_ops = sum(layer_op_counts[:mapping_2_num_engines])
    part_2_ops = sum(layer_op_counts[mapping_2_num_engines:])

    mapping_2_segments_underutilization = []
    ideal_time = part_1_ops / (part_1_pes * hw_config.frequency)
    mapping_2_segments_underutilization.append((mapping_2_seg_exec_times[0] - ideal_time)/ ideal_time)
    print(mapping_2_seg_exec_times[0], ideal_time)
    ideal_time = part_2_ops / (part_2_pes * hw_config.frequency)
    mapping_2_segments_underutilization.append((mapping_2_seg_exec_times[1] - ideal_time)/ ideal_time)
    print(mapping_2_seg_exec_times[1], ideal_time)

    min_underutilization = min(min(mapping_1_segments_underutilization), min(mapping_2_segments_underutilization))
    
    for i in range(mapping_1_num_engines - 2):#2 is number of hybrid segments
        mapping_2_segments_underutilization.append(0)
        
    normalized_underutilization = {mapping_1_label: [], mapping_2_label: []}
    for i in range(len(mapping_1_segments_underutilization)):
        normalized_underutilization[mapping_1_label].append(mapping_1_segments_underutilization[i] / min_underutilization)
        normalized_underutilization[mapping_2_label].append(mapping_2_segments_underutilization[i] / min_underutilization)
        

    plot_results.plot_bar_groups(normalized_underutilization, 'segmetend_hybrid_underutilization',
                                 x_ticks_dict={1: [1,2, 3, 4]},  y_title= 'Underutilization\n(normalized)',
                                        relative_save_path='custom/underutilization')

    mapping_2_segment_weights_on_chip = mapping2.get_segment_weights_buffer_sizes()
    mapping_2_segment_fms_on_chip = mapping2.get_segment_fms_buffer_sizes()
    mapping_2_segment_on_chip = mapping2.get_segment_buffer_sizes()
    sum_on_chip = sum(mapping_1_segment_on_chip)
        
    for i in range(mapping_1_num_engines - 2):#2 is number of hybrid segments
        mapping_2_segment_on_chip.append(0)
        
    segment_on_chip_normalized = {}
    for i in range(mapping_1_num_engines):
        segment_on_chip_normalized['Seg' + str(i + 1)] = []
        
    for i in range(len(mapping_1_segments_underutilization)):
        segment_on_chip_normalized['Seg' + str(i + 1)].append(mapping_1_segment_on_chip[i] / sum_on_chip)
        segment_on_chip_normalized['Seg' + str(i + 1)].append(mapping_2_segment_on_chip[i] / sum_on_chip)
    
    plot_results.plot_bar_groups(segment_on_chip_normalized, 'segmetend_hybrid_buffers', x_ticks_dict={1: [mapping_1_label, mapping_2_label]},
                                 y_title= 'Buffers\n(normalized)', relative_save_path='custom/underutilization', breakdown=True)
    
    
def generate_best_mappings_table(board_names, model_names, min_engines, max_engines, metrics):
    best_mappings = exps.get_best_mappings(board_names, model_names,
                                       min_engines, max_engines, metrics, exec_v2 = True)

    board_names_str = '\multicolumn{1}{c|}{}'
    model_names_str = '\multicolumn{1}{c|}{}'
    performance_str = ''

    mappings_colors = {'Segmented': 'seg_color!70','Hybrid': 'hyb_color!70','SegmentedRR': 'segrr_color!70'}
    mapping_key_list = ['Segmented','SegmentedRR', 'Hybrid']
    num_mappings = len(mapping_key_list)
    board_num = 0
    for board_name in board_names:
        board_num += 1
        board_names_str += ' &\multicolumn{' + str(len(model_names)) +'}{c|}{\cellcolor{gray!10}' + board_name + '}'
        if board_num < len(board_names):
            board_names_str += ' & '
        for model_name in model_names:
            model_names_str += ' & \cellcolor{gray!10} \\rotatebox[origin=c]{90}{' + constants.model_display_names[model_name] + '}'
        if board_num < len(board_names):
            model_names_str += ' & '
            
    board_names_str += '\\\\ \\hhline{~-----------------------}'        
    model_names_str += '\\\\ \\hline'        

    metric_num = 0
    for metric in metrics:
        mapping_num = 0
        for mapping_key in mapping_key_list:
            performance_str += '\cellcolor{gray!10}'
            mapping_num += 1
            if mapping_num == num_mappings:
                performance_str += '\multirow{' + str(- num_mappings) + '}{*}{\cellcolor{gray!10}' + metric + '}'
            board_num = 0
            for board_name in board_names:
                for model_name in model_names:
                    if mapping_key in best_mappings[board_name][model_name][metric]:
                        num_engines_index = best_mappings[board_name][model_name][metric].index(mapping_key)
                        num_engines = best_mappings[board_name][model_name][metric][num_engines_index - 1]
                        performance_str += ' &' + '\cellcolor{' + mappings_colors[mapping_key] + '}' + str(num_engines)
                    else:
                        performance_str += ' & '
                    
                board_num += 1
                if board_num < len(board_names):
                    performance_str += ' & '
            performance_str += '\\\\ \n'
        performance_str += '\\hline \n'
        metric_num += 1
        
    print(board_names_str)
    print(model_names_str)
    print(performance_str)