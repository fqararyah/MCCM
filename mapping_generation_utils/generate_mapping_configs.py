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
import experiments as exps
import os
from preformance_record import *


def generate_mapping_configs(board_names, model_names, min_engines, max_engines, mapping_labels, 
                             configs_path = '../code_gen/mappings_configs'):

    if not os.path.exists(configs_path):
        os.mkdir(configs_path)
    for board_name in board_names:
        hw_cfg = HWConfig(board_name)
        for model_name in model_names:
            print(board_name, model_name)
            bests = {}
            bests_engines = {}
            model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
            first_layer = 0
            layers = utils.get_conv_layer_indices_in_range(
                model_dag, first_layer, len(model_dag))

            for num_engines in range(min_engines, max_engines + 1):
                mappings = []
                spliting_point = num_engines
                for label in mapping_labels:
                    if HybridMapping.MAPPING_LABEL == label:
                        mappings.append(HybridMapping(
                            hw_cfg, model_dag, layers, spliting_point))
                        if HybridMapping.MAPPING_LABEL not in bests:
                            bests[HybridMapping.MAPPING_LABEL] = {}
                            bests_engines[HybridMapping.MAPPING_LABEL] = {}
                    elif SegmentMapping.MAPPING_LABEL == label:
                        mappings.append(SegmentMapping(
                            hw_cfg, model_dag, layers, num_engines))
                        if SegmentMapping.MAPPING_LABEL not in bests:
                            bests[SegmentMapping.MAPPING_LABEL] = {}
                            bests_engines[SegmentMapping.MAPPING_LABEL] = {}
                    elif SegmentMappingRR.MAPPING_LABEL == label:
                        mappings.append(SegmentMappingRR(
                            hw_cfg, model_dag, layers, num_engines))
                        if SegmentMappingRR.MAPPING_LABEL not in bests:
                            bests[SegmentMappingRR.MAPPING_LABEL] = {}
                            bests_engines[SegmentMappingRR.MAPPING_LABEL] = {}

                for i in range(len(mappings)):
                    mapping = mappings[i]
                    mapping_label = mapping_labels[i]                    

                    mapping_configs_dir = configs_path + '/' + board_name
                    if not os.path.exists(mapping_configs_dir):
                        os.mkdir(mapping_configs_dir)

                    mapping_configs_dir += '/' + model_name
                    if not os.path.exists(mapping_configs_dir):
                        os.mkdir(mapping_configs_dir)

                    mapping_configs_dir += '/' + mapping_label
                    if not os.path.exists(mapping_configs_dir):
                        os.mkdir(mapping_configs_dir)
                    par_file_name = str(num_engines) + '_parallelism.txt'
                    tmp_channels_file = str(
                        num_engines) + '_tmp_channels_off_chip.txt'
                    engines_weights_in_chip_file = str(
                        num_engines) + '_engines_weights_on_chip.txt'
                    engines_inter_seg_buffers_file = str(
                        num_engines) + '_inter_seg_buffers.txt'
                    engines_weights_buffers_file = str(
                        num_engines) + '_engines_weights_buffers.txt'
                    engines_fms_buffers_file = str(
                        num_engines) + '_engines_fms_buffers.txt'
                    engine_layer_map_file = str(
                        num_engines) + '_engine_layer_map.txt'

                    with open(mapping_configs_dir + '/' + par_file_name, 'w') as f:
                        for engine in mapping.get_engines():
                            f.write(str(engine.get_parallelism_dims()) + '\n')
                    if mapping_label == HybridMapping.MAPPING_LABEL:
                        with open(mapping_configs_dir + '/' + tmp_channels_file, 'w') as f:
                            f.write(
                                str(mapping.get_off_chip_tmp_channels_layers()))
                    elif mapping_label == SegmentMapping.MAPPING_LABEL:
                        with open(mapping_configs_dir + '/' + tmp_channels_file, 'w') as f:
                            f.write(
                                str(mapping.get_off_chip_tmp_channels_layers()))
                        with open(mapping_configs_dir + '/' + engines_inter_seg_buffers_file, 'w') as f:
                            f.write(str(mapping.get_inter_group_fms_on_chip()))
                        with open(mapping_configs_dir + '/' + engines_fms_buffers_file, 'w') as f:
                            fms_buffers = mapping.get_max_eingines_fms_buffers_layers()
                            f.write(str(fms_buffers) + '\n')
                        with open(mapping_configs_dir + '/' + engine_layer_map_file, 'w') as f:
                            engine_layer_mapping = mapping.get_engine_layer_mapping()
                            for _engine, engine_layers in engine_layer_mapping.items():
                                f.write(
                                    str(engine_layers) + '\n')
                    elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                        with open(mapping_configs_dir + '/' + engines_weights_in_chip_file, 'w') as f:
                            f.write(str(mapping.get_layer_weights_on_chip()))
                        with open(mapping_configs_dir + '/' + engines_inter_seg_buffers_file, 'w') as f:
                            f.write(str(mapping.get_inter_group_fms_on_chip()))
                        with open(mapping_configs_dir + '/' + engines_weights_buffers_file, 'w') as f:
                            f.write(
                                str(mapping.get_max_eingines_weight_buffer_layers()))
                        with open(mapping_configs_dir + '/' + engines_fms_buffers_file, 'w') as f:
                            fms_buffers = mapping.get_max_eingines_fms_buffers_layers()
                            f.write(str(fms_buffers[0]) + '\n' + str(fms_buffers[1]) + '\n' +
                                    str(fms_buffers[2]) + '\n' + str(fms_buffers[3]) + '\n' +
                                    str(fms_buffers[4]) + '\n' + str(fms_buffers[5]))
                        with open(mapping_configs_dir + '/' + engine_layer_map_file, 'w') as f:
                            engine_layer_mapping = mapping.get_engine_layer_mapping()
                            engine_layer_mapping_arr = []
                            for _engine in engine_layer_mapping.keys():
                                engine_layer_mapping_arr.append([])
                            for _engine, engine_layers in engine_layer_mapping.items():
                                for engine_layer in engine_layers:
                                    engine_layer_mapping_arr[_engine].append(
                                        engine_layer)
                            for i in range(len(engine_layer_mapping_arr)):
                                f.write(
                                    str(engine_layer_mapping_arr[i]) + '\n')

                    mapping_record = mapping_general_utils.build_performance_record(
                        mapping, board_name, model_name, num_engines)

                    if Metrics.ACCESS not in bests[mapping_label]:
                        bests[mapping_label][Metrics.ACCESS] = mapping_record.off_chip_access
                        bests_engines[mapping_label][Metrics.ACCESS] = num_engines
                    elif not mapping_record.is_better(Metrics.ACCESS, bests[mapping_label][Metrics.ACCESS]):
                        bests[mapping_label][Metrics.ACCESS] = mapping_record.off_chip_access
                        bests_engines[mapping_label][Metrics.ACCESS] = num_engines

                    if Metrics.BUFFER not in bests[mapping_label]:
                        bests[mapping_label][Metrics.BUFFER] = mapping_record.on_chip_buffer
                        bests_engines[mapping_label][Metrics.BUFFER] = num_engines
                    elif not mapping_record.is_better(Metrics.BUFFER, bests[mapping_label][Metrics.BUFFER]):
                        bests[mapping_label][Metrics.BUFFER] = mapping_record.on_chip_buffer
                        bests_engines[mapping_label][Metrics.BUFFER] = num_engines

                    if Metrics.LATENCY not in bests[mapping_label]:
                        bests[mapping_label][Metrics.LATENCY] = mapping_record.latency
                        bests_engines[mapping_label][Metrics.LATENCY] = num_engines
                    elif not mapping_record.is_better(Metrics.LATENCY, bests[mapping_label][Metrics.LATENCY]):
                        bests[mapping_label][Metrics.LATENCY] = mapping_record.latency
                        bests_engines[mapping_label][Metrics.LATENCY] = num_engines

                    if Metrics.THROUGHPUT not in bests[mapping_label]:
                        bests[mapping_label][Metrics.THROUGHPUT] = mapping_record.throughput
                        bests_engines[mapping_label][Metrics.THROUGHPUT] = num_engines
                    elif not mapping_record.is_better(Metrics.THROUGHPUT, bests[mapping_label][Metrics.THROUGHPUT]):
                        bests[mapping_label][Metrics.THROUGHPUT] = mapping_record.throughput
                        bests_engines[mapping_label][Metrics.THROUGHPUT] = num_engines

                    with open(mapping_configs_dir + '/bests.txt', 'w') as f:
                        f.write(
                            'access::' + str(bests_engines[mapping_label][Metrics.ACCESS]) + '\n')
                        f.write(
                            'buffer::' + str(bests_engines[mapping_label][Metrics.BUFFER]) + '\n')
                        f.write(
                            'latency::' + str(bests_engines[mapping_label][Metrics.LATENCY]) + '\n')
                        f.write(
                            'throughput::' + str(bests_engines[mapping_label][Metrics.THROUGHPUT]) + '\n')


# model_names = ['resnet50', 'resnet152', 'mob_v2', 'dense121', 'xce_r']
# min_engines = 2
# max_engines = 11
# board_names = mapping_utils.read_board_names()
# mappings_to_generate_configs_of = [SegmentMappingRR.MAPPING_LABEL]


# generate_mapping_configs(board_names, model_names, min_engines,
#                          max_engines, mappings_to_generate_configs_of)
