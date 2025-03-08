
from .generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *
from .seml_mapping_lbl import *
from .seml_mapping_fused import *
from .sesl_mapping import *
from .segment_grained_mapping_rr import *
from hw_config import *
import copy
import mapping_utils.mapping_general_utils as mapping_utils
from basic_mapping import BasicMapping


def infer_mapping_types(mapping_details, model_dag):
    mappings_segments_list = []
    for key, val in mapping_details.items():
        engine_list = mapping_utils.clean_engine_range(val)
        layer_lits = mapping_utils.clean_layer_range(key, model_dag)
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

class CustomMapping(BasicMapping):
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'Custom'

    def __init__(self, hw_config, model_dag, layers, mappings_segments_config_list,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 apply_fusion=False):

        super().__init__(hw_config, model_dag, layers, [])
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.model_dag = model_dag
        self.num_layers = len(layers)
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.apply_fusion = apply_fusion
        self.mappings_segments_config_list = mappings_segments_config_list
        self.num_segments = len(mappings_segments_config_list)
        self.mapping_list = []
        self.initialize_mappings()

    def balance_pes(self):
        overall_op_count = sum(utils.get_layers_op_counts(self.model_dag))
        for _, layer_set in self.mappings_layers_dict.items():
            current_op_count = sum(
                utils.get_layers_op_counts_by_indices(self.model_dag, layer_set))
            self.mappings_hw_configs.append(copy.deepcopy(self.hw_config))
            current_pes = (current_op_count *
                           self.hw_config.num_pes) // overall_op_count
            self.mappings_hw_configs[-1].num_pes = current_pes

    def initialize_tmp_mappings(self):
        segment_index = 0
        tmp_mappings = []
        for config_dict in self.mappings_segments_config_list:
            first_layer_ifms_are_on_chip = segment_index > 0 or self.first_layer_ifms_are_on_chip
            last_layer_ofms_are_on_chip = segment_index < self.num_segments - 1 or self.last_layer_ofms_are_on_chip
            mapping_label = config_dict['mapping']
            engine_list = config_dict['engine_list']
            layer_list = config_dict['layer_list']
            if mapping_label == SESLMapping.MAPPING_LABEL:
                tmp_mappings.append(SESLMapping(self.mappings_hw_configs[segment_index],
                                                self.model_dag, layer_list,
                                                first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                engine_parallelization_strategy=ParallelizationStrategies.OFMS_W))
            elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                tmp_mappings.append(SegmentMappingRR(
                    self.mappings_hw_configs[segment_index], self.model_dag, layer_list, len(
                        engine_list),
                    first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                    last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip))
            elif mapping_label == SEMLMapping_LBL.MAPPING_LABEL:
                tmp_mappings.append(SEMLMapping_LBL(self.mappings_hw_configs[segment_index],
                                                    self.model_dag, layer_list,
                                                    first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                    last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip))
            segment_index += 1

        return tmp_mappings

    def distribute_on_chip_memory(self, tmp_mappings):
        segment_buffers = []
        for _, segments_index_list in self.first_engine_segment_index_dict.items():
            current_segment_max_buffer_sz = 0
            for segment_index in segments_index_list:
                current_segment_max_buffer_sz = max(current_segment_max_buffer_sz,
                                                    tmp_mappings[segment_index].calc_on_chip_buffer_sz_pure())
            segment_buffers.append(current_segment_max_buffer_sz)

        total_segment_buffers = sum(segment_buffers)
        for i in range(len(segment_buffers)):
            self.mappings_hw_configs[i].on_chip_memory = (
                segment_buffers[i] * self.hw_config.on_chip_memory) // total_segment_buffers

    def initialize_mappings(self):
        self.segment_first_engine_dict = {}
        self.segment_engine_set_list = []
        self.segment_layer_set_list = []
        self.first_engine_segment_index_dict = {}
        self.mappings_layers_dict = {}
        self.mappings_hw_configs = []
        self.mappings = []

        segment_index = 0
        for config_dict in self.mappings_segments_config_list:
            engine_list = config_dict['engine_list']
            layer_list = config_dict['layer_list']
            first_engine_id = engine_list[0]
            self.segment_first_engine_dict[segment_index] = first_engine_id
            self.segment_engine_set_list.append(engine_list)
            self.segment_layer_set_list.append(layer_list)
            if segment_index not in self.first_engine_segment_index_dict:
                self.first_engine_segment_index_dict[first_engine_id] = []
                self.mappings_layers_dict[first_engine_id] = []
            self.first_engine_segment_index_dict[first_engine_id].append(
                segment_index)
            self.mappings_layers_dict[first_engine_id].extend(layer_list)

            segment_index += 1

        self.balance_pes()
        tmp_mappings = self.initialize_tmp_mappings()
        self.distribute_on_chip_memory(tmp_mappings)

        segment_index = 0
        for config_dict in self.mappings_segments_config_list:
            first_layer_ifms_are_on_chip = segment_index > 0 or self.first_layer_ifms_are_on_chip
            last_layer_ofms_are_on_chip = segment_index < self.num_segments - 1 or self.last_layer_ofms_are_on_chip
            mapping_label = config_dict['mapping']
            engine_list = config_dict['engine_list']
            layer_list = config_dict['layer_list']
            if mapping_label == SESLMapping.MAPPING_LABEL:
                self.mapping_list.append(SESLMapping(self.mappings_hw_configs[segment_index],
                                                     self.model_dag, layer_list,
                                                     first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                     last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                     engine_parallelization_strategy=ParallelizationStrategies.OFMS_W))
            elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                self.mapping_list.append(SegmentMappingRR(
                    self.mappings_hw_configs[segment_index], self.model_dag, layer_list, len(
                        engine_list),
                    first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                    last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip))
            elif mapping_label == SEMLMapping_LBL.MAPPING_LABEL:
                self.mapping_list.append(SEMLMapping_LBL(self.mappings_hw_configs[segment_index],
                                                         self.model_dag, layer_list,
                                                         first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                         last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip))
            segment_index += 1

    def get_label(self):
        return self.MAPPING_LABEL

    def calc_exec_time(self, print_desc = False):
        exec_time = 0
        desc_str = ' '
        for mapping in self.mapping_list:
            exec_time += mapping.calc_exec_time()
            desc_str += str(exec_time) + ' ' 

        if print_desc:
            print(self.MAPPING_LABEL ,'get_stages_exec_times', desc_str)

        return exec_time
    
    def get_segment_exec_times(self):
        exec_times = []
        for mapping in self.mapping_list:
            exec_times.append(mapping.calc_exec_time())
        
        return exec_times

    def get_num_engines(self):
        all_engines = 0
        for mapping in self.mapping_list:
            all_engines += (mapping.get_num_engines())
        
        return int(all_engines)

    def calc_throughput(self):
        max_exec_time = 0
        for mapping in self.mapping_list:
            max_exec_time = max(max_exec_time, mapping.calc_exec_time())

        return 1 / max_exec_time

    def calc_fms_buffer_sz(self, print_desc = False):
        fms_buffer_sz = 0
        if print_desc:
            print(self.MAPPING_LABEL)
        for _, segments_index_list in self.first_engine_segment_index_dict.items():
            current_segment_max_fms_buffer_sz = 0
            for segment_index in segments_index_list:
                current_segment_max_fms_buffer_sz = max(current_segment_max_fms_buffer_sz,
                                                        self.mapping_list[segment_index].calc_fms_buffer_sz(print_desc))

            fms_buffer_sz += current_segment_max_fms_buffer_sz
            if print_desc:
                print(fms_buffer_sz)

        return fms_buffer_sz

    def calc_weights_buffer_sz(self):
        weights_buffer_sz = 0
        for _, segments_index_list in self.first_engine_segment_index_dict.items():
            current_segment_max_weights_buffer_sz = 0
            for segment_index in segments_index_list:
                current_segment_max_weights_buffer_sz = max(current_segment_max_weights_buffer_sz,
                                                            self.mapping_list[segment_index].calc_weights_buffer_sz())

            weights_buffer_sz += current_segment_max_weights_buffer_sz

        return weights_buffer_sz

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        off_chip_weight_accesses = 0
        for mapping in self.mapping_list:
            off_chip_weight_accesses += mapping.calc_off_chip_weights_access()

        return off_chip_weight_accesses

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access(self, print_desc = False):
        off_chip_fms_accesses = 0
        desc_str = ''
        for mapping in self.mapping_list:
            segment_off_chip_access = mapping.calc_off_chip_fms_access()
            off_chip_fms_accesses += segment_off_chip_access
            desc_str += '{} '.format(segment_off_chip_access)

        if print_desc:
            print(self.MAPPING_LABEL, 'calc_off_chip_fms_access', desc_str)

        return off_chip_fms_accesses

    def get_engines(self):
        engines = []
        for mapping in self.mapping_list:
            engines.extend(mapping.engines)

        return engines

    def get_off_chip_tmp_channels_layers(self):
        are_tmp_channels_on_chip = []
        for mapping in self.mapping_list:
            are_tmp_channels_on_chip.extend(mapping.get_off_chip_tmp_channels_layers())
