
from .generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *
from .seml_mapping_lbl import *
from .seml_mapping_fused import *
from hw_config import *
import mapping_utils
import copy
from basic_mapping import BasicMapping


class SegmentMapping(BasicMapping):
    EXTRA_MEMORY_OVERHEADS_W = 0
    EXTRA_MEMORY_OVERHEADS_FM = 0
    EXTRA_MEMORY_OVERHEADS_CONST = 0 * constants.KiB
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'Segmented'

    def __init__(self, hw_config, model_dag, layers, num_engines, engines=None,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 fused_engines=False,
                 exec_v2= False):
        super().__init__(hw_config, model_dag, layers, engines)
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.engine_layer_map = {}
        self.engine_layer_offset_map = {}
        self.mapping_pes = hw_config.num_pes
        self.exec_time = -1
        self.engine_group_pes = []
        self.num_layers = len(layers)
        self.group_weights_on_chip = []
        self.group_fms_on_chip = []
        self.num_engines = num_engines
        self.engines_mappings = []
        self.segments_hw_configs = []
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.fused_engines = fused_engines
        self.exec_v2 = exec_v2
        self.initialize_engines_and_mappings()

    def get_label(self):
        return self.MAPPING_LABEL

    def initialize_engines_and_mappings(self):
        self.initialize_engine_layers_kmeans()
        self.initialize_balanced_segments_pes_and_buffers()
        self.initialize_mappings()

    def initialize_engine_layers_kmeans(self):
        cluster_layer_count_map = mapping_general_utils.cluster_layers_using_kmeans(
            self.model_dag, self.num_engines)
        num_layers_so_far = 0
        for i in range(self.num_engines):
            current_engine_num_layers = cluster_layer_count_map[i]
            last_layer_offset = num_layers_so_far + current_engine_num_layers
            self.engine_layer_map[i] = self.layers[num_layers_so_far: last_layer_offset]
            self.engine_layer_offset_map[i] = [
                *range(num_layers_so_far, last_layer_offset)]
            
            num_layers_so_far += current_engine_num_layers
            
    def initialize_engine_layers_uniform(self):
        engine_least_num_of_layers = int(
            math.floor(self.num_layers / self.num_engines))
        extra_layers = self.num_layers - engine_least_num_of_layers * self.num_engines
        first_layer_offset = 0
        for i in range(self.num_engines):
            current_engine_num_layers = engine_least_num_of_layers + \
                (1 if extra_layers > 0 else 0)
            last_layer_offset = min(
                first_layer_offset + current_engine_num_layers, self.num_layers)

            self.engine_layer_map[i] = self.layers[first_layer_offset: last_layer_offset]
            self.engine_layer_offset_map[i] = [
                *range(first_layer_offset, last_layer_offset)]
            first_layer_offset += current_engine_num_layers
            extra_layers -= 1

    def initialize_balanced_segments_pes_and_buffers(self):
        op_counts = utils.get_layers_op_counts_by_indices(self.model_dag, self.layers)
        ifms_sizes = utils.get_ifms_sizes(self.model_dag)
        weights_sizes = utils.get_weights_sizes(self.model_dag)
        max_fms_sizes_per_engine = []
        weight_sizes_per_engine = []
        overall_op_count = sum(op_counts)
        for i in range(self.num_engines):
            first_layer_offset = self.engine_layer_offset_map[i][0]
            last_layer_offset = min(
                self.engine_layer_offset_map[i][-1], self.num_layers - 1)
            current_pes = (max(len(self.engine_layer_offset_map[i]),
                int(self.mapping_pes * sum(op_counts[first_layer_offset: last_layer_offset + 1]) / overall_op_count)))
            current_hw_config = copy.deepcopy(self.hw_config)
            current_hw_config.num_pes = current_pes
            self.segments_hw_configs.append(current_hw_config)
            if self.fused_engines:
                weight_sizes_per_engine.append(
                    sum(weights_sizes[first_layer_offset: last_layer_offset + 1]))
                max_fms_sizes_per_engine.append(max(ifms_sizes[first_layer_offset: last_layer_offset + 1]) /
                                                self.get_pipe_num_passes())
            else:
                weight_sizes_per_engine.append(
                    max(weights_sizes[first_layer_offset: last_layer_offset + 1]))
                max_fms_sizes_per_engine.append(
                    max(ifms_sizes[first_layer_offset: last_layer_offset + 1]))

        combined_fms_buffer_sz = sum(max_fms_sizes_per_engine)
        combined_weights_buffer_sz = sum(weight_sizes_per_engine)
        for i in range(self.num_engines):
            self.segments_hw_configs[i].on_chip_memory = self.hw_config.on_chip_memory * \
                ((max_fms_sizes_per_engine[i] + weight_sizes_per_engine[i]) /
                 (combined_fms_buffer_sz + combined_weights_buffer_sz))

    def initialize_mappings(self):
        for i in range(self.num_engines):
            # assume fused mappings inside,
            # otherwise the latency will grow with numbre of engines which is impractical, plus most of the
            # resources are going to be idle most of the time
            first_layer_ifms_are_on_chip = True
            last_layer_ofms_are_on_chip = True
            if i == 0:
                first_layer_ifms_are_on_chip = False
            if i == self.num_engines - 1:
                last_layer_ofms_are_on_chip = False
            if self.fused_engines:
                self.engines_mappings.append(SEMLMapping_FUSED(self.segments_hw_configs[i],
                                                               self.model_dag, self.engine_layer_map[i],
                                                               first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                               last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip))
            else:
                self.engines_mappings.append(SEMLMapping_LBL(self.segments_hw_configs[i],
                                                             self.model_dag, self.engine_layer_map[i],
                                                             first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                             last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip, exec_v2 = self.exec_v2))

    def get_engine_layer_mapping(self):
        return self.engine_layer_map
    
    def get_engines(self):
        engines = []
        for layer_group_index in range(self.num_engines):
            engines.extend(self.engines_mappings[layer_group_index].get_engines())

        return engines
    
    def get_pipe_num_passes(self):
        last_conv_layer_index = self.layers[-1]
        last_conv_layer_specs = self.model_dag[last_conv_layer_index]
        last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]

        return last_conv_layer_ofms_height / self.rows_to_produce_in_pipe_pass

    def get_stages_exec_times(self, print_desc = False):
        stages_exec_times = [0] * self.num_engines
        desc_str = ' '
        for i in range(self.num_engines):
            stages_exec_times[i] = self.engines_mappings[i].calc_exec_time()
            desc_str += '{} '.format(stages_exec_times[i]) 

        if print_desc:
            print(self.MAPPING_LABEL ,'get_stages_exec_times', desc_str)

        return stages_exec_times        
    
    def get_stages_rm_rows(self):
        stages_rms = [0] * self.num_engines
        for i in range(self.num_engines):
            first_layer_index = self.engine_layer_map[i][0]
            last_layer_index = self.engine_layer_map[i][-1]
            stages_rms[i] = self.engines_mappings[i].rows_to_produce_in_pipe_pass * \
                utils.get_layer_ofms_shape(self.model_dag[first_layer_index])[1] / \
                utils.get_layer_ofms_shape(self.model_dag[last_layer_index])[1]

        return stages_rms

    def calc_stages_exec_times_in_a_pass(self):
        stages_exec_times_in_a_pass = [0] * self.num_engines
        stages_exec_times = self.get_stages_exec_times()
        stages_rm_rows = self.get_stages_rm_rows()

        for i in range(self.num_engines):
            if i == self.num_engines - 1:
                rows_to_produce_in_a_pass = self.engines_mappings[-1].rows_to_produce_in_pipe_pass
            else:
                rows_to_produce_in_a_pass = stages_rm_rows[i+1]

            next_stahe_rm_rows_ratio = rows_to_produce_in_a_pass / \
                utils.get_layer_ifms_shape(
                    self.model_dag[self.engine_layer_map[i][-1]])[1]

            stages_exec_times_in_a_pass[i] = stages_exec_times[i] * \
                next_stahe_rm_rows_ratio

        return stages_exec_times_in_a_pass

    def calc_pipe_filling_time(self, stages_exec_times_in_a_pass):
        pipe_filling_time = 0
        for i in range(1, self.num_engines):
            pipe_filling_time += max(stages_exec_times_in_a_pass[0: i])

        return pipe_filling_time

    # ff stands for fully fused
    def calc_compute_time_ff(self):
        stages_exec_times_in_a_pass = self.calc_stages_exec_times_in_a_pass()
        pipe_filing_time = self.calc_pipe_filling_time(
            stages_exec_times_in_a_pass)
        pipe_bottleneck = max(stages_exec_times_in_a_pass)

        return pipe_filing_time + pipe_bottleneck * self.get_pipe_num_passes()

    # nf stands for not fused
    def calc_exec_time_nf(self, print_desc = False):
        return sum(self.get_stages_exec_times(print_desc))

    def calc_exec_time(self, print_desc = False):
        if self.fused_engines:
            self.exec_time = self.calc_compute_time_ff()
        else:
            self.exec_time = self.calc_exec_time_nf(print_desc)

        return self.exec_time
    
    def get_segment_comp_and_access_times(self):
        stages_exec_times = [0] * self.num_engines
        for i in range(self.num_engines):
            stages_exec_times[i] = (self.engines_mappings[i].calc_compute_time_only(), self.engines_mappings[i].calc_off_chip_access_time_only())

        return stages_exec_times       
    
    def get_segment_exec_times(self):
        return self.get_stages_exec_times(False)
    
    def get_num_engines(self):
        if utils.has_dw_layers(self.model_dag, self.layers[0], len(self.layers)):
            return self.num_engines / 2

        return self.num_engines

    def calc_throughput(self):
        return 1 / max(self.get_stages_exec_times())

    def calc_fms_buffer_sz(self, print_desc = False):
        #avaialble_on_chip_memory = self.hw_config.on_chip_memory
        groups_fms_buffer_sz = [0] * self.num_engines
        for layer_group_index in range(self.num_engines):
            current_mapping = self.engines_mappings[layer_group_index]
            group_on_chip_fms_buffer_sz = current_mapping.calc_fms_buffer_sz()
            #if group_on_chip_fms_buffer_sz < avaialble_on_chip_memory:
            #avaialble_on_chip_memory -= group_on_chip_fms_buffer_sz
            groups_fms_buffer_sz[layer_group_index] = group_on_chip_fms_buffer_sz

        return sum(groups_fms_buffer_sz)

    def calc_weights_buffer_sz(self):
        weights_buffer_szs = [0] * self.num_engines

        for layer_group_index in range(self.num_engines):
            weights_buffer_szs[layer_group_index] = self.engines_mappings[layer_group_index].calc_weights_buffer_sz(
            )

        return sum(weights_buffer_szs)
    
    def get_segment_buffer_sizes(self):
        segment_buffer_sizes = []
        for layer_group_index in range(self.num_engines):
            segment_buffer_sizes.append(self.engines_mappings[layer_group_index].calc_weights_buffer_sz(
            ) + self.engines_mappings[layer_group_index].calc_fms_buffer_sz())

        return segment_buffer_sizes

    def get_segment_fms_buffer_sizes_intra(self):
        segment_buffer_sizes = []
        for layer_group_index in range(self.num_engines):
            segment_buffer_sizes.append(self.engines_mappings[layer_group_index].calc_fms_buffer_sz_intra())

        return segment_buffer_sizes
    
    def get_segment_fms_buffer_sizes(self):
        segment_buffer_sizes = []
        for layer_group_index in range(self.num_engines):
            segment_buffer_sizes.append(self.engines_mappings[layer_group_index].calc_fms_buffer_sz())

        return segment_buffer_sizes
    
    def get_segment_weights_buffer_sizes(self):
        segment_buffer_sizes = []
        for layer_group_index in range(self.num_engines):
            segment_buffer_sizes.append(self.engines_mappings[layer_group_index].calc_weights_buffer_sz())

        return segment_buffer_sizes
    
    def get_off_chip_tmp_channels_layers(self):
        segments_tmp_channels_layers = []
        for layer_group_index in range(self.num_engines):
            segments_tmp_channels_layers.extend(self.engines_mappings[layer_group_index].get_off_chip_tmp_channels_layers())
        
        return segments_tmp_channels_layers
        
    def get_inter_group_fms_on_chip(self):
        inter_segments_channels_buffer_on_chip = []
        for layer_group_index in range(self.num_engines):
            self.engines_mappings[layer_group_index].calc_main_fms_buffer_sz() # to make sure first_layer_on_chip_buffer
            inter_segments_channels_buffer_on_chip.append(self.engines_mappings[layer_group_index].first_layer_on_chip_buffer != 0)
        
        return inter_segments_channels_buffer_on_chip
    
    def get_max_eingines_fms_buffers_layers(self):
        segment_fms_on_chip_size = []
        for layer_group_index in range(self.num_engines):
            segment_fms_on_chip_size.append(self.engines_mappings[layer_group_index].max_on_chip_fms_layers)
        
        return segment_fms_on_chip_size
        
    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        off_chip_access = [0] * self.num_engines
        for layer_group_index in range(self.num_engines):
            off_chip_access[layer_group_index] = self.engines_mappings[layer_group_index].calc_off_chip_weights_access()

        return sum(off_chip_access)
    
    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access(self, print_desc = False):
        off_chip_fms_access = 0
        desc_str = ''
        for layer_group_index in range(self.num_engines):
            segment_off_chip_access = self.engines_mappings[layer_group_index].calc_off_chip_fms_access()
            off_chip_fms_access += segment_off_chip_access
            desc_str += '{} '.format(segment_off_chip_access)
        
        if print_desc:
            print(self.MAPPING_LABEL, 'calc_off_chip_fms_access', desc_str)

        return off_chip_fms_access
    
    
