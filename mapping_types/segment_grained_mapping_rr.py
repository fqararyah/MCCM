
from generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *
from simple_mapping import SimpleMapping
import constants

class SegmentMappingRR(SimpleMapping):
    EXTRA_MEMORY_OVERHEADS_W = 0#0.05
    EXTRA_MEMORY_OVERHEADS_FM = 0#0.05
    EXTRA_MEMORY_OVERHEADS_CONST = 0 * constants.KiB
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'SegmentedRR'

    def __init__(self, hw_config, model_dag, layers, num_engines, engines=None,
                 pre_balanced_engines=False,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False):
        super().__init__(hw_config, layers, engines, first_layer_ifms_are_on_chip,
                         last_layer_ofms_are_on_chip)
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.engine_layer_map = {}
        self.engine_layer_offset_map = {}
        self.model_dag = model_dag
        self.mapping_pes = hw_config.num_pes
        self.num_layers = len(layers)
        self.group_weights_on_chip = []
        self.layer_weights_on_chip = []
        self.layers_ifms_buffer_szs_main = {}
        self.layers_ifms_buffer_szs_extra = {}
        self.layers_ifms_buffer_rows_main = {}
        self.layers_ifms_buffer_rows_extra = {}
        self.layers_ofms_buffer_szs = {}
        self.layers_ofms_buffer_rows = {}
        self.inter_segment_buffer_on_chip = []
        self.segment_exec_times = []
        self.segment_off_chip_access_times = []
        self.segment_comp_times = []
        self.num_engines = num_engines
        self.inter_segment_buffer_sz = -1
        self.exec_time = -1
        self.num_layer_groups = int(
            math.ceil(self.num_layers / self.num_engines))
        self.calc_required_dw_and_conv_engines()
        if not pre_balanced_engines:
            self.initialize_engines()
        self.initialize_engine_layers()
        self.layers_to_produce_row_counts = [
            self.DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS] * self.num_layers
        self.calc_to_produce_row_counts()
        self.layers_exec_times = [-1] * len(layers)
        if not pre_balanced_engines:
            self.allocate_and_balance_pes_to_engines(hw_config.num_pes)

    def get_label(self):
        return self.MAPPING_LABEL

    def calc_required_dw_and_conv_engines(self):
        dw_layers_in_range = [0] * \
            int(math.ceil(self.num_layers / self.num_engines))
        conv_pw_layers_in_range = [
            0] * int(math.ceil(self.num_layers / self.num_engines))
        for i in range(0, len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                dw_layers_in_range[i // self.num_engines] += 1
            else:
                conv_pw_layers_in_range[i // self.num_engines] += 1

        self.num_conv_pw_engines = max(conv_pw_layers_in_range)
        self.num_dw_engines = max(dw_layers_in_range)
        self.num_engines_refined = self.num_conv_pw_engines + self.num_dw_engines

    def initialize_engines(self):
        if self.engines == None:
            self.engines = []
        for i in range(self.num_engines_refined):
            if i < self.num_conv_pw_engines:
                if constants.USE_BSAELINES_OWN_PARALLELIZTION_STRATEGIES:
                    self.engines.append(Engine(1, parallelization_strategy=ParallelizationStrategies.OFMS_IFM))
                else:
                    self.engines.append(Engine(1))
            else:
                self.engines.append(
                    Engine(1, parallelization_strategy=ParallelizationStrategies.IN_FILTER_H_W))

    def initialize_engine_layers(self):
        dw_layers_in_segment_so_far = 0
        conv_layers_in_segment_so_far = 0
        for i in range(self.num_layers):
            if i % self.num_engines == 0:
                dw_layers_in_segment_so_far = 0
                conv_layers_in_segment_so_far = 0

            layer_specs = self.model_dag[self.layers[i]]
            if utils.is_dw_layer(layer_specs):
                # assuming that first are pw and conv engines then dw engines
                engine_index = self.num_conv_pw_engines + dw_layers_in_segment_so_far
                dw_layers_in_segment_so_far += 1
            else:
                engine_index = conv_layers_in_segment_so_far
                conv_layers_in_segment_so_far += 1

            if engine_index not in self.engine_layer_map:
                self.engine_layer_map[engine_index] = []
                self.engine_layer_offset_map[engine_index] = []
            self.engine_layer_map[engine_index].append(self.layers[i])
            self.engine_layer_offset_map[engine_index].append(i)

    def calc_to_produce_row_counts(self):
        for layer_group_index in range(self.num_layer_groups):
            first_layer_offset = layer_group_index * self.num_engines
            last_conv_layer_offset = min(
                self.num_layers - 1, first_layer_offset + self.num_engines - 1)
            last_conv_layer_index = self.layers[last_conv_layer_offset]
            last_conv_layer_specs = self.model_dag[last_conv_layer_index]
            last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]
            for i in range(self.num_engines):
                current_layer_offset = first_layer_offset + i
                if current_layer_offset > last_conv_layer_offset:
                    break
                layer_index = self.layers[current_layer_offset]
                layer_specs = self.model_dag[layer_index]
                ofms_height = layer_specs['ofms_shape'][1]

                self.layers_to_produce_row_counts[current_layer_offset] = self.rows_to_produce_in_pipe_pass * \
                    ofms_height / last_conv_layer_ofms_height

    def get_pipe_num_passes(self):
        num_passes = []
        for layer_group_index in range(self.num_layer_groups):
            first_layer_offset = layer_group_index * self.num_engines
            last_conv_layer_offset = min(
                self.num_layers - 1, first_layer_offset + self.num_engines - 1)
            last_conv_layer_index = self.layers[last_conv_layer_offset]
            last_conv_layer_specs = self.model_dag[last_conv_layer_index]
            last_conv_layer_ofms_height = last_conv_layer_specs['ofms_shape'][1]
            num_passes.append(last_conv_layer_ofms_height /
                              self.rows_to_produce_in_pipe_pass)

        return num_passes

    def allocate_and_balance_pes_to_engines(self, num_pes):
        used_pes = 0
        engines_overall_workloads = [0] * self.num_engines_refined
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        estimated_exec_times = [0] * self.num_engines_refined

        # initialize the estimated exec times
        for engine_index, layer_offsets in self.engine_layer_offset_map.items():
            self.engines[engine_index].num_pes = 1
            for layer_offset in layer_offsets:
                engines_overall_workloads[engine_index] += op_counts[layer_offset]
                estimated_exec_times[engine_index] += op_counts[layer_offset] / \
                    self.engines[engine_index].num_pes

        # balance
        while used_pes < num_pes:
            max_latency = max(estimated_exec_times)
            max_latency_index = estimated_exec_times.index(max_latency)
            to_add_pes = self.engines[max_latency_index].num_pes
            if used_pes + to_add_pes < num_pes:
                self.engines[max_latency_index].num_pes += to_add_pes
                used_pes += to_add_pes
                estimated_exec_times[max_latency_index] = \
                    engines_overall_workloads[max_latency_index] / \
                    self.engines[max_latency_index].num_pes
            else:
                break

        # adjust pes on dims
        for i in range(self.num_engines_refined):
            self.engines[i].distribute_PEs_on_dims(self.model_dag, self.engine_layer_map[i],
                                                   [self.layers_to_produce_row_counts[j] for j in self.engine_layer_offset_map[i]])

    def calc_layers_exec_times(self):
        for engine_index, layer_indices in self.engine_layer_map.items():
            for i in range(len(layer_indices)):
                layer_index = layer_indices[i]
                layer_offset = self.engine_layer_offset_map[engine_index][i]
                layer_specs = self.model_dag[layer_index]
                self.layers_exec_times[layer_offset] = \
                    self.engines[engine_index].calc_layer_exec_time(
                        layer_specs, self.layers_to_produce_row_counts[layer_offset])

        return self.layers_exec_times     

    def calc_pipe_filling_times(self):
        pipe_filling_times = [0] * self.num_layer_groups
        for layer_group_index in range(self.num_layer_groups):
            first_layer_offset = layer_group_index * self.num_engines
            for i in range(1, self.num_engines):
                current_layer_offset = first_layer_offset + i
                if current_layer_offset >= self.num_layers:
                    break
                pipe_filling_times[layer_group_index] += max(
                    self.layers_exec_times[first_layer_offset: current_layer_offset])

        return pipe_filling_times

    def calc_compute_time(self):
        self.calc_layers_exec_times()
        pipe_filing_times = self.calc_pipe_filling_times()
        pipe_num_passes = self.get_pipe_num_passes()
        self.segment_comp_times = [0] * self.num_layer_groups
        for layer_group_index in range(self.num_layer_groups):
            first_layer_offset = layer_group_index * self.num_engines
            last_layer_offset = min(
                self.num_layers, first_layer_offset + self.num_engines)
            pipe_bottleneck = max(
                self.layers_exec_times[first_layer_offset: last_layer_offset])
            self.segment_comp_times[layer_group_index] = (pipe_filing_times[layer_group_index] +
                                                  pipe_bottleneck * pipe_num_passes[layer_group_index]) / self.hw_config.frequency

        return self.segment_comp_times

    def calc_segment_off_chip_access_time(self, segment_index, available_on_chip_memory):
        return (self.calc_segment_off_chip_weights_access(segment_index) +
                self.calc_segment_off_chip_fms_access(segment_index, available_on_chip_memory)) / self.hw_config.bw

    def calc_exec_time(self, print_desc = False):
        if print_desc:
            print(self.MAPPING_LABEL)
        on_chip_memory = self.hw_config.on_chip_memory
        weights_buffer_sz = self.calc_weights_buffer_sz()
        on_chip_memory -= weights_buffer_sz
        self.calc_compute_time()
        exec_time = 0
        for layer_group_index in range(self.num_layer_groups):
            self.segment_off_chip_access_times.append(self.calc_segment_off_chip_access_time(layer_group_index, on_chip_memory)
                                                                                              )
            if self.group_weights_on_chip[layer_group_index]:
                current_segment_exec_time = max(self.segment_off_chip_access_times[layer_group_index],
                                 self.segment_comp_times[layer_group_index])
            else:
                current_segment_exec_time = self.segment_off_chip_access_times[layer_group_index] + \
                    self.segment_comp_times[layer_group_index]
                    
            exec_time += current_segment_exec_time
            self.segment_exec_times.append(current_segment_exec_time)
            
        self.exec_time = exec_time

        return exec_time

    def calc_segments_exec_times(self):
        if len(self.segment_exec_times) == 0:
            self.calc_exec_time()
            
        return self.segment_exec_times
    
    def get_segment_comp_and_access_times(self):
        segments_comp_and_access_times = []
        if len(self.segment_exec_times) == 0:
            self.calc_exec_time()
            
        for layer_group_index in range(self.num_layer_groups):
            segments_comp_and_access_times.append((self.segment_comp_times[layer_group_index], self.segment_off_chip_access_times[layer_group_index]))
            
        return segments_comp_and_access_times
            
    def get_segment_exec_times(self):
        return self.calc_segments_exec_times()
    
    def get_num_engines(self):
        return self.num_engines
        
    def calc_throughput(self):
        if self.exec_time == -1:
            self.calc_exec_time()

        return 1 / self.exec_time

    def calc_fms_buffer_szs_intra_segment(self):
        on_chip_memory = self.hw_config.on_chip_memory
        groups_fms_buffer_sz = [0] * self.num_layer_groups
        for layer_group_index in range(self.num_layer_groups):
            first_layer_offset = layer_group_index * self.num_engines
            for i in range(self.num_engines):
                layer_offset = first_layer_offset + i
                if layer_offset >= self.num_layers:
                    break
                layer_index = self.layers[layer_offset]
                layer_specs = self.model_dag[layer_index]
                ifms_depth = utils.get_layer_ifms_shape(layer_specs)[0]
                [ofms_depth, _, ofms_width] = utils.get_layer_ofms_shape(
                    layer_specs)
                filter_dim = utils.get_filter_dim(layer_specs)
                strides = utils.get_strides(layer_specs)
                filter_dim_minus_strides = filter_dim - strides
                if filter_dim_minus_strides < 0:
                    filter_dim_minus_strides = 0
                ofms_buffer_sz = ofms_depth * ofms_width * \
                    self.layers_to_produce_row_counts[layer_offset]
                self.layers_ofms_buffer_szs[layer_index] = ofms_buffer_sz
                self.layers_ofms_buffer_rows[layer_index] = self.layers_to_produce_row_counts[layer_offset]
                ofms_buffer_sz = self.calc_actual_bram_cons(
                    ofms_buffer_sz, self.engines[i].get_parallelism_fms())
                if utils.get_layer_num_children(layer_specs) > 1:
                    ofms_buffer_sz += 2 * ofms_buffer_sz
                ifms_buffer_sz_main = ifms_depth * ofms_width * strides * self.layers_to_produce_row_counts[layer_offset] * strides
                ifms_buffer_sz_extra = ifms_depth * ofms_width * strides * filter_dim_minus_strides
                ifms_buffer_sz = ifms_buffer_sz_main + ifms_buffer_sz_extra
                self.layers_ifms_buffer_szs_main[layer_index] = ifms_buffer_sz_main
                self.layers_ifms_buffer_szs_extra[layer_index] = ifms_buffer_sz_extra
                self.layers_ifms_buffer_rows_main[layer_index] = self.layers_to_produce_row_counts[layer_offset] * strides
                self.layers_ifms_buffer_rows_extra[layer_index] = filter_dim_minus_strides

                ifms_buffer_sz = self.calc_actual_bram_cons(
                    ifms_buffer_sz, self.engines[i].get_parallelism_fms())

                groups_fms_buffer_sz[layer_group_index] += ifms_buffer_sz + \
                    ofms_buffer_sz

        return groups_fms_buffer_sz

    def calc_fms_buffer_sz_inter_segment(self):
        inter_segment_fm_szs = [0] * self.num_layer_groups
        for layer_group_index in range(1, self.num_layer_groups):
            first_layer_offset = layer_group_index * self.num_engines
            first_layer_specs = self.model_dag[self.layers[first_layer_offset]]
            inter_segment_fm_szs[layer_group_index] = utils.get_layer_ifms_size(
                first_layer_specs)

        return inter_segment_fm_szs

    def calc_fms_buffer_sz(self, print_desc = False):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = max(
            self.calc_fms_buffer_szs_intra_segment())
        weights_buffer_sz = self.calc_weights_buffer_sz()
        on_chip_memory -= (fms_buffer_sz + weights_buffer_sz)
        inter_segment_buffer_sz = 0

        already_filled_inter_segment_buffer_on_chip = len(self.inter_segment_buffer_on_chip) >= self.num_layer_groups
        if not already_filled_inter_segment_buffer_on_chip:
            self.inter_segment_buffer_on_chip.append(0)
        if on_chip_memory > 0:
            inter_segment_fms_szs = self.calc_fms_buffer_sz_inter_segment()
            for layer_group_index in range(1, self.num_layer_groups):
                inter_stage_fms_sz = inter_segment_fms_szs[layer_group_index]
                inter_segment_buffer_sz = max(
                            inter_segment_buffer_sz, inter_stage_fms_sz)
                if not already_filled_inter_segment_buffer_on_chip:
                    if inter_stage_fms_sz < on_chip_memory:
                        self.inter_segment_buffer_on_chip.append(1)
                    else:
                        self.inter_segment_buffer_on_chip.append(0)

        self.inter_segment_buffer_sz = self.calc_actual_bram_cons(
            inter_segment_buffer_sz, self.engines[0].get_parallelism_fms())
        fms_buffer_sz += self.inter_segment_buffer_sz

        return fms_buffer_sz

     # rm stands for requiren minimum
    def calc_weights_buffer_sz_in_layers_rm(self, layers):
        weights_buffer_sz = 0
        i = 0
        for layer_index in layers:
            layer_specs = self.model_dag[layer_index]
            rm_weights_sz = utils.get_layer_weights_size(layer_specs) * \
                self.engines[i].par_ofms / \
                utils.get_layer_weights_shape(layer_specs)[0]
            rm_weights_sz = self.calc_actual_bram_cons(
                rm_weights_sz, self.engines[i].get_ports_weights())
            weights_buffer_sz += rm_weights_sz
            i += 1

        return weights_buffer_sz

    def calc_total_weights_buffer_in_layers(self, layers):
        weights_buffer_sz = 0
        i = 0
        for layer_index in layers:
            layer_specs = self.model_dag[layer_index]
            weights_buffer_sz += self.calc_actual_bram_cons(utils.get_layer_weights_size(
                layer_specs), self.engines[i].get_ports_weights())
            i += 1

        return weights_buffer_sz

    def calc_weights_buffer_sz_in_layers(self, layers):
        on_chip_memory = self.hw_config.on_chip_memory
        fms_buffer_sz = max(
            self.calc_fms_buffer_szs_intra_segment())
        full_weights_sz = self.calc_total_weights_buffer_in_layers(layers)

        if full_weights_sz + fms_buffer_sz < on_chip_memory:
            return full_weights_sz, True

        return self.calc_weights_buffer_sz_in_layers_rm(layers), False

    # max of segments buffers
    def calc_weights_buffer_sz(self):
        weights_buffer_sz = 0
        max_eingines_weight_buffer_layers = self.get_max_eingines_weight_buffer_layers()
        for i in range(len(max_eingines_weight_buffer_layers)):
            max_eingines_weight_buffer_layer_index = max_eingines_weight_buffer_layers[i]
            if max_eingines_weight_buffer_layer_index == 0:
                continue
            layer_specs = self.model_dag[max_eingines_weight_buffer_layer_index]
            weights_buffer_sz += self.calc_actual_bram_cons(utils.get_layer_weights_size(
                layer_specs), self.engines[i].get_ports_weights())

        return weights_buffer_sz
    
    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_segment_off_chip_weights_access(self, segment_index):
        self.get_layer_weights_on_chip()
        pipe_num_passes = self.get_pipe_num_passes()
        first_layer_offset = segment_index * self.num_engines
        last_layer_offset = min(
            first_layer_offset + self.num_engines, self.num_layers)
        group_layers = self.layers[first_layer_offset: last_layer_offset]
        weights_buffer_sz = self.calc_total_weights_in_layers(group_layers)
        if self.group_weights_on_chip[segment_index]:
            return weights_buffer_sz
        else:
            return weights_buffer_sz * \
                pipe_num_passes[segment_index]

    def calc_off_chip_weights_access(self):
        self.get_layer_weights_on_chip()

        off_chip_access = [0] * self.num_layer_groups
        for layer_group_index in range(self.num_layer_groups):
            off_chip_access[layer_group_index] = self.calc_segment_off_chip_weights_access(
                layer_group_index)

        return sum(off_chip_access)

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip

    def calc_segment_off_chip_fms_access(self, segment_index, available_on_chip_memory):
        fms_buffer_szs = self.calc_fms_buffer_szs_intra_segment()

        first_layer_offset = segment_index * self.num_engines
        last_layer_offset = min(
            first_layer_offset + self.num_engines, self.num_layers)
        group_layers = self.layers[first_layer_offset: last_layer_offset]
        if available_on_chip_memory - fms_buffer_szs[segment_index] < 0:
            return self.calc_total_fms_in_layers(
                group_layers)
        else:
            return 0

    def calc_off_chip_fms_access(self, print_desc = False):
        inter_segment_fms_szs = self.calc_fms_buffer_sz_inter_segment()
        if self.inter_segment_buffer_sz == -1:
            self.calc_fms_buffer_sz()

        off_chip_fms_accesses = 0
        on_chip_memory = self.hw_config.on_chip_memory
        weights_buffer_sz = self.calc_weights_buffer_sz()
        intra_segments_fms_buffer_sz = max(
            self.calc_fms_buffer_szs_intra_segment())
        on_chip_memory -= (weights_buffer_sz + intra_segments_fms_buffer_sz)

        layer_specs = self.model_dag[self.layers[0]]
        first_layer_ifms_sz = utils.get_layer_ifms_size(layer_specs)
        layer_specs = self.model_dag[self.layers[-1]]
        last_layer_ifms_sz = utils.get_layer_ofms_size(layer_specs)
        if self.off_chip_fms_access_of_first_and_last_layers == -1:
            self.on_chip_buffer_sz_for_first_last_iofms(
                on_chip_memory, first_layer_ifms_sz, last_layer_ifms_sz)

        for layer_group_index in range(1, self.num_layer_groups):
            current_off_chip_access = self.calc_segment_off_chip_fms_access(
                layer_group_index, on_chip_memory)
            if current_off_chip_access != 0:
                off_chip_fms_accesses += current_off_chip_access
            else:
                stage_ifms_sz = inter_segment_fms_szs[layer_group_index]
                if stage_ifms_sz > self.inter_segment_buffer_sz:
                    off_chip_fms_accesses += 2 * stage_ifms_sz

        off_chip_fms_accesses += self.off_chip_fms_access_of_first_and_last_layers
        return off_chip_fms_accesses

    def get_layer_weights_on_chip(self):
        if len(self.group_weights_on_chip) == 0:
            weights_buffer_szs = [0] * self.num_layer_groups
            for layer_group_index in range(self.num_layer_groups):
                first_layer_offset = layer_group_index * self.num_engines
                last_layer_offset = min(
                    first_layer_offset + self.num_engines, self.num_layers)
                group_layers = self.layers[first_layer_offset: last_layer_offset]
                weights_buffer_szs[layer_group_index], group_weights_on_chip = \
                    self.calc_weights_buffer_sz_in_layers(group_layers)
                if group_weights_on_chip:
                    for i in range(0, self.num_engines):
                        self.layer_weights_on_chip.append(1)
                else:
                    for i in range(0, self.num_engines):
                        self.layer_weights_on_chip.append(0)
                self.group_weights_on_chip.append(group_weights_on_chip)

        return self.layer_weights_on_chip

    def get_inter_group_fms_on_chip(self):
        if len(self.inter_segment_buffer_on_chip) == 0:
            self.calc_fms_buffer_sz()
        return self.inter_segment_buffer_on_chip

    def get_engine_layer_mapping(self):
        return self.engine_layer_map

    def get_max_eingines_weight_buffer_layers(self):
        engines_weight_buffers = [0] * self.num_engines_refined
        engines_weight_buffer_indices = [0] * self.num_engines_refined
        layer_weights_on_chip = self.get_layer_weights_on_chip()
        for engine_index in range(self.num_engines_refined):
            for layer_index in self.engine_layer_map[engine_index]:
                if layer_weights_on_chip[self.layers.index(layer_index)]:
                    if engines_weight_buffers[engine_index] <= \
                            utils.get_layer_weights_size(self.model_dag[layer_index]):
                        engines_weight_buffers[engine_index] = utils.get_layer_weights_size(
                            self.model_dag[layer_index])
                        engines_weight_buffer_indices[engine_index] = layer_index

        return engines_weight_buffer_indices

    def get_max_eingines_fms_buffers_layers(self):
        engines_ifms_buffers_main = [0] * self.num_engines_refined
        engines_ifms_buffers_extra = [0] * self.num_engines_refined
        engines_ofms_buffers = [0] * self.num_engines_refined
        engines_ifms_buffer_indices_main = [0] * self.num_engines_refined
        engines_ifms_buffer_indices_extra = [0] * self.num_engines_refined
        engines_ifms_buffer_rows_main = [0] * self.num_engines_refined
        engines_ifms_buffer_rows_extra = [0] * self.num_engines_refined
        engines_ofms_buffer_indices = [0] * self.num_engines_refined
        engines_ofms_buffer_rows = [0] * self.num_engines_refined
        if len(self.layers_ifms_buffer_szs_main) == 0:
            self.calc_fms_buffer_szs_intra_segment()
        for engine_index in range(self.num_engines_refined):
            offset = 0
            for layer_index in self.engine_layer_map[engine_index]:
                if engines_ifms_buffers_main[engine_index] <= self.layers_ifms_buffer_szs_main[layer_index]:
                    engines_ifms_buffers_main[engine_index] = int(self.layers_ifms_buffer_szs_main[layer_index])
                    engines_ifms_buffer_rows_main[engine_index] = int(self.layers_ifms_buffer_rows_main[layer_index])
                    engines_ifms_buffer_indices_main[engine_index] = layer_index
                if engines_ifms_buffers_extra[engine_index] <= self.layers_ifms_buffer_szs_extra[layer_index]:
                    engines_ifms_buffers_extra[engine_index] = int(self.layers_ifms_buffer_szs_extra[layer_index])
                    engines_ifms_buffer_rows_extra[engine_index] = int(self.layers_ifms_buffer_rows_extra[layer_index])
                    engines_ifms_buffer_indices_extra[engine_index] = layer_index
                if engines_ofms_buffers[engine_index] <= self.layers_ofms_buffer_szs[layer_index]:
                    engines_ofms_buffers[engine_index] = int(self.layers_ofms_buffer_szs[layer_index])
                    engines_ofms_buffer_rows[engine_index] = int(self.layers_ofms_buffer_rows[layer_index])
                    engines_ofms_buffer_indices[engine_index] = layer_index
                offset += 1

        return engines_ifms_buffer_indices_main,engines_ifms_buffer_rows_main, \
            engines_ifms_buffer_indices_extra,engines_ifms_buffer_rows_extra, \
            engines_ofms_buffer_indices, engines_ofms_buffer_rows 

    def get_dict_representation(self):
        mapping_dict = {}
        num_layers = self.num_layers
        num_engines = self.get_num_engines()
        layers_str = '0-{}'.format(num_layers)
        engines_str = '0-{}'.format(num_engines)

        mapping_dict[layers_str] = engines_str

        return mapping_dict