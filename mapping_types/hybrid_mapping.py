
from .generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *
from .seml_mapping_lbl import *
from .seml_mapping_fused import *
from .sesl_mapping import *
from hw_config import *
import copy

class HybridMapping(GenericMapping):
    EXTRA_MEMORY_OVERHEADS_W = 0#0.05
    EXTRA_MEMORY_OVERHEADS_FM = 0#0.05
    EXTRA_MEMORY_OVERHEADS_CONST = 0 * constants.KiB
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    FIRST_PART_DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'Hybrid'

    def __init__(self, hw_config, model_dag, layers, spliting_point,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_part_rows_to_produce_in_pipe_pass=FIRST_PART_DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 fuse_in_the_second_part=False,
                 exec_v2=False):
        super().__init__(hw_config, layers, [])
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.first_part_rows_to_produce_in_pipe_pass = first_part_rows_to_produce_in_pipe_pass
        self.model_dag = model_dag
        self.all_pes = hw_config.num_pes
        self.num_layers = len(layers)
        self.spliting_point = spliting_point
        self.on_chip_memory = hw_config.on_chip_memory
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.fuse_in_the_second_part = fuse_in_the_second_part
        self.exec_v2 = exec_v2
        self.initialize_engines_and_mappings()

    def get_label(self):
        return self.MAPPING_LABEL

    def initialize_engines_and_mappings(self):
        self.initialize_segment_layers()
        self.initialize_balanced_segments_pes_and_buffers()
        self.initialize_mappings()

    def initialize_segment_layers(self):
        self.first_part_layers = self.layers[0: self.spliting_point]
        self.second_part_layers = self.layers[self.spliting_point: self.num_layers]

    def initialize_balanced_segments_pes_and_buffers(self):
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        overall_op_count = sum(op_counts)
        first_part_op_count = sum(utils.get_layers_op_counts_by_indices(
            self.model_dag, self.first_part_layers))
        first_part_pes = int(
            first_part_op_count * self.all_pes / overall_op_count)
        self.first_part_hw_config = copy.deepcopy(self.hw_config)
        self.first_part_hw_config.num_pes = first_part_pes
        second_part_pes = self.all_pes - first_part_pes
        self.second_part_hw_config = copy.deepcopy(self.hw_config)
        self.second_part_hw_config.num_pes = second_part_pes

        # heuristic
        weights_szs = utils.get_weights_sizes(self.model_dag)
        fms_szs = utils.get_fms_sizes(self.model_dag)
        ifms_szs = utils.get_ifms_sizes(self.model_dag)
        ofms_szs = utils.get_ofms_sizes(self.model_dag)
        # first paty contribution
        first_part_num_passes = utils.get_layer_ofms_shape(self.model_dag[self.first_part_layers[-1]])[1] / \
            self.first_part_rows_to_produce_in_pipe_pass
        first_part_szs = sum(
            weights_szs[0:self.spliting_point]) + sum(fms_szs[0:self.spliting_point]) * 2 / first_part_num_passes
        # second part contribution, in the secod part the FMs size is the dominant and weights size is negligable
        second_part_sz = max(max(ifms_szs[self.spliting_point: self.num_layers]),
                             max(ofms_szs[self.spliting_point: self.num_layers])) * 2
        comibed_sz = first_part_szs + second_part_sz
        first_part_on_chip_mem = int(
            self.on_chip_memory * first_part_szs / comibed_sz)
        self.first_part_hw_config.on_chip_memory = first_part_on_chip_mem
        second_part_on_chip_mem = self.on_chip_memory - first_part_on_chip_mem
        self.second_part_hw_config.on_chip_memory = second_part_on_chip_mem

    def initialize_mappings(self):
        self.first_part_mapping = SESLMapping(self.first_part_hw_config,
                                              self.model_dag, self.first_part_layers,
                                              last_layer_ofms_are_on_chip=True,
                                              engine_parallelization_strategy=ParallelizationStrategies.OFMS_W)
        if self.fuse_in_the_second_part:
            self.second_part_mapping = SEMLMapping_FUSED(self.second_part_hw_config,
                                                         self.model_dag, self.second_part_layers,
                                                         self.rows_to_produce_in_pipe_pass, first_layer_ifms_are_on_chip=True)
        else:
            self.second_part_mapping = SEMLMapping_LBL(self.second_part_hw_config,
                                                       self.model_dag, self.second_part_layers, first_layer_ifms_are_on_chip=True,
                                                       exec_v2 = self.exec_v2)

    def calc_exec_time(self, print_desc=False):
        if print_desc:
            print(self.first_part_mapping.calc_exec_time(),
                  self.second_part_mapping.calc_exec_time())
        return self.first_part_mapping.calc_exec_time() + self.second_part_mapping.calc_exec_time()

    def get_segment_exec_times(self):
        return [self.first_part_mapping.calc_exec_time(), self.second_part_mapping.calc_exec_time()]

    def get_segment_buffer_sizes(self):
        segment_buffer_sizes = [self.first_part_mapping.calc_fms_buffer_sz() + self.first_part_mapping.calc_weights_buffer_sz(),
                                self.second_part_mapping.calc_fms_buffer_sz() + self.second_part_mapping.calc_weights_buffer_sz()]

        return segment_buffer_sizes

    def get_segment_fms_buffer_sizes_intra(self):
        segment_buffer_sizes = [self.first_part_mapping.calc_fms_buffer_sz_intra(),
                                self.second_part_mapping.calc_fms_buffer_sz_intra]

        return segment_buffer_sizes

    def get_segment_fms_buffer_sizes(self):
        segment_buffer_sizes = [self.first_part_mapping.calc_fms_buffer_sz(),
                                self.second_part_mapping.calc_fms_buffer_sz()]

        return segment_buffer_sizes

    def get_segment_weights_buffer_sizes(self):
        segment_buffer_sizes = [self.first_part_mapping.calc_weights_buffer_sz(),
                                self.second_part_mapping.calc_weights_buffer_sz()]

        return segment_buffer_sizes

    def calc_throughput(self):
        return 1 / max(self.first_part_mapping.calc_exec_time(), self.second_part_mapping.calc_exec_time())

    def get_first_part_on_chip_fms_buffer_sz(self, print_desc=False):
        return self.first_part_mapping.calc_fms_buffer_sz(print_desc)

    def get_second_part_on_chip_fms_buffer_sz(self, print_desc=False):
        return self.second_part_mapping.calc_fms_buffer_sz(print_desc)

    def calc_fms_buffer_sz(self, print_desc=False):
        if print_desc:
            print(self.MAPPING_LABEL, 'calc_fms_buffer_sz')
        first_part_fms_buffer_sz = self.get_first_part_on_chip_fms_buffer_sz(
            print_desc)
        second_part_fms_buffer_sz = self.get_second_part_on_chip_fms_buffer_sz(
            print_desc)

        if print_desc:
            print(first_part_fms_buffer_sz, second_part_fms_buffer_sz)

        return first_part_fms_buffer_sz + second_part_fms_buffer_sz

    def calc_weights_buffer_sz(self):
        first_part_weights_buffer_sz = self.first_part_mapping.calc_weights_buffer_sz()
        second_part_weights_buffer_sz = self.second_part_mapping.calc_weights_buffer_sz()

        return first_part_weights_buffer_sz + second_part_weights_buffer_sz

    def get_num_engines(self):
        return int(self.first_part_mapping.get_num_engines() + self.second_part_mapping.get_num_engines() - 1)

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    def calc_off_chip_weights_access(self):
        first_part_weights_off_chip_access = self.first_part_mapping.calc_off_chip_weights_access()
        second_part_weights_off_chip_access = self.second_part_mapping.calc_off_chip_weights_access()

        return first_part_weights_off_chip_access + second_part_weights_off_chip_access

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    def calc_off_chip_fms_access(self, print_desc=False):
        first_part_off_chip_fms_access = self.first_part_mapping.calc_off_chip_fms_access()
        second_part_off_chip_fms_access = self.second_part_mapping.calc_off_chip_fms_access()
        return first_part_off_chip_fms_access + second_part_off_chip_fms_access

    def get_engines(self):
        return self.first_part_mapping.engines + self.second_part_mapping.engines

    def get_off_chip_tmp_channels_layers(self):
        return self.second_part_mapping.get_off_chip_tmp_channels_layers()
