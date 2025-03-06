
from generic_mapping import GenericMapping
import __init__
import utils
from engines.engine import *
from seml_mapping_lbl import *
from seml_mapping_fused import *
from sesl_mapping import *
from segment_grained_mapping_rr import *
from hw_config import *
import copy
from preformance_record import Metrics
from mapping_utils import custom_mapping_utils


class CustomMapping(GenericMapping):
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'Custom'

    def __init__(self, hw_config, model_dag, layers, mappings_segments_config_list,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 apply_fusion=False,
                 timing_metric=Metrics.THROUGHPUT):

        super().__init__(hw_config, layers, [])
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.model_dag = model_dag
        self.num_layers = len(layers)
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.apply_fusion = apply_fusion
        self.mappings_segments_config_list = mappings_segments_config_list
        self.timing_metric = timing_metric
        self.num_segments = len(mappings_segments_config_list)
        self.mapping_list = []
        self.initialize_mappings()


    def balance_pes(self):
        overall_op_count = sum(utils.get_layers_op_counts(self.model_dag))
        segment_op_counts_list = []
        for _, layer_set in self.mappings_layers_dict.items():
            self.mappings_hw_configs.append(self.hw_config.copy_hw_config())

            current_op_count = sum(
                utils.get_layers_op_counts_by_indices(self.model_dag, layer_set))
            segment_op_counts_list.append(current_op_count)

            if self.timing_metric == Metrics.THROUGHPUT:
                current_pes = max(1, (current_op_count *
                                      self.hw_config.num_pes) // overall_op_count)
                self.mappings_hw_configs[-1].num_pes = current_pes

        if self.timing_metric == Metrics.LATENCY:
            engine_pes_list = custom_mapping_utils.proportional_allocation(
                self.hw_config.num_pes, segment_op_counts_list)
            for i in range(len(self.mappings_hw_configs)):
                self.mappings_hw_configs[i].num_pes = engine_pes_list[i]

    def __str__(self):
        return str(self.get_dict_representation())
    
    def initialize_tmp_mappings(self):
        tmp_mappings = [None] * self.num_segments
        adjust_pes = True
        segments_with_increased_pes = {}
        for i in range(0, 2):
            mappings_exec_times = []
            mapping_indices = []
            segment_index = 0
            for config_dict in self.mappings_segments_config_list:
                first_layer_ifms_are_on_chip = segment_index > 0 or self.first_layer_ifms_are_on_chip
                last_layer_ofms_are_on_chip = segment_index < self.num_segments - \
                    1 or self.last_layer_ofms_are_on_chip
                mapping_label = config_dict['mapping']
                engine_list = config_dict['engine_list']
                layer_list = config_dict['layer_list']
                tmp_mapping = None
                if i == 0 or segment_index in segments_with_increased_pes:
                    if segment_index >= len(self.mappings_hw_configs):
                        print(segment_index, len(self.mappings_hw_configs))
                        print(self.mappings_segments_config_list)
                        print(self.mappings_layers_dict)
                    if mapping_label == SESLMapping.MAPPING_LABEL:
                        tmp_mapping = SESLMapping(self.mappings_hw_configs[segment_index],
                                                  self.model_dag, layer_list,
                                                  first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                  last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                  engine_parallelization_strategy=ParallelizationStrategies.OFMS_W)
                    elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                        tmp_mapping = SegmentMappingRR(
                            self.mappings_hw_configs[segment_index], self.model_dag, layer_list, len(
                                engine_list),
                            first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                            last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip)
                    elif mapping_label == SEMLMapping_LBL.MAPPING_LABEL:
                        tmp_mapping = SEMLMapping_LBL(self.mappings_hw_configs[segment_index],
                                                      self.model_dag, layer_list,
                                                      first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                      last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                      pw_conv_parallelization_strategy=ParallelizationStrategies.CUSTOM)
                    tmp_mappings[segment_index] = tmp_mapping

                mappings_exec_times.append(
                    tmp_mappings[segment_index].calc_exec_time())
                mapping_indices.append(segment_index)
                segment_index += 1

            if adjust_pes and i == 0:
                zipped_lists = zip(mappings_exec_times, mapping_indices)
                zipped_lists = sorted(zipped_lists, reverse=True)
                _, sorted_mapping_indices = zip(*zipped_lists)
                unused_pes = 0
                segment_index = 0
                segment_used_pes_list = []
                segments_used_pe_sums = []
                for mapping in tmp_mappings:
                    segment_pes = 0
                    segment_used_pes = 0
                    segment_used_pes_list.append([])
                    for eng in mapping.get_engines():
                        segment_pes += eng.num_pes
                        segment_used_pes += eng.get_parallelism()
                        segment_used_pes_list[-1].append(eng.get_parallelism())
                    segments_used_pe_sums.append(segment_used_pes)
                    self.mappings_hw_configs[segment_index].num_pes = segment_used_pes
                    unused_pes += segment_pes - segment_used_pes
                    segment_index += 1
                if self.timing_metric == Metrics.THROUGHPUT:
                    for segment_index in sorted_mapping_indices:
                        segment_used_pe_sum = segments_used_pe_sums[segment_index]
                        if unused_pes >= segment_used_pe_sum:
                            self.mappings_hw_configs[segment_index].num_pes += segment_used_pe_sum
                            segments_with_increased_pes[segment_index] = 1
                            unused_pes -= segment_used_pe_sum
                        else:
                            num_engines = len(
                                segment_used_pes_list[segment_index])
                            segment_median_pes = segment_used_pes_list[segment_index][num_engines//2]
                            segment_min_pes = min(
                                segment_used_pes_list[segment_index])
                            if unused_pes >= segment_median_pes:
                                self.mappings_hw_configs[segment_index].num_pes += segment_median_pes
                                segments_with_increased_pes[segment_index] = 1
                                unused_pes -= segment_median_pes
                            elif unused_pes >= segment_min_pes:
                                self.mappings_hw_configs[segment_index].num_pes += segment_min_pes
                                segments_with_increased_pes[segment_index] = 1
                                unused_pes -= segment_min_pes
                else:
                    for segment_index in sorted_mapping_indices:
                        segment_used_pe_sum = segments_used_pe_sums[segment_index]
                        if unused_pes >= segment_used_pe_sum:
                            self.mappings_hw_configs[segment_index].num_pes += segment_used_pe_sum
                            segments_with_increased_pes[segment_index] = 1
                            unused_pes -= segment_used_pe_sum

                    for segment_index in sorted_mapping_indices:
                        num_engines = len(segment_used_pes_list[segment_index])
                        segment_median_pes = segment_used_pes_list[segment_index][num_engines//2]
                        if unused_pes >= segment_median_pes:
                            self.mappings_hw_configs[segment_index].num_pes += segment_median_pes
                            segments_with_increased_pes[segment_index] = 1
                            unused_pes -= segment_median_pes

                    for segment_index in sorted_mapping_indices:
                        segment_min_pes = min(
                            segment_used_pes_list[segment_index])
                        if unused_pes >= segment_min_pes:
                            self.mappings_hw_configs[segment_index].num_pes += segment_min_pes
                            segments_with_increased_pes[segment_index] = 1
                            unused_pes -= segment_min_pes

        # print('>', mappings_exec_times)

        # print(i, mappings_exec_times)
        # segment_index = 0
        # for mapping in tmp_mappings:
        #     print(self.mappings_hw_configs[segment_index].num_pes)
        #     segment_index += 1
        #     for eng in mapping.get_engines():
        #         print(eng.get_parallelism_dims())
        #     print('*****************')

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
            if first_engine_id not in self.first_engine_segment_index_dict:
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
            last_layer_ofms_are_on_chip = segment_index < self.num_segments - \
                1 or self.last_layer_ofms_are_on_chip
            mapping_label = config_dict['mapping']
            engine_list = config_dict['engine_list']
            layer_list = config_dict['layer_list']
            if mapping_label == SESLMapping.MAPPING_LABEL:
                self.mapping_list.append(SESLMapping(self.mappings_hw_configs[segment_index],
                                                     self.model_dag, layer_list,
                                                     engines=tmp_mappings[segment_index].engines,
                                                     pre_balanced_engines=True,
                                                     first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                     last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                     engine_parallelization_strategy=ParallelizationStrategies.CUSTOM))
            elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                self.mapping_list.append(SegmentMappingRR(
                    self.mappings_hw_configs[segment_index], self.model_dag, layer_list, len(
                        engine_list),
                    engines=tmp_mappings[segment_index].engines,
                    pre_balanced_engines=True,
                    first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                    last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip))
            elif mapping_label == SEMLMapping_LBL.MAPPING_LABEL:
                self.mapping_list.append(SEMLMapping_LBL(self.mappings_hw_configs[segment_index],
                                                         self.model_dag, layer_list,
                                                         engines=tmp_mappings[segment_index].engines,
                                                         pre_balanced_engines=True,
                                                         first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                         last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                         pw_conv_parallelization_strategy=ParallelizationStrategies.CUSTOM))
            segment_index += 1

    def get_label(self):
        return self.MAPPING_LABEL

    def calc_exec_time(self, print_desc=False):
        exec_time = 0
        desc_str = ' '
        for mapping in self.mapping_list:
            exec_time += mapping.calc_exec_time(print_desc)
            desc_str += str(exec_time) + ' '

        if print_desc:
            print(self.MAPPING_LABEL, 'get_stages_exec_times', desc_str)

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

    def calc_fms_buffer_sz(self, print_desc=False):
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
    def calc_off_chip_fms_access(self, print_desc=False):
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
            are_tmp_channels_on_chip.extend(
                mapping.get_off_chip_tmp_channels_layers())

    def get_dict_representation(self):
        mapping_dict = {}
        starting_engine = 0
        starting_layer = 0
        for mapping in self.mapping_list:
            num_engines = mapping.get_num_engines()
            num_layers = mapping.get_num_layers()
            layers_str = str(starting_layer)
            engines_str = str(starting_engine)
            if num_layers > 1:
                layers_str += '-{}'.format(starting_layer + num_layers)
            if num_engines > 1:
                engines_str += '-{}'.format(starting_engine + num_engines)

            starting_engine += num_engines
            starting_layer += num_layers

            mapping_dict[layers_str] = engines_str

        return mapping_dict

    def calc_energy(self):
        energy_cons = 0
        #energy_cons_breakdown = {}
        for mapping in self.mapping_list:
            #cons, cons_breakdown = mapping.calc_energy()
            cons = mapping.calc_energy()
            energy_cons += cons
            # for component, component_energy in cons_breakdown.items():
            #     if component not in energy_cons_breakdown:
            #         energy_cons_breakdown[component] = 0
            #     energy_cons_breakdown[component] = component_energy

        return energy_cons#, energy_cons_breakdown