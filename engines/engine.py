import statistics
import __init__
import utils
import math
import mapping_utils.mapping_general_utils as mapping_general_utils
from generic_engine import *


class Engine(GenericEngine):

    MIN_OFMS_PARALLEL = 8

    def __init__(self, num_pes, par_ofms=1, par_ifms=1, par_width=1, par_height=1, par_in_filter=1,
                 parallelization_strategy=ParallelizationStrategies.OFMS_H_W):
        super().__init__(num_pes, par_ofms, par_ifms, par_height, par_width,
                         par_in_filter, parallelization_strategy)

    def scale_dims_ofms_h_w(self, min_layer_filters, min_layer_height, min_layer_width,
                            par_ofms, par_h, par_w,
                            ofms_saturated, h_saturated, w_saturated):
        if (par_ofms == min(par_ofms, par_h, par_w) or (w_saturated and h_saturated)):
            par_ofms *= 2
        elif (par_w == min(par_ofms, par_h, par_w) or (h_saturated)):
            par_w *= 2
        elif not h_saturated:
            par_h *= 2
        ofms_saturated = par_ofms >= min_layer_filters
        h_saturated = par_h >= min_layer_height
        w_saturated = par_w >= min_layer_width

        return par_ofms, par_h, par_w, ofms_saturated, h_saturated, w_saturated

    def reset_parallelism(self):
        self.par_ofms = 1
        self.par_height = 1
        self.par_width = 1
        self.par_ifms = 1

    def scale_dims_h_w(self, min_layer_height, min_layer_width,
                       par_h, par_w, h_saturated, w_saturated):

        if (par_w == min(par_h, par_w) or (h_saturated)) or (min_layer_height % (par_h * 2) != 0 and not w_saturated):
            par_w *= 2
        elif not h_saturated:
            par_h *= 2
        h_saturated = par_h >= min_layer_height
        w_saturated = par_w >= min_layer_width

        return par_h, par_w, h_saturated, w_saturated

    def allocate_PEs_OFMS_H_W(self, min_layer_filters, min_layer_height, min_layer_width):
        par_ofms = min(self.MIN_OFMS_PARALLEL, self.num_pes)
        assert self.num_pes != 0
        par_h = 1
        par_w = 1
        ofms_saturated = par_ofms >= min_layer_filters
        h_saturated = par_h >= min_layer_height
        w_saturated = par_w >= min_layer_width

        while par_ofms * par_h * par_w <= self.num_pes / 2 and not (h_saturated and w_saturated and ofms_saturated):
            par_ofms, par_h, par_w, ofms_saturated, h_saturated, w_saturated = \
                self.scale_dims_ofms_h_w(min_layer_filters, min_layer_height, min_layer_width,
                                         par_ofms, par_h, par_w,
                                         ofms_saturated, h_saturated, w_saturated)

        self.par_ofms = par_ofms
        self.par_height = par_h
        self.par_width = par_w

    def allocate_PEs_IN_FILTER_H_W(self, min_layer_height, min_layer_width, max_filter_dim):
        assert self.num_pes != 0
        max_filter_area = max_filter_dim ** 2
        if self.num_pes >= max_filter_dim:
            self.par_in_filter = max_filter_area
        elif self.num_pes >= max_filter_dim:
            self.par_in_filter = max_filter_dim

        par_h = 1
        par_w = 1
        par_in_filter = self.par_in_filter
        h_saturated = par_h >= min_layer_height
        w_saturated = par_w >= min_layer_width

        while par_in_filter * par_h * par_w <= self.num_pes / 2 and not (h_saturated and w_saturated):
            par_h, par_w, h_saturated, w_saturated = \
                self.scale_dims_h_w(
                    min_layer_height, min_layer_width, par_h, par_w, h_saturated, w_saturated)

        self.par_height = par_h
        self.par_width = par_w

    def allocate_PEs_OFMS_H_W_st2(self, min_layer_filters, min_layer_height, min_layer_width,
                                  median_layer_filters, median_layer_height, median_layer_width):
        assert self.num_pes != 0
        par_ofms = mapping_general_utils.highest_power_of_2_divides_num(
            median_layer_filters)
        par_h = mapping_general_utils.highest_power_of_2_divides_num(
            min_layer_height)
        par_w = mapping_general_utils.highest_power_of_2_divides_num(
            min_layer_width)
        max_disprop = 32

        # make sure of porportions
        par_ofms = min(par_ofms, max_disprop * min(par_h, par_w))
        par_h = min(par_h, max_disprop * max(par_ofms, par_w))
        par_w = min(par_w, max_disprop * max(par_ofms, par_h))

        overall_pes = par_ofms * par_h * par_w
        if overall_pes <= self.num_pes / 2:
            # here: the pes on the dims are the max that divide the dims
            # so, it is better to distribute over two dims rather than three (better utilization)
            if par_ofms == max(par_ofms, par_h, par_w):
                par_w = min(par_ofms, utils.pow_of_2_geq(min_layer_width),
                            utils.pow_of_2_leq(int(self.num_pes / overall_pes)))

        overall_pes = par_ofms * par_h * par_w
        # scale in the direction that gives min underutilization
        if overall_pes <= self.num_pes / 2:
            if par_ofms < max_disprop * min(par_h, par_w):
                while par_h <= par_w and (par_ofms * 2 <= max_disprop * par_h) and \
                    (median_layer_filters % (par_ofms * 2)) / median_layer_filters <= \
                        (min_layer_height % (par_h * 2))/min_layer_height:
                    par_ofms *= 2
                while par_w < par_h and (par_ofms * 2 <= max_disprop * par_w) and \
                        (median_layer_filters % (par_ofms * 2)) / median_layer_filters < \
                        (min_layer_width % (par_w * 2))/min_layer_width:
                    par_ofms *= 2

        # make sure of being within the pes budget
        while par_ofms * par_h * par_w > self.num_pes:
            if par_h == max(par_ofms, par_h, par_w):
                par_h /= 2
            elif par_w == max(par_ofms, par_h, par_w):
                par_w /= 2
            else:
                if par_ofms > self.MIN_OFMS_PARALLEL or par_ofms > self.num_pes:
                    par_ofms /= 2
                elif par_h == max(par_h, par_w):
                    par_h /= 2
                else:
                    par_w /= 2

        min_ofms_saturated = par_ofms >= min_layer_filters
        min_h_saturated = par_h >= min_layer_height
        min_w_saturated = par_w >= min_layer_width
        median_ofms_saturated = par_ofms >= median_layer_filters
        median_h_saturated = par_h >= median_layer_height
        median_w_saturated = par_w >= median_layer_width

        # minimize worst case underutilization
        while par_ofms * par_h * par_w <= self.num_pes / 2:
            if not (min_h_saturated and min_w_saturated and min_ofms_saturated):
                par_ofms, par_h, par_w, min_ofms_saturated, min_h_saturated, min_w_saturated = self.scale_dims_ofms_h_w(
                    min_layer_filters, min_layer_height, min_layer_width,
                    par_ofms, par_h, par_w,
                    min_ofms_saturated, min_h_saturated, min_w_saturated
                )
            elif not (median_h_saturated and median_w_saturated and median_ofms_saturated):
                par_ofms, par_h, par_w, median_ofms_saturated, median_h_saturated, median_w_saturated = self.scale_dims_ofms_h_w(
                    median_layer_filters, median_layer_height, median_layer_width,
                    par_ofms, par_h, par_w,
                    median_ofms_saturated, median_h_saturated, median_w_saturated
                )
            else:
                break

        self.par_ofms = par_ofms
        self.par_height = par_h
        self.par_width = par_w

    def allocate_PEs_OFMS_W(self, min_layer_filters, min_layer_width,
                            median_layer_filters, median_layer_width):
        par_ofms = min(self.MIN_OFMS_PARALLEL, self.num_pes)
        par_w = 1
        ofms_saturated = (
            par_ofms * 2) >= min_layer_filters or min_layer_filters % (par_ofms * 2) != 0
        w_saturated = (
            par_w * 2) >= min_layer_width or min_layer_width % (par_w * 2) != 0
        # aiming for 0 underutilization
        while par_ofms * par_w <= self.num_pes / 2 and not (w_saturated and ofms_saturated):
            if (par_ofms == min(par_ofms, par_w) or w_saturated) and not ofms_saturated:
                par_ofms *= 2
            elif not w_saturated:
                par_w *= 2

            ofms_saturated = (
                par_ofms * 2) >= min_layer_filters or min_layer_filters % (par_ofms * 2) != 0
            w_saturated = (
                par_w * 2) >= min_layer_width or min_layer_width % (par_w * 2) != 0
        # reducing the median underutilization
        while par_ofms * par_w <= self.num_pes / 2:
            if (median_layer_filters % (par_ofms * 2)) / median_layer_filters <= \
                    (median_layer_width % (par_w * 2)) / median_layer_width:
                par_ofms *= 2
            else:
                par_w *= 2

        self.par_ofms = par_ofms
        self.par_width = par_w

    def allocate_PEs_OFMS_IFM(self, min_layer_filters, min_layer_depth,
                              median_layer_filters, median_layer_depth):
        par_ofms = min(self.MIN_OFMS_PARALLEL, self.num_pes)
        par_d = 1
        ofms_saturated = (
            par_ofms * 2) >= min_layer_filters or min_layer_filters % (par_ofms * 2) != 0
        d_saturated = (
            par_d * 2) >= min_layer_depth or min_layer_depth % (par_d * 2) != 0
        # aiming for 0 underutilization
        while par_ofms * par_d <= self.num_pes / 2 and not (d_saturated and ofms_saturated):
            if (par_ofms == min(par_ofms, par_d) or d_saturated) and not ofms_saturated:
                par_ofms *= 2
            elif not d_saturated:
                par_d *= 2

            ofms_saturated = (
                par_ofms * 2) >= min_layer_filters or min_layer_filters % (par_ofms * 2) != 0
            d_saturated = (
                par_d * 2) >= min_layer_depth or min_layer_depth % (par_d * 2) != 0

        # reducing the median underutilization
        while par_ofms * par_d <= self.num_pes / 2:
            if (median_layer_filters % (par_ofms * 2)) / median_layer_filters < \
                    (median_layer_depth % (par_d * 2)) / median_layer_depth:
                par_ofms *= 2
            else:
                par_d *= 2

        self.par_ofms = par_ofms
        self.par_ifms = par_d

    def calc_layer_exec_time(self, layer_specs, to_produce_row_count=-1):
        [num_filters, ofms_height,
            ofms_width] = utils.get_layer_ofms_shape(layer_specs)
        if to_produce_row_count != -1:
            ofms_height = to_produce_row_count
        if utils.is_dw_layer(layer_specs):
            ifms_depth = 1
        else:
            ifms_depth = layer_specs['ifms_shape'][0]
        filter_dim = utils.get_filter_dim(layer_specs)

        estimated_time = math.ceil(num_filters / self.par_ofms) * math.ceil(ofms_height / self.par_height) * \
            math.ceil(ofms_width / self.par_width) * \
            math.ceil(ifms_depth / self.par_ifms) * \
            (filter_dim * filter_dim / self.par_in_filter)

        return estimated_time

    def calc_layers_exec_times(self, model_dag, engine_layers, to_produce_row_counts=None, targeted_layer_type='all'):
        layers_exec_times = {}
        for i in range(len(engine_layers)):
            layer_index = engine_layers[i]
            layer_specs = model_dag[layer_index]
            if 'all' not in targeted_layer_type and layer_specs['type'] not in targeted_layer_type:
                continue

            layer_ofms_shape = utils.get_layer_ofms_shape(layer_specs)
            layer_ofms_height = layer_ofms_shape[1]
            to_produce_row_count = layer_ofms_height
            if to_produce_row_counts is not None and len(to_produce_row_counts) != 0:
                to_produce_row_count = int(to_produce_row_counts[i])
            layers_exec_times[i] = self.calc_layer_exec_time(
                layer_specs, to_produce_row_count)

        return layers_exec_times

    def distribute_PEs_on_dims(self, model_dag, engine_layers, to_produce_row_counts=None, targeted_layer_type='all'):
        layers_num_filters = []
        layers_depths = []
        layers_heights = []
        layers_widths = []
        layers_filter_shapes = []
        num_layers = 0
        for i in range(len(engine_layers)):
            layer_index = engine_layers[i]
            layer_specs = model_dag[layer_index]
            if 'all' not in targeted_layer_type and layer_specs['type'] not in targeted_layer_type:
                continue

            num_layers += 1
            layer_ifms_shape = utils.get_layer_ifms_shape(layer_specs)
            layer_ofms_shape = utils.get_layer_ofms_shape(layer_specs)
            layer_ofms_height = layer_ofms_shape[1]
            if to_produce_row_counts is not None and len(to_produce_row_counts) != 0:
                layer_ofms_height = int(to_produce_row_counts[i])
            layers_num_filters.append(layer_ofms_shape[0])
            layers_filter_shapes.append(utils.get_filter_dim(layer_specs))
            layers_depths.append(layer_ifms_shape[0])
            layers_heights.append(layer_ofms_height)
            layers_widths.append(layer_ofms_shape[2])

        layers_num_filters.sort()
        layers_heights.sort()
        layers_widths.sort()
        layers_depths.sort()
        min_layer_filters = layers_num_filters[0]
        min_layer_height = layers_heights[0]
        min_layer_width = layers_widths[0]
        min_layer_depth = layers_depths[0]

        max_filter_dim = max(layers_filter_shapes)
        self.max_filter_dim = max_filter_dim

        median_layer_filters = layers_num_filters[int(num_layers / 2)]
        median_layer_height = layers_heights[int(num_layers / 2)]
        median_layer_width = layers_widths[int(num_layers / 2)]
        median_layer_depth = layers_depths[int(num_layers / 2)]

        if self.parallelization_strategy == ParallelizationStrategies.OFMS_H_W:
            self.allocate_PEs_OFMS_H_W_st2(min_layer_filters, min_layer_height, min_layer_width,
                                           median_layer_filters, median_layer_height, median_layer_width)
        elif self.parallelization_strategy == ParallelizationStrategies.OFMS_W:
            self.allocate_PEs_OFMS_W(
                min_layer_filters, min_layer_width, median_layer_filters, median_layer_width)
        elif self.parallelization_strategy == ParallelizationStrategies.OFMS_IFM:
            self.allocate_PEs_OFMS_IFM(
                min_layer_filters, min_layer_depth, median_layer_filters, median_layer_depth)
        elif self.parallelization_strategy == ParallelizationStrategies.IN_FILTER_H_W:
            self.allocate_PEs_IN_FILTER_H_W(
                min_layer_height, min_layer_width, max_filter_dim)
        elif self.parallelization_strategy == ParallelizationStrategies.CUSTOM:
            self.allocate_PEs_OFMS_H_W_st2(min_layer_filters, min_layer_height, min_layer_width,
                                           median_layer_filters, median_layer_height, median_layer_width)
            exec_time1 = sum(list(self.calc_layers_exec_times(model_dag, engine_layers,
                                                             to_produce_row_counts, targeted_layer_type).values()))
            self.reset_parallelism()
            self.allocate_PEs_OFMS_W(
                min_layer_filters, min_layer_width, median_layer_filters, median_layer_width)
            exec_time2 = sum(list(self.calc_layers_exec_times(model_dag, engine_layers,
                                                             to_produce_row_counts, targeted_layer_type).values()))
            self.reset_parallelism()
            self.allocate_PEs_OFMS_IFM(
                min_layer_filters, min_layer_depth, median_layer_filters, median_layer_depth)
            exec_time3 = sum(list(self.calc_layers_exec_times(model_dag, engine_layers,
                                                             to_produce_row_counts, targeted_layer_type).values()))
            self.reset_parallelism()
            if exec_time1 < exec_time2 and exec_time1 < exec_time3:
                self.parallelization_strategy = ParallelizationStrategies.OFMS_H_W
                self.allocate_PEs_OFMS_H_W_st2(min_layer_filters, min_layer_height, min_layer_width,
                                           median_layer_filters, median_layer_height, median_layer_width)
            elif exec_time2 < exec_time3:
                self.parallelization_strategy = ParallelizationStrategies.OFMS_W
                self.allocate_PEs_OFMS_W(
                    min_layer_filters, min_layer_width, median_layer_filters, median_layer_width)
            else:
                self.parallelization_strategy = ParallelizationStrategies.OFMS_IFM
                self.allocate_PEs_OFMS_IFM(
                    min_layer_filters, min_layer_depth, median_layer_filters, median_layer_depth)
            
    def get_parallelism_fms(self):
        return int(self.par_ifms * self.par_height * self.par_width)

    def get_parallelism_weights(self):
        return int(self.par_ofms * self.par_ifms * self.par_in_filter)

    def get_ports_weights(self):
        return int(self.par_ofms * self.par_in_filter)
    
    def get_parallelism(self):
        return int(self.par_ofms * self.par_ifms * self.par_in_filter * self.par_height * self.par_width)

    def get_parallelism_dims(self):
        return int(self.par_ofms), int(self.par_ifms), int(self.par_in_filter), int(self.par_height), int(self.par_width)
