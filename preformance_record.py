from enum import Enum


class Metrics(Enum):
    THROUGHPUT = 0
    LATENCY = 1
    BUFFER = 2
    ACCESS = 3
    REQUIED_BW = 4
    ENERGY = 5

def first_better_than_second(metric, val1, val2):
    ret_val = False
    if val1 < val2:
        ret_val = True
        
    if metric == Metrics.THROUGHPUT:
        ret_val = not ret_val
        
    return ret_val

def first_better_than_or_equal_second(metric, val1, val2):
    return first_better_than_second(metric, val1, val2) or val1 == val2

def normalize_metrics(metric, base_val, val2, speedup = True):
    if metric == Metrics.LATENCY and speedup:
        return base_val / val2
    else:
        return val2 / base_val
    
class PerformanceRecord():

    def __init__(self, board_name=None, model_name=None, mapping_name=None, num_engines=0,
                 latency=0, throughput=0,
                 on_chip_buffer_fms=0,
                 on_chip_buffer_weights=0, on_chip_buffer=0,
                 off_chip_access_fms = 0, off_chip_access_weights = 0, 
                 required_bw = 0,
                 energy = 0,
                 segment_exec_times=None) -> None:
        self.board_name = board_name
        self.model_name = model_name
        self.mapping_name = mapping_name
        self.latency = latency
        self.segment_exec_times = segment_exec_times
        self.throughput = throughput
        self.num_engines = num_engines
        self.on_chip_buffer_fms = on_chip_buffer_fms
        self.on_chip_buffer_weights = on_chip_buffer_weights
        self.on_chip_buffer = on_chip_buffer
        self.off_chip_access_fms = off_chip_access_fms
        self.off_chip_access_weights = off_chip_access_weights
        self.off_chip_access = self.off_chip_access_fms + self.off_chip_access_weights
        self.required_bw = required_bw
        self.energy = energy

    def __str__(self):
        return "board: {}, model: {}, mapping: {}, latency {}, throughput {}, on_chip {}, access {}, bw {}".format(
            self.board_name, self.model_name, self.mapping_name, self.latency, self.throughput,
            self.on_chip_buffer, self.off_chip_access, self.required_bw)

    def totals_as_list(self):
        return [self.mapping_name, self.latency, self.throughput, self.on_chip_buffer, self.off_chip_access]

    def is_better(self, metric, value):
        if metric == Metrics.ACCESS:
            return value < self.off_chip_access
        elif metric == Metrics.BUFFER:
            return value < self.on_chip_buffer
        elif metric == Metrics.THROUGHPUT:
            return value > self.throughput
        elif metric == Metrics.LATENCY:
            return value < self.latency
        elif metric == Metrics.REQUIED_BW:
            return value < self.required_bw
        elif metric == Metrics.ENERGY:
            return value < self.energy

    def get_metric_val(self, metric):
        if metric == Metrics.ACCESS:
            return self.off_chip_access
        elif metric == Metrics.BUFFER:
            return self.on_chip_buffer
        elif metric == Metrics.THROUGHPUT:
            return self.throughput
        elif metric == Metrics.LATENCY:
            return self.latency
        elif metric == Metrics.REQUIED_BW:
            return self.required_bw
        elif metric == Metrics.ENERGY:
            return self.energy

    def init_from_dict(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)
