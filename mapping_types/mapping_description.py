from preformance_record import *
import __init__
import utils
import copy
import mapping_utils.mapping_general_utils as mapping_general_utils

class MappingDescription():

    def __init__(self, board_name=None, model_name=None, model_dag=None, num_segments=-1, num_blocks=-1, 
                 segment_layers_list=None, block_engines_list=None,
                 segment_block_list=None, timing_metric=Metrics.THROUGHPUT):
        self.board_name = board_name
        self.model_name = model_name
        self.model_dag = model_dag
        self.num_segments = num_segments
        self.num_blocks = num_blocks
        self.segment_layers_list = segment_layers_list
        self.block_engines_list = block_engines_list
        self.segment_block_list = segment_block_list
        self.timing_metric = timing_metric

    def copy_mapping_desc(self):
        return MappingDescription(board_name=self.board_name, model_name=self.model_name, model_dag=copy.copy(self.model_dag), 
                                  num_segments=self.num_segments, num_blocks=self.num_blocks, 
                                  segment_layers_list= mapping_general_utils.copy_2d_list(self.segment_layers_list), 
                                  block_engines_list= mapping_general_utils.copy_2d_list(self.block_engines_list),
                                  segment_block_list=copy.copy(self.segment_block_list), 
                                  timing_metric=self.timing_metric)

    def mapping_desc_from_dict(self, mapping_desc_dict):
        num_conv_layers = utils.get_num_conv_layer_count_in_range(self.model_dag, 0, len(self.model_dag))
        self.segment_layers_list = []
        self.block_engines_list = []
        for layers, engines in mapping_desc_dict.items():
            self.segment_layers_list.append([])
            self.block_engines_list.append([])
            splits = layers.split('-')
            self.segment_layers_list[-1].append(int(splits[0]))
            if len(splits) > 1:
                if 'last' in splits[1].lower():
                    self.segment_layers_list[-1].append(num_conv_layers)
                else:
                    self.segment_layers_list[-1].append(int(splits[1]))
            splits = engines.split('-')
            self.block_engines_list[-1].append(int(splits[0]))
            if len(splits) > 1:
                self.block_engines_list[-1].append(int(splits[1]))

        self.num_segments = len(self.segment_layers_list)
        self.num_blocks = len(self.block_engines_list)

        if self.segment_block_list is None:
            self.segment_block_list = []
            for segment in range(self.num_segments):
                self.segment_block_list.append(segment)

    def __str__(self):
        return self.board_name + ' ' + str(self.segment_layers_list) + ' ' + str(self.block_engines_list)
