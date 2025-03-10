import __init__
from enum import Enum
from abc import ABC, abstractmethod

class ParallelizationStrategies(Enum):
    OFMS_IFM = 0
    OFMS_W = 1
    OFMS_H = 2
    OFMS_H_W = 3
    IN_FILTER_H_W = 4
    CUSTOM = 11

class GenericEngine(ABC):

    def __init__(self, num_pes, par_ofms=1, par_ifms=1, par_width=1, par_height=1, par_in_filter=1,
                 parallelization_strategy=ParallelizationStrategies.OFMS_H_W):
        '''
        Constructor.

        Parameters:
        num_pes (int): number of PEs in the engine.
        par_ofms (int): parallelism on OFMs / filters dimension.
        par_ifms (int): parallelism on IFMs channels dimension.
        par_width (int): parallelism on FMs width dimension.
        par_height (int): parallelism on FMs height dimension.
        par_in_filter (int): parallelism within a filter, i.e. its width and height.
        parallelization_strategy (ParallelizationStrategies): on which dimensions the parallelism of the engine is.
        '''
        
        self.num_pes = num_pes
        self.par_ofms = par_ofms
        self.par_ifms = par_ifms
        self.par_height = par_height
        self.par_width = par_width
        self.par_in_filter = par_in_filter
        self.parallelization_strategy = parallelization_strategy

    @abstractmethod
    def calc_layer_exec_time(self, layer_specs, to_produce_row_count=-1):
        '''
        This function calculates a layer execution time in cycles.

        Args:
        layer_specs(dict): specifications of a layer.
        to_produce_row_count(int): in case the engine produces subset of the layer output rows at a time.

        Returns:
        int: execution time in cycles.
        '''

        pass

    @abstractmethod
    def distribute_PEs_on_dims(self, model_dag, engine_layers, to_produce_row_counts=None, targeted_layer_type='all'):
        '''
        This decides an engine paralllelism given a set of layers it will process.

        Args:
        model_dag (dict): DNN model description.
        engine_layers(list): indices of the dag layers processed by the engine.
        to_produce_row_counts (list): in case the engine processes subset of the layers rows at a time.
        targeted_layer_type (layer_type): in case a CNN has multiple layer types and the engine processes
        only certain type/s.
        '''

        pass
    
    @abstractmethod
    def get_parallelism_fms(self):
        '''
        This function reterns the parallelism on the FMs (depth * height * width).

        Returns:
        int: parallelism on the FMs.
        '''

        pass

    @abstractmethod
    def get_parallelism_weights(self):
        '''
        This function reterns the parallelism on the weights (filters * depth * width * height).

        Returns:
        int: parallelism on the weights.
        '''

        pass

    @abstractmethod
    def get_ports_weights(self):
        pass

    @abstractmethod    
    def get_parallelism(self):
        '''
        This function reterns the engine parallelism (on tweights * FMs).

        Returns:
        int: engine parallelism.
        '''

        pass

    @abstractmethod
    def get_parallelism_dims(self):
        '''
        This function reterns the engine parallelism on each of the dimensions.

        Returns:
        tuple: engine parallelism (par_ofms / filters, par_ifms, par_in_filter (width * height), FMs height, FMs width)).
        '''
        pass
