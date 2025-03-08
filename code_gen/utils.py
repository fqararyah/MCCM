from multiprocessing.dummy import active_children
import sys
import pathlib
import json

current_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(current_dir)
print(current_dir)

DELIMITER = '::'

NET_PREFIX = 'mob_v2'
NET_FULL_NAME = 'mobilenet_v2'
input_folder = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/models_archs/models/'\
    + NET_FULL_NAME + '/'
IFMS_FILE = input_folder + 'layers_inputs.txt'
OFMS_FILE = input_folder + 'layers_outputs.txt'
LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
SECONDARY_LAYERS_TYPES_FILE = input_folder + 'secondary_layers_types.txt'
LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'
LAYERS_SKIP_CONNECTIONS_FILE = input_folder + 'skip_connections_indices.txt'
LAYERS_ACTIVATIONS_FILE = input_folder + 'layers_activations.txt'
LAYERS_EXECUTION_SEQUENCE = input_folder + 'layers_execution_sequence.txt'
MODEL_DAG_FILE = input_folder + 'model_dag.json'

def set_globals(prefix, full_name):
    global NET_PREFIX, NET_FULL_NAME, input_folder, IFMS_FILE, OFMS_FILE, LAYERS_TYPES_FILE, LAYERS_WEIGHTS_FILE, LAYERS_STRIDES_FILE, EXPANSION_PROJECTION_FILE, LAYERS_RELUS_FILE, LAYERS_SKIP_CONNECTIONS_FILE, SECONDARY_LAYERS_TYPES_FILE, LAYERS_ACTIVATIONS_FILE,\
        LAYERS_EXECUTION_SEQUENCE, MODEL_DAG_FILE
    NET_PREFIX = prefix
    NET_FULL_NAME = full_name
    input_folder = '../extract_tflite_model_metadata/models_archs/models/'\
        + NET_FULL_NAME + '/'
    IFMS_FILE = input_folder + 'layers_inputs.txt'
    OFMS_FILE = input_folder + 'layers_outputs.txt'
    LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
    SECONDARY_LAYERS_TYPES_FILE = input_folder + 'secondary_layers_types.txt'
    LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
    LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
    EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
    LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'
    LAYERS_ACTIVATIONS_FILE = input_folder + 'layers_activations.txt'
    LAYERS_SKIP_CONNECTIONS_FILE = input_folder + 'skip_connections_indices.txt'
    LAYERS_EXECUTION_SEQUENCE = input_folder + 'layers_execution_sequence.txt'
    MODEL_DAG_FILE = input_folder + 'model_dag.json'


def clean_line(line):
    return line.replace(' ', '').replace('\n', '')

def read_model_dag():
    f = open(MODEL_DAG_FILE)
    return json.load(f)

def has_dw_layers(model_dag):
    
    for layer_specs in model_dag:
        if('type' in layer_specs and layer_specs['type'] == 'dw'):
            return True
    return False
