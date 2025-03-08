
GENERATE_WEIGHTS = True    

if GENERATE_WEIGHTS:
    import prepare_weights
    import biases_and_quantization_gen_v2

import layer_specs_gen