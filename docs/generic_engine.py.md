<!-- markdownlint-disable -->

<a href="../engines/generic_engine.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `generic_engine.py`






---

## <kbd>class</kbd> `GenericEngine`




<a href="../engines/generic_engine.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    num_pes,
    par_ofms=1,
    par_ifms=1,
    par_width=1,
    par_height=1,
    par_in_filter=1,
    parallelization_strategy=<ParallelizationStrategies.OFMS_H_W: 3>
)
```

Constructor. 



**Parameters:**
 num_pes (int): number of PEs in the engine. par_ofms (int): parallelism on OFMs / filters dimension. par_ifms (int): parallelism on IFMs channels dimension. par_width (int): parallelism on FMs width dimension. par_height (int): parallelism on FMs height dimension. par_in_filter (int): parallelism within a filter, i.e. its width and height. parallelization_strategy (ParallelizationStrategies): on which dimensions the parallelism of the engine is. 




---

<a href="../engines/generic_engine.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_layer_exec_time`

```python
calc_layer_exec_time(layer_specs, to_produce_row_count=-1)
```

This function calculates a layer execution time in cycles. 



**Args:**
 layer_specs(dict): specifications of a layer. to_produce_row_count(int): in case the engine produces subset of the layer output rows at a time. 



**Returns:**
 int: execution time in cycles. 

---

<a href="../engines/generic_engine.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `distribute_PEs_on_dims`

```python
distribute_PEs_on_dims(
    model_dag,
    engine_layers,
    to_produce_row_counts=None,
    targeted_layer_type='all'
)
```

This decides an engine paralllelism given a set of layers it will process. 



**Args:**
 model_dag (dict): DNN model description. engine_layers(list): indices of the dag layers processed by the engine. to_produce_row_counts (list): in case the engine processes subset of the layers rows at a time. targeted_layer_type (layer_type): in case a CNN has multiple layer types and the engine processes only certain type/s. 

---

<a href="../engines/generic_engine.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_parallelism`

```python
get_parallelism()
```

This function reterns the engine parallelism (on tweights * FMs). 



**Returns:**
 int: engine parallelism. 

---

<a href="../engines/generic_engine.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_parallelism_dims`

```python
get_parallelism_dims()
```

This function reterns the engine parallelism on each of the dimensions. 



**Returns:**
 tuple: engine parallelism (par_ofms / filters, par_ifms, par_in_filter (width * height), FMs height, FMs width)). 

---

<a href="../engines/generic_engine.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_parallelism_fms`

```python
get_parallelism_fms()
```

This function reterns the parallelism on the FMs (depth * height * width). 



**Returns:**
 int: parallelism on the FMs. 

---

<a href="../engines/generic_engine.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_parallelism_weights`

```python
get_parallelism_weights()
```

This function reterns the parallelism on the weights (filters * depth * width * height). 



**Returns:**
 int: parallelism on the weights. 

---

<a href="../engines/generic_engine.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ports_weights`

```python
get_ports_weights()
```






---

## <kbd>class</kbd> `ParallelizationStrategies`
An enumeration. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
