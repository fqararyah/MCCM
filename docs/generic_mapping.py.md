<!-- markdownlint-disable -->

<a href="../mapping_types/generic_mapping.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `generic_mapping.py`






---

## <kbd>class</kbd> `GenericMapping`
This class defines the basic interface of a mapping (i.e. mutlipe-CE accelerator). A customized multiple-CE accelerator needs mus inherit this class 

<a href="../mapping_types/generic_mapping.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    hw_config,
    model_dag,
    layers=None,
    engines=None,
    first_layer_ifms_are_on_chip=False,
    last_layer_ofms_are_on_chip=False
) → None
```

Constructor. 



**Parameters:**
 hw_config (HWConfig): basic HW configurations, see HWConfig Class. model_dag (dict): DNN model description. layers (list): in case the mapping targets subset of the layers rather than the whole model. engines (list): in case the engines configurations are already specified. first_layer_ifms_are_on_chip (boolean): in case the mapping targets subet of the model and the input of the first targeted layer is already stored on-chip. last_layer_ofms_are_on_chip (boolean): in case the mapping targets subet of the model and the output of the last targeted layer is to be stored on-chip. 




---

<a href="../mapping_types/generic_mapping.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_exec_time`

```python
calc_exec_time(print_desc=False)
```

This function calculates the end-to-end inference time. 



**Returns:**
 float: execution time in seconds. 

---

<a href="../mapping_types/generic_mapping.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_fms_buffer_sz`

```python
calc_fms_buffer_sz(print_desc=False)
```

This function calculates the size of the on-chip FMs buffer. 



**Returns:**
 int: FMs buffer size. 

---

<a href="../mapping_types/generic_mapping.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_off_chip_fms_access`

```python
calc_off_chip_fms_access(print_desc=False)
```

This function calculates the number of the off-chip FMs accesses. 



**Returns:**
 int: number of FMs accesses. 

---

<a href="../mapping_types/generic_mapping.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_off_chip_weights_access`

```python
calc_off_chip_weights_access()
```

This function calculates the number of the off-chip weights accesses. 



**Returns:**
 int: number of weight accesses. 

---

<a href="../mapping_types/generic_mapping.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_throughput`

```python
calc_throughput()
```

This function calculates the throughput as frames per second (FPS). 



**Returns:**
 float: FPS. 

---

<a href="../mapping_types/generic_mapping.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_weights_buffer_sz`

```python
calc_weights_buffer_sz()
```

This function calculates the size of the on-chip weights buffer. 



**Returns:**
 int: weights buffer size. 

---

<a href="../mapping_types/generic_mapping.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_segment_exec_times`

```python
get_segment_exec_times()
```

This function returns the breakdown of per-segment execution time. 



**Returns:**
 list: per-segment execution time. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
