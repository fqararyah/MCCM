<!-- markdownlint-disable -->

<a href="../mapping_types/generic_mapping.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `generic_mapping.py`






---

## <kbd>class</kbd> `GenericMapping`
This class defines the basic interface of a mapping (i.e. mutlipe-CE accelerator) A customized multiple-CE accelerator needs mus inherit this class 

<a href="../mapping_types/generic_mapping.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    hw_config,
    layers=None,
    engines=None,
    first_layer_ifms_are_on_chip=False,
    last_layer_ofms_are_on_chip=False
) → None
```








---

<a href="../mapping_types/generic_mapping.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_exec_time`

```python
calc_exec_time(print_desc=False)
```





---

<a href="../mapping_types/generic_mapping.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_fms_buffer_sz`

```python
calc_fms_buffer_sz(print_desc=False)
```





---

<a href="../mapping_types/generic_mapping.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_off_chip_fms_access`

```python
calc_off_chip_fms_access(print_desc=False)
```





---

<a href="../mapping_types/generic_mapping.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_off_chip_weights_access`

```python
calc_off_chip_weights_access()
```





---

<a href="../mapping_types/generic_mapping.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_throughput`

```python
calc_throughput()
```





---

<a href="../mapping_types/generic_mapping.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calc_weights_buffer_sz`

```python
calc_weights_buffer_sz()
```





---

<a href="../mapping_types/generic_mapping.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_segment_exec_times`

```python
get_segment_exec_times()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
