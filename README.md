# An Analytical Cost Model for Fast Evaluation of Multiple Compute-Engine CNN Accelerators

This repository contains an implementation of <u>M</u>ultiple <u>C</u>ompute-engine accelerator analytical <u>C</u>ost <u>M</u>odel (__MCCM__) proposed in following paper: [An Analytical Cost Model for Fast Evaluation of Multiple Compute-Engine CNN Accelerators](https://doi.org/10.48550/arXiv.2503.07242)

If you find this repository useful for your research, we would appreciate it if you could cite us using the following BibTeX entry:

```
@misc{qararyah2025analyticalcostmodelfast,
      title={An Analytical Cost Model for Fast Evaluation of Multiple Compute-Engine CNN Accelerators}, 
      author={Fareed Qararyah and Mohammad Ali Maleki and Pedro Trancoso},
      year={2025},
      eprint={2503.07242},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2503.07242}, 
}
```

## Multiple-CE Accelerators
We use the term **multiple Compute-Engine (multiple-CE)** accelerators to describe accelerators that organize FPGA resources into an adjustable number of dedicated Compute Engines (CEs). These multiple-CE accelerators adjust the number of CEs to optimize performance and/or efficiency metrics based on the structure of the CNN model and the available FPGA resources.

The existing multiple-CE accelerators can be categorized into three categories, shown in the figure below. We call these categories **Segmented**, **SegmentedRR**, and **Hybrid**. These categories of accelerators differ in how they arrange the FPGA resources into CEs and in how they map the CNN layers to these CEs. For more details about these accelerator's architectures and pointers to the related literature, please check the paper ([An Analytical Cost Model for Fast Evaluation of Multiple Compute-Engine CNN Accelerators](https://doi.org/10.48550/arXiv.2503.07242)).

In this repository, we use slightly different terminology. We use the term **mapping** to refer to multiple-CE accelerators since each multiple-CE architecture is basically a mapping of a CNN to hardware. 

![Multiple-CE Accelerators](./paper_figures/multi_engine_mappings_detailed.drawio.png "Multiple-CE Accelerators")*Figure 1: Existing multiple-CE accelerator categories.*

## <u>M</u>ultiple <u>C</u>ompute-engine accelerator analytical <u>C</u>ost <u>M</u>odel (MCCM)

Analysing the existing multiple-CE accelerators, e.g. the examples shown in ![Figure1](#), reveals that they are composed of two basic building blocks. We call the first block *single-CE* processing a range of layers one by one. For instance, `CE_1` processing layers `L_1-L_4` in the Segmented, and `CE_4` processing `L_4` to the last layer in the Hybrid. The second building block is a set of *pipelined-CEs*, each processing a layer. For example, `CE_1-CE_4` processing layers `L_1-L_4` in SegmentedRR, and `CE_1-CE_3` processing layers `L_1-L_3` in Hybrid. These two blocks can be used to express a generic multiple-CE architecture using the following notation:

- `CE_x`: denotes a single-CE block, and `CE_x-CE_y`: denotes `(y - x) + 1` pipelined-CEs block.
- `{L_x-L_y: CE_z}`: denotes that layers `x` to `y` are processed using a single-CE block (`CE_z`) sequentially. A special case is to have one layer only, namely `{L_x: CE_z}`.
- `{L_x-L_y: CE_z-CE_w}`: denotes that layers `x` to `y` are processed using pipelined-CEs block. If the number of layers exceeds the number of CEs, the pipelined-CEs block is assumed to process `(w - z) + 1` layers at a time.

MCCM models a generic multiple-CE accelerator using a bottom-up approach. It evaluates the accelerator performance and efficiency using the models of the single-CE and pipelined-CE blocks, and the interfaces between them. MCCM mainly estimates a multiple-CE accelerator *latency*, *throughput*, *on-chip buffer requirements* and *off-chip accesses*. For more details on the model, please check the paper.

### MCCM example
MCCM takes as an input\
__(1)__ Multiple-CE accelerator description, using the notation described above.\
__(2)__ A CNN DAG representation. Examples can be found in `./models_archs/`\
__(3)__ FPGA's number of PEs (DSPs), off-chip memory bandwidth, and on-chip memory capacity. Examples can be fond in  `./config_files/`

Using MCCM is very simple. [estimations_example.py](./estimations_example.py) shows how to use MCCM to estimate the performance, on-chip buffers, and off-chip accesses of ResNet50 on ZCU102.

### Extending MCCM

MCCM models are simple and generic. However, there is a trade-off between generality and accuracy. For users interested in modeling specialized architecture accurately, MCCM can be extended by the integration of new models.\
We expect that modifications of extensions of MCCM would happen at one of two levels:\
__(1)__ Intra-CE:\
Intra-CE modifications may include adding or modifying the processing element distribution heuristics and parallelism strategies, or the execution time estimation. A new engine model needs to extend `GenericEngine` in [generic_engine.py](engines/generic_engine.py). The documentation of the functions in `GenericEngine` is found in [generic_engine.py.md](docs/generic_engine.py.md). 

__(2)__ Inter-CE (block and mapping-level):\
We offer modeling of a generic multiple-CE mapping, the blocks mapping, and customized modeling of some mappings including the mentioned **Segmented**, **SegmentedRR**, and **Hybrid**. A user interested in adding new custom mappings or modifying the existing block or mapping estimators needs to extend `GenericMapping` in [generic_mapping.py](mapping_types/generic_mapping.py). The documentation of the functions in `GenericMapping` is found in [generic_mapping.py.md](docs/generic_mapping.py.md). 