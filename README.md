# Official Repo for "Terafly : A Multi-Node FPGA Based Accelerator Design for Efficient Cooperative Inference in LLMs"


![demo](assets/opt-1.3b.gif)

## Prerequesites

Before you start, you should better align with our experiment environment. For Alveo U50lv Card, our environment is 

* System: Ubuntu 18.04
* Shell:  xilinx-u50lv-gen3x4-xdma-base_2
* Xrt: 2023.2
* Vitis HLS & Vivado: 2023.2

##  Code Structure

```
template/ 													# template HLS code for generation framework
OPT-1.3b_optimize/ 											# generated code for vitis development flow
LLM-demo-gui/ 												# for webui interaction
OPT-1.3b_optimize/connectivity.cfg  						# configuration file to specify topology
codegen.py 													# script to modify the template according to configuration
OPT-1.3b.json 												# configuration file to specify performance
weight_packer.py 											# script to pack model weight into Terafly memory layout
```

## Quick Start

You can download our packed data (model weight) from here xxx

```
cd OPT-1.3b_optimize/
make run
```

This will automatically generate the xclbin file and program your Alveo card.

```
cd tokenizer/
sh ./command.sh
```

This will compile the host-side application to run the `lambada` benchmark. You may check the `tokenizer_predict_eigen.cpp` to make sure the code correctly loaded the packed data.

You can also run this code 

```
cd LLM-demo-gui/alveo
(python==3.6) python client-v3.py
```

Then open the html file`LLM-demo-gui/llm-gui/web/index.html ` to chat with LLM.



 




