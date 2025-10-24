# 🚀 Terafly: A Multi-Node FPGA-Based Accelerator for Efficient Cooperative LLM Inference

> **Terafly** enables high-throughput, low-latency inference of Large Language Models (LLMs) by leveraging a multi-node FPGA architecture optimized for cooperative execution.

![Demo](assets/opt-1.3b.gif)

---

## 💡 Highlight

We provide **HLS kernels** that can be rapidly customized for research purposes, enabling efficient experimentation and algorithm validation on FPGAs.

---

## 🔍 Overview

Terafly is designed to maximize memory bandwidth and computational efficiency on FPGA platforms—specifically targeting embedded and datacenter FPGAs like the **Xilinx Alveo U50lv**. It supports end-to-end LLM inference with minimal host intervention, and includes tooling for weight packing, hardware generation, and interactive demo deployment.

---

## 📚 Related Work

If you're exploring FPGA-based LLM acceleration, you might also be interested in:

- [**llama-fpga**](https://github.com/adamgallas/llama-fpga)  

---

## ⚙️ Prerequisites

To ensure compatibility, we recommend replicating our experimental environment:

| Component        | Version / Configuration                     |
|------------------|---------------------------------------------|
| **OS**           | Ubuntu 18.04                                |
| **Shell**        | `xilinx-u50lv-gen3x4-xdma-base_2`           |
| **XRT**          | 2023.2                                      |
| **Vitis HLS & Vivado** | 2023.2                              |

> 💡 Ensure your Alveo U50lv card is properly flashed with the matching shell.

---

## 📂 Code Structure

| File/Directory | Description |
| :--- | :--- |
| `template/` | Template **HLS code** used by the generation framework. |
| `OPT-1.3b_optimize/` | Directory for the **generated code** tailored for the Vitis development flow. |
| `LLM-demo-gui/` | Contains files for **WebUI interaction**. |
| `OPT-1.3b_optimize/connectivity.cfg` | **Configuration file** to specify the multi-node accelerator topology. |
| `codegen.py` | Python **script** to modify the template based on configuration. |
| `OPT-1.3b.json` | **Configuration file** to specify performance and model parameters. |
| `weight_packer.py` | Python **script** to pack model weights into the Terafly memory layout. |

---

## ⚡ Quick Start

Follow these steps to quickly set up and run the Terafly accelerator.

### 1. Download Model Weights
Download the pre-packed model weights (**OPT-1.3B**) from the provided link:
[Model Weights Download](https://pan.baidu.com/s/1HENc02MA4etf2cCWuMtApw?pwd=bcbf) (Password: `bcbf`).

### 2. Compile and Program the FPGA
Navigate to the optimized code directory and run the compilation command. This will automatically generate the `xclbin` file and program your Alveo card.

```shell
cd OPT-1.3b_optimize/
make run
```

### 3. Run the Benchmark (`lambada`)
Compile and execute the host-side application to run the **`lambada` benchmark**.
* **Note**: Check `tokenizer_predict_eigen.cpp` to verify that the code correctly loads the packed data.

```shell
cd tokenizer/
sh ./command.sh
```

### 4. Run the Web Demo
You can also interact with the LLM via a WebUI interface:

1.  Start the Python server (requires `python==3.6`).

2.  Open the web interface in your browser: `LLM-demo-gui/llm-gui/web/index.html`.
    (Please open the HTML file directly in your browser to chat with the LLM.)

```shell
cd LLM-demo-gui/alveo
(python==3.6) python client-v3.py
```


## 📝 Citation

If you find Terafly or LoopLynx useful in your research or project, please cite our papers. We appreciate your interest in our work!

```bibtex
@ARTICLE{Terafly,
  author={Zheng, Jianing and Chen, Gang and Huang, Libo and Lou, Xin and Zheng, Wei-shi},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={Terafly : A Multi-Node FPGA Based Accelerator Design for Efficient Cooperative Inference in LLMs},
  year={2025},
  volume={},
  number={},
  pages={1-1}}

@inproceedings{LoopLynx,
  author         = {Jianing Zheng and Gang Chen},
  title          = {LoopLynx: {A} Scalable Dataflow Architecture for Efficient {LLM} Inference},
  booktitle      = {Design, Automation {\&} Test in Europe Conference, {DATE} 2025, Lyon, France, March 31 - April 2, 2025},
  pages          = {1--7},
  publisher      = {{IEEE}},
  year           = {2025}}
```
