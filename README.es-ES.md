

# 🚀 Terafly: Un Acelerador de Múltiples Nodos Basado en FPGA para la Inferencia Cooperativa Eficiente de LLMs

> **Terafly** permite la inferencia de Modelos de Lenguaje Grande (LLM) de alto rendimiento y baja latencia, aprovechando una arquitectura FPGA de múltiples nodos optimizada para la ejecución cooperativa.

![Demo](assets/opt-1.3b.gif)

---

## 💡 Destacado

Proporcionamos **núcleos HLS** que pueden personalizarse rápidamente con fines de investigación, lo que permite una experimentación eficiente y la validación de algoritmos en FPGAs.

---

## 🔍 Descripción General

Terafly está diseñado para maximizar el ancho de banda de memoria y la eficiencia computacional en plataformas FPGA, orientándose específicamente a FPGAs integradas y para centros de datos como la **Xilinx Alveo U50lv**. Soporta la inferencia de LLM de extremo a extremo con mínima intervención del host e incluye herramientas para la compactación de pesos, generación de hardware y despliegue de demostraciones interactivas.

---

## 📚 Trabajos Relacionados

Si está explorando la aceleración de LLM basadas en FPGAs, también podría interesarle:

- [**llama-fpga**](https://github.com/adamgallas/llama-fpga)  

---

## ⚙️ Prerrequisitos

Para garantizar la compatibilidad, recomendamos replicar nuestro entorno experimental:

| Componente | Versión / Configuración |
| :--- | :--- |
| **OS** | Ubuntu 18.04 |
| **Shell** | `xilinx-u50lv-gen3x4-xdma-base_2` |
| **XRT** | 2023.2 |
| **Vitis HLS & Vivado** | 2023.2 |

> 💡 Asegúrese de que su tarjeta Alveo U50lv esté actualizada (flashed) con el shell correspondiente.

---

## 📂 Estructura del Código

| Archivo/Directorío | Descripción |
| :--- | :--- |
| `template/` | Código **HLS** de plantilla utilizado por el framework de generación. |
| `OPT-1.3b_optimize/` | Directorio para el **código generado** adaptado al flujo de desarrollo de Vitis. |
| `LLM-demo-gui/` | Contiene archivos para la **interacción con la WebUI**. |
| `OPT-1.3b_optimize/connectivity.cfg` | **Archivo de configuración** para especificar la topología del acelerador de múltiples nodos. |
| `codegen.py` | **Script** de Python para modificar la plantilla según la configuración. |
| `OPT-1.3b.json` | **Archivo de configuración** para especificar parámetros de rendimiento y del modelo. |
| `weight_packer.py` | **Script** de Python para compactar los pesos del modelo en el diseño de memoria de Terafly. |

---

## ⚡ Inicio Rápido

Siga estos pasos para configurar y ejecutar rápidamente el acelerador Terafly.

### 1. Descargar los Pesos del Modelo
Descargue los pesos del modelo precompactados (**OPT-1.3B**) desde el enlace proporcionado:
[Model Weights Download](https://pan.baidu.com/s/1HENc02MA4etf2cCWuMtApw?pwd=bcbf) (Password: `bcbf`).

### 2. Compilar y Programar la FPGA
Navegue al directorio del código optimizado y ejecute el comando de compilación. Esto generará automáticamente el archivo `xclbin` y programará su tarjeta Alveo.

```shell
cd OPT-1.3b_optimize/
make run
```

### 3. Ejecutar la Benchmark (`lambada`)
Compile y ejecute la aplicación del lado del host para ejecutar la **`lambada` benchmark**.
* **Nota**: Verifique `tokenizer_predict_eigen.cpp` para confirmar que el código carga correctamente los datos compactados.

```shell
cd tokenizer/
sh ./command.sh
```

### 4. Ejecutar la Demo Web
También puede interactuar con el LLM a través de una interfaz WebUI:

1.  Inicie el servidor de Python (requiere `python==3.6`).

2.  Abra la interfaz web en su navegador: `LLM-demo-gui/llm-gui/web/index.html`.
    (Por favor, abra el archivo HTML directamente en su navegador para chatear con el LLM.)

```shell
cd LLM-demo-gui/alveo
(python==3.6) python client-v3.py
```


## 📝 Referencias

Si encuentra Terafly o LoopLynx útil en su investigación o proyecto, cite nuestros artículos. Agradecemos su interés en nuestro trabajo.

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
