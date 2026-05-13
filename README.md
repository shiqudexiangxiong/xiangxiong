# PrivPGD:基于粒子梯度下降与最优传输的隐私保护表格数据合成方法

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2401.17823-B31B1B.svg)](https://arxiv.org/abs/2401.17823)
[![Python 3.11.5](https://img.shields.io/badge/python-3.11.5-blue.svg)](https://python.org/downloads/release/python-3115/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pytorch 2.1.2](https://img.shields.io/badge/pytorch-2.1.2-green.svg)](https://pytorch.org/)

本仓库包含 **PrivPGD** 的 Python 实现。PrivPGD 是一种基于边缘分布(marginal-based)的隐私数据合成生成方法,在 **ICML 2024 论文** [《Privacy-preserving data release leveraging optimal transport and particle gradient descent》](https://arxiv.org/abs/2401.17823) 中提出。

## 目录

* [项目概述](#项目概述)
* [代码结构](#代码结构)
* [快速开始](#快速开始)
* [示例与教程](#示例与教程)
* [新增功能模块](#新增功能模块)
* [参与贡献](#参与贡献)
* [联系方式](#联系方式)
* [引用](#引用)

## 项目概述

敏感数据集的发布在医疗、政务等数据驱动决策的诸多领域中扮演着关键角色,但发布这类数据往往会引发严重的隐私问题。差分隐私(Differential Privacy,DP)作为一种有效的范式,在日益数据化的世界中提供了切实可行的隐私保护方案。

PrivPGD 是一种用于差分隐私表格数据合成的新方法。它能够基于受保护数据集边缘分布的带噪测量值,生成高质量的私有数据副本。PrivPGD 将粒子梯度下降与基于最优传输的散度相结合,使得在数据集生成过程中可以高效整合边缘分布信息。

PrivPGD 的主要优势包括:

- **业界领先的性能**:在多项基准测试与下游任务中表现优异。
- **强大的可扩展性**:针对现代 GPU 优化了梯度计算,支持并行化处理,在面对大规模数据集与众多边缘分布时尤为高效。
- **几何结构保持**:保留数据集特征的几何结构(例如排序关系),与真实数据的细微特性更相契合。
- **支持领域约束**:允许在合成数据中加入额外的领域特定约束条件。

## 代码结构

`src` 文件夹包含本工具包的核心代码,按功能划分为多个子目录。本项目代码大量借鉴了 [PGM 仓库](https://github.com/ryan112358/private-pgm) 的实现思路。

### 1. 机制模块(`src/mechanisms`):
   - 负责边缘分布的选择与隐私化处理。
   - 主要文件及对应机制:
     - `kway.py`:实现 K-Way 机制。
     - `mwem.py`:实现 [MWEM](https://arxiv.org/pdf/1901.09136.pdf) 机制。
     - `mst.py`:实现 [MST](https://arxiv.org/pdf/2108.04978.pdf) 机制。
     - `aim.py`:实现 [AIM](https://arxiv.org/pdf/2201.12677.pdf) 机制。
   - 此文件夹还包含其他辅助这些机制的工具文件。

### 2. 生成方法模块(`src/inference`):
   - 包含数据生成方法的代码实现。
   - 子目录及对应方法:
     - `pgm`:[PGM](https://arxiv.org/pdf/1901.09136.pdf) 方法的实现。
     - `privpgd`:PrivPGD 方法的实现,即本项目提出的差分隐私数据生成新方法。

## 快速开始

### 依赖环境

- Python 3.11.5
- Numpy 1.26.2
- Scipy 1.11.4
- Scikit-learn 1.2.2
- Pandas 2.1.4
- Torch 2.1.2
- CVXPY 1.4.1
- Disjoint Set 0.7.4
- Networkx 3.1
- Autodp 0.2.3.1
- POT 0.9.1
- Folktables 0.0.12
- Openml 0.14.1
- Seaborn 0.13.0

### 安装步骤

请按以下步骤搭建环境并安装本工具包:

#### 创建并激活 Conda 环境

首先创建一个使用 Python 3.11.5 的 Conda 环境,以确保运行环境的 Python 版本正确。

```bash
conda create -n privpgd python=3.11.5
conda activate privpgd
```

#### 安装工具包

提供两种安装方式:

1. **本地安装:**
   先从 GitHub 克隆仓库,然后升级 `pip` 至最新版本,再通过本地 setup 文件安装包。此方式适合开发场景或已持有源代码的用户。

   ```bash
   git clone https://github.com/jaabmar/private-pgd.git
   cd private-pgd
   pip install --upgrade pip
   pip install -e .
   ```

2. **从 GitHub 直接安装(推荐):**
   也可直接通过 GitHub 安装,该方式简单快捷,可确保获取最新版本。

   ```bash
   pip install git+https://github.com/jaabmar/private-pgd.git
   ```

## 示例与教程

在 `examples` 文件夹中提供了若干实践示例,帮助你理解如何使用本工具包中包含的不同机制与方法。

### 数据准备

运行脚本之前,需要先下载 ACS 数据集:

1. 在本地下载仓库后,进入 `data` 目录:

   ```bash
   cd path/to/private-pgd/data
   ```

2. 运行 `create_data.py` 脚本下载数据集:

   ```bash
   python create_data.py
   ```

   数据集将下载并存储在 `data` 目录下的 `datasets` 文件夹中。

### 核心实验脚本

1. **`experiment.py`**:通用的实验运行脚本,可灵活适配多种实验配置。

2. **`mst+pgm.py`**:运行 PGM 生成方法 + MST 边缘选择机制的实验脚本。

3. **`aim+pgm.py`**:运行 PGM 生成方法 + AIM 边缘选择机制的实验脚本。

4. **`privpgd.py`**:运行 PrivPGD 方法的专用脚本,即本项目提出的差分隐私数据合成新方法。

5. **`privpgd_with_constraint.py`**:演示如何向 PrivPGD 中加入领域特定约束的脚本。

### 运行实验

实验通过命令行调用脚本运行,命令处理由 Click(8.1.7 版本)支持。以使用默认超参数和论文中描述的设置,在 ACS Income California 2018 数据集上运行 PrivPGD 为例:

1. 切换(`cd`)到 `examples` 文件夹。
2. 执行以下命令:

   ```bash
   python privpgd.py
   ```

该命令将使用指定数据集和默认设置启动 PrivPGD 实验。

### 分步教程

如需详细了解 PrivPGD 的逐步使用方法,请参阅 [Tutorial](examples/Tutorial.ipynb) Jupyter notebook。该 notebook 包含完整的解释与可视化展示,带你完整走通使用 PrivPGD 进行差分隐私数据合成的全过程。

## 新增功能模块

为了让 PrivPGD 的使用更加便捷,我们在 `examples` 文件夹中新增了三个功能模块,分别负责模型定义与参数配置、主流程运行以及生成数据的还原。三者协同构成了一条完整的"训练 → 运行 → 还原"工作流。

### 1. `examples/privpgd_new.py` —— 核心功能模块

该文件是新增功能的核心,封装了 PrivPGD 模型的启动逻辑与详细参数配置,所有与模型相关的功能均集中在此实现。其主要职责包括:

- **模型初始化**:完成 PrivPGD 模型的实例化与启动配置。
- **参数管理**:提供完整的超参数定义与默认值设置(如隐私预算 ε、迭代轮次、学习率、粒子数量、边缘分布相关配置等)。
- **训练与生成流程**:封装数据加载、边缘分布测量、粒子梯度下降优化、合成数据生成等关键步骤。
- **接口暴露**:对外提供清晰的函数接口,供 `main.py` 等上层脚本调用。

如需自定义模型行为或调整超参数,可直接修改此文件中的相关配置。

### 2. `examples/main.py` —— 主运行入口

该文件是整个工作流的运行入口,通过调用 `privpgd_new.py` 中的核心功能,完成端到端的数据合成过程。其主要功能包括:

- **流程调度**:按顺序调用 `privpgd_new.py` 中的核心运行函数,串起完整流程。
- **运行环境配置**:负责加载配置项、初始化日志、设置随机种子等准备工作。
- **结果输出**:将生成的合成表格数据保存至指定路径,以供后续使用或还原。

运行示例:

```bash
cd examples
python main.py
```

执行后,程序会自动调用 `privpgd_new.py` 中的核心逻辑,完成模型训练与合成数据生成。

### 3. `examples/recover.py` —— 数据还原模块

该文件负责将 PrivPGD 生成的合成表格数据还原为原始数据格式。由于 PrivPGD 在训练前会对数据进行编码与离散化处理,直接生成的合成数据可能并非原始字段格式,需要通过本模块进行反向映射。其主要功能包括:

- **格式还原**:将合成数据中的离散编码、归一化数值等还原为原始字段含义(如类别名称、连续数值范围等)。
- **结构对齐**:保证还原后数据的列名、字段顺序与原始数据集一致。
- **结果导出**:将还原后的数据导出为常见的表格格式(如 CSV),便于后续分析与使用。

运行示例:

```bash
cd examples
python recover.py
```

运行后即可获得与原始数据集结构对齐、可直接用于下游任务的合成数据。

### 推荐工作流

完整的使用流程如下:

1. 配置 `privpgd_new.py` 中的模型与超参数。
2. 运行 `main.py` 进行模型训练与合成数据生成。
3. 运行 `recover.py` 将合成数据还原为原始格式。

## 参与贡献

我们欢迎社区为本项目贡献力量,贡献方式如下:

1. Fork 本项目
2. 创建你的特性分支(`git checkout -b feature/AmazingFeature`)
3. 提交你的修改(`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支(`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 联系方式

如有任何问题,欢迎通过以下方式联系我们:

- Javier Abad Martinez - [javier.abadmartinez@ai.ethz.ch](mailto:javier.abadmartinez@ai.ethz.ch)
- Konstantin Donhauser - [konstantin.donhauser@ai.ethz.ch](mailto:konstantin.donhauser@ai.ethz.ch)
- Neha Hulkund - [nhulkund@mit.edu](mailto:nhulkund@mit.edu)

## 引用

如果本代码对你的工作有所帮助,欢迎引用我们的论文:

```
@article{donhauser2024privacy,
  title={Privacy-Preserving Data Release Leveraging Optimal Transport and Particle Gradient Descent},
  author={Donhauser, Konstantin and Abad, Javier and Hulkund, Neha and Yang, Fanny},
  journal={International Conference on Machine Learning},
  year={2024}
}
```

## 持续开发中

我们正在持续开发新的功能与改进:

- **评估流水线**:引入更多评估指标(如协方差矩阵差异及其他高阶查询),并对下游任务表现进行评估。
- **更多算法基准对比**:正在集成 Private GSD、RAP、GEM 等算法,以提供更全面的基准对比。

以上更新是我们持续完善框架、打造差分隐私数据合成领域稳健基准工具的一部分努力。
