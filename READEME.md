# OCR：训练与推理

## 项目介绍

光学字符识别（Optical Character Recognition, OCR）是指对文本材料的图像文件进行分析识别处理，以获取文字和版本信息的过程。也就是说将图象中的文字进行识别，并返回文本形式的内容。

### 数据集
Synthetic Chinese String Dataset

### 模型

CRNN

## 如何使用

首先输入如下命令，进入工作目录：

```
cd /workspace/
```

### 目录结构

```
.
|-- README.md
|-- ocrinfer.py             # 推理脚本
|-- requirements.txt        # python 依赖包
|-- train_pytorch_ctc.py    # 训练脚本
|-- config.py               # 参数配置
|-- mydataset.py            # 数据处理
|-- oneline_test.py         # 工具类
|-- trans_utils.py          # 图像处理工具类
|-- trans.py                # 图像处理
|-- crnn.py                 # 模型
|-- checkpoints             # 模型
|   |-- crnn.pth   # 模型


```

### 推理

本项目已经训练得出了不错的模型参数 `checkpoints/CRNN.pth`，推理程序默认使用该模型参数。

在命令行中输入以下命令，运行推理程序：

```
python ocrinfer.py
```

### 训练

在命令行中输入以下命令，开始新的训练：

```
python train_pytorch_ctc.py
```

注意，完整的训练时间预计三个小时以上。

### 参数配置

参数的配置在 `config.py` 文件中查看，参数表如下所示：

| 参数名 | 解释 |
| ------ | ------ |
| train_infofile | 训练数据的路径 |
| val_infofile | 验证数据的路径 |
| workers | worker的数量 |
| batch_size | 每个 batch 的大小 |
| niter | 在数据集上训练的轮数 |
| lr | 优化器的学习率 |
| beta1 | Adam 优化器的 betas 参数 |
| saved_model_dir | 训练脚本保存的模型参数的路径 |
| pretrained_model | 推理脚本加载的模型参数的路径 |
| saved_model_prefix | 训练脚本保存的模型参数保存的前缀名称 |

你可以直接在该文件中修改这些参数，也可以在命令行中指定，例如：

```
python train_pytorch_ctc.py --batch_size 64
```
