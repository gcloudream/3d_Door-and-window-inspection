# Point-SAM Linux/GPU 实机部署说明

这份说明用于把当前项目切到真实 Point-SAM 后端。  
适用目标机器：

- Linux
- NVIDIA GPU
- 已正确安装 CUDA 驱动
- 计划运行官方 Point-SAM 推理代码

## 当前工程的接入方式

当前项目已经支持两种后端：

- `heuristic_region_growing`
- `point_sam`

运行时通过环境变量选择：

```bash
export POINT_SEGMENTER_BACKEND=point_sam
```

如果 Point-SAM 依赖或 checkpoint 不可用，服务会自动回退到启发式后端，不会直接崩掉。

## 官方依据

本地接入方式基于 Point-SAM 官方仓库和项目页：

- 项目页：<https://point-sam.github.io/>
- 官方仓库：<https://github.com/zyc00/Point-SAM>

官方 README 中给出的关键点：

- 需要 `python>=3.8`
- 需要 `pytorch>=2.1.0`
- 需要 `torchvision>=0.16.0`
- 需要 `timm>=0.9.0`
- 官方建议安装第三方模块：
  - `third_party/torkit3d`
  - `third_party/apex`
- 官方说明可通过 `evaluation/inference.py` 和 `demo/app.py` 对自有点云与 prompt points 做推理

## 推荐步骤

### 1. 在 Linux/GPU 机器上准备 PyTorch

先安装 GPU 版 PyTorch。  
这里不要硬编码 wheel 命令，直接使用官方安装选择器：

<https://pytorch.org/get-started/locally/>

要求：

- PyTorch 版本满足官方 README 中的 `>=2.1.0`
- `torch.cuda.is_available()` 返回 `True`

### 2. 运行环境初始化脚本

项目里已经提供：

```bash
bash /Users/gengchen/Desktop/3d/scripts/setup_point_sam_linux.sh
```

这个脚本会：

- 创建专用 virtualenv
- clone / update 官方 Point-SAM 仓库
- 安装当前项目和 Point-SAM 所需的 Python 依赖
- 初始化并安装官方 README 中提到的 `torkit3d` / `apex`
- 给出下一步运行命令

默认路径：

- `POINT_SAM_VENV_DIR=/Users/gengchen/Desktop/3d/.venv-point-sam`
- `POINT_SAM_REPO_DIR=/Users/gengchen/Desktop/3d/.vendor/Point-SAM`

你也可以在执行前改环境变量：

```bash
export POINT_SAM_VENV_DIR=/data/envs/point-sam
export POINT_SAM_REPO_DIR=/data/models/Point-SAM
bash /Users/gengchen/Desktop/3d/scripts/setup_point_sam_linux.sh
```

### 3. 下载官方 checkpoint

根据官方 README，需要提供本地 checkpoint 路径。  
当前项目运行时需要以下变量：

```bash
export POINT_SEGMENTER_BACKEND=point_sam
export POINT_SAM_REPO_DIR=/absolute/path/to/Point-SAM
export POINT_SAM_CHECKPOINT=/absolute/path/to/model.safetensors
export POINT_SAM_CONFIG_NAME=large
export POINT_SAM_DEVICE=cuda
```

其中最关键的是：

- `POINT_SAM_REPO_DIR`
- `POINT_SAM_CHECKPOINT`
- `POINT_SEGMENTER_BACKEND`

## 运行前自检

项目里提供了一个轻量自检脚本：

```bash
/absolute/path/to/venv/bin/python /Users/gengchen/Desktop/3d/scripts/point_sam_doctor.py
```

输出会告诉你：

- 当前请求的后端
- 当前实际激活的后端
- 如果回退，回退原因是什么

理想情况下应看到：

```json
{
  "segmentation": {
    "requested_backend": "point_sam",
    "active_backend": "point_sam",
    "fallback_reason": ""
  }
}
```

如果看到：

- `active_backend = heuristic_region_growing`

说明真实 Point-SAM 还没成功接管，需要先解决依赖、repo 路径或 checkpoint 路径问题。

## 启动命令

项目里已经提供运行脚本：

```bash
POINT_SAM_VENV_DIR=/absolute/path/to/venv \
POINT_SAM_REPO_DIR=/absolute/path/to/Point-SAM \
POINT_SAM_CHECKPOINT=/absolute/path/to/model.safetensors \
bash /Users/gengchen/Desktop/3d/scripts/run_demo_point_sam.sh
```

这个脚本会自动：

- 激活 Point-SAM virtualenv
- 把 `POINT_SAM_REPO_DIR` 注入 `PYTHONPATH`
- 设置 `POINT_SEGMENTER_BACKEND=point_sam`
- 启动当前项目的 Web demo

默认启动地址：

- <http://127.0.0.1:8000>

## 常见失败点

### 1. `torch.cuda.is_available() == False`

说明 GPU PyTorch 没装对，或者 CUDA 驱动不可用。  
先回到 PyTorch 官方安装流程检查。

### 2. `POINT_SAM_CHECKPOINT` 路径错误

当前项目会自动回退到启发式后端，并在 `point_sam_doctor.py` 或 `/api/scene` 的 `segmentation.fallback_reason` 里给出原因。

### 3. `pc_sam` 无法 import

通常是：

- 官方 repo 没有 clone 完整
- `PYTHONPATH` 没包含 `POINT_SAM_REPO_DIR`
- 依赖没装齐

### 4. `torkit3d` / `apex` 编译失败

这是 Point-SAM 官方安装链路里最容易卡住的地方。  
当前脚本已经尽量按官方 README 走，但如果机器编译器、CUDA Toolkit、PyTorch ABI 不匹配，仍然需要在目标机单独排查。

## 和当前项目的关系

当前项目不会直接绑定 Point-SAM demo server。  
我们做的是更轻的集成方式：

- 当前 Web 前端保持不变
- 当前 Python API 保持不变
- 只在分割适配层里切换 `point_sam` / `heuristic`

所以后续如果要替换成别的 3D promptable segmentation 模型，也不需要重写前端点击与 ROI 流程。
