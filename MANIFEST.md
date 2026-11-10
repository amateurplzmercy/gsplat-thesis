# 3DGS 缓存实验复现清单

在 Windows x64 和支持 CUDA 的 NVIDIA GPU 上，从项目目录依次运行。

```powershell
.\reproduce.ps1 -Stage env
.\reproduce.ps1 -Stage data
.\reproduce.ps1 -Stage main
.\reproduce.ps1 -Stage analyze
.\reproduce.ps1 -Stage demo
```

第一次从 Git 检出时，给 env 和 data 分别加 `-Download` 补齐缺失资源，或按下文准备本地缓存。首次引导 Python 需要 Windows Python Launcher 的 `py -3` 和宿主 pip 22.3 以上版本，安装目标为项目内的便携 Python。已有运行时可直接进入验证。 [pip 指定目标解释器](https://pip.pypa.io/en/stable/topics/python-option/)

默认输出到 `reports/reproduction`。所有阶段可用同一个 `-OutputRoot` 指定另一轮输出目录，相对路径按项目根目录解析。完成 main 后会写 `main-complete.json`，再次测量使用新的输出根目录。给所有阶段加 `-Quick` 可以检查流程，默认写入 `reports/reproduction-quick`。快速模式只跑 Playroom 连续和原始顺序的四轮候选矩阵，其计时用于验证流程。

## 场景和数据

Train 与 Playroom 都来自 Graphdeco 的 30,000 次迭代预训练模型。下载脚本用 HTTP Range 提取每个场景的相机文件和 PLY，合计网络传输约 709 MiB。已有文件放入下列目录。data 阶段用固定清单核对 SHA256，然后按原有布局生成数据和轨迹。 [Graphdeco 官方模型合集](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)

| 场景 | 本地文件 | SHA256 |
| --- | --- | --- |
| Train | `data/train/cameras.json` | `f54e605c11aa577e3cd83702ce69385577da07eb9ae150f0f195116018506563` |
| Train | `data/train/point_cloud.ply` | `70e055d46f53636c83547a59c4998a0a27f64e42fceba45c6b78dfc77f52958d` |
| Playroom | `data/playroom/cameras.json` | `948d67f7ff0ebaa1ac2942daf58fa798a956b42dce10bb37cfc6c128991c5957` |
| Playroom | `data/playroom/point_cloud.ply` | `3627c41163a903782ff710b955ba6f1e0cab236b1ffacf5c5f912775df6624da` |

ZIP 成员路径和下载散列由 [reproduction_sources.json](configs/reproduction_sources.json) 固定。常规场景块大小为 2048 个高斯，跳转对照另用 Playroom 512。两种布局均采用 0.04、0.16、0.64 的尺度分带。渲染实验为 1280×720，GPU 场景块预算与进程显存分开。

## 依赖和测量条件

| 项目 | 固定版本或测量设置 |
| --- | --- |
| Python 与安装器 | CPython 3.10.11 Windows x64 便携版，首次引导 pip 24.3.1 |
| CUDA 渲染 | PyTorch 2.4.1+cu124，CUDA runtime 12.4，gsplat 1.5.3+pt24cu124 |
| 数据与其余依赖 | NumPy 1.26.4，packaging 24.2，setuptools 75.3.0，20 个运行依赖的完整 wheel 散列见 [requirements-lock.txt](requirements-lock.txt) |
| 参考设备 | NVIDIA RTX 4060 Laptop GPU，8 GiB 显存，驱动 592.00，Windows 11 |
| 供电与电源方案 | 交流供电，参考设备的 OEM Performance，GUID 为 `27fa6203-3987-4dcc-918d-748559d549ec`，GPU 保持动态时钟 |
| CPU 线程 | 计时进程 PyTorch、OMP 和 OpenBLAS 均为 1 线程 |
| 统计 | 四组对照 16 轮 Williams 顺序并交错预算，独立双组复验用 16 轮 AB／BA，热态排除各遍首帧，整轮配对 bootstrap 2000 次 |

CUDA wheel 与 Python ZIP 放入 `downloads/`，其余 wheel 可放在 `downloads/wheels/`。离线引导 pip 时把 `pip-24.3.1-py3-none-any.whl` 放入任一 wheel 目录。依赖通过带散列的锁文件安装到项目内。 [pip 24.3.1 的官方文件](https://pypi.org/project/pip/24.3.1/#files)

冷启动验证发现，gsplat 的 Windows 后端加载会导入 PyTorch 的扩展工具，从而需要 setuptools。锁文件补入参考环境中的 75.3.0，并按 PyPI 的 wheel 散列校验。 [setuptools 75.3.0 的官方文件](https://pypi.org/project/setuptools/75.3.0/#files)

main 阶段由 `run_controlled.ps1` 临时使用已有 OEM Performance，结束恢复原电源方案。设备没有该 GUID 时，先在 Windows 选择可用方案，再运行 `.eproduce.ps1 -Stage main -KeepPowerPlan`。复测记录 GPU 和 CPU 遥测，报告机器与供电差异。

## 阶段和产物

耗时按参考笔记本和本地缓存估计。首次下载、安装和浏览器操作另计。

| 阶段 | 内容 | 预计耗时 | 输出 |
| --- | --- | --- | --- |
| env | 引导便携环境，核对依赖，验证 CUDA 与合成渲染 | 已有环境约 15–30 秒，首次安装另计 | `tools/python310/`、`downloads/`、`reports/reproduction/env/` |
| data | 校验模型、生成场景块和三条轨迹 | 已有模型约 15 秒–2 分钟 | `data/<scene>/`、`reports/reproduction/workloads/` |
| main | 两套 Playroom 矩阵、Train 高命中对照、上传诊断、管线消融、布局对照和 RGB 审计 | 约 25–40 分钟，Quick 约 1–3 分钟 | `reports/reproduction/<实验>/` 下的逐帧 CSV、轮次 JSON、遥测和画质记录 |
| analyze | 逐帧配对、区间、慢帧清单及跳转需求诊断 | 约 1–2 分钟 | 各实验的 `paired_statistics.json`、`slow_frames.csv`，以及 `playroom-capture/` |
| demo | 生成真实参考图像与实测曲线数据 | 约 20–60 秒 | `reports/reproduction/demo/index.html` 和 316 帧素材 |

main 的实验名与历史报告对应。`candidate-matrix` 和 `train-221` 是短预测正式对照。`diagnostic`、`diagnostic-capture`、`formal`、`ablation-pipeline`、`ablation-admission`、`candidate` 对应测量协议修复。`layout-pair` 对应跳转分块对照。`short-pipeline`、`short-pipeline-stress`、`short-pipeline-confirm` 对应管线选择。

## 核对结果

优先核对可重复的输入、样本数、字节和 RGB。以下数字以完整模式及固定输入为准，字节量按 MiB 四舍五入。

| 实验 | 核对项 | 参考值 |
| --- | --- | --- |
| Playroom 候选矩阵 | 轮数与画质 | 448 遍、66,176 个热态帧，1041 个 RGB 对照帧最大绝对误差为 0 |
| Playroom 连续 326 MiB | LRU／fixed／short_only 每遍总 H→D | 242.92／595.69／245.06 MiB |
| Playroom 连续 326 MiB | short_only 每遍无效预取 | 1.04 MiB |
| Train 连续 221 MiB | LRU 与 short_only 每遍总 H→D | 均为 14.67 MiB，138 帧 RGB 最大绝对误差为 0 |
| Playroom 跳转 375 MiB 布局对照 | 2048／512 每遍需求上传 | 5799.88／5223.88 MiB |
| 原始顺序独立管线复验 | 逐块／批量每遍逻辑 D→D | 0／264.31 MiB |

时延是历史参考值。Playroom 连续 326 MiB 的 short_only 组 p99 为 **15.30 ms**，同轮 LRU 为 18.04 ms，差为 −2.75 ms［−3.59，−1.86］。Train 同轮差为 −1.59 ms［−2.35，+0.50］。原始顺序独立管线复验中，逐块减批量的组 p99 差为 +0.86 ms［−1.24，+1.91］，据此采用渲染先提交、批量上传、向量化预测。复测用自己的样本及区间报告结果。 [完整矩阵](docs/10_测量协议修复与配对复测.md) · [跨场景与跳转](docs/11_跨场景与跳转负载.md) · [管线选择](docs/12_短预测管线减法.md)

Git 跟踪代码、配置、锁文件、文档和演示页面源文件。运行时、下载、模型、实验报告和生成的演示资源保存在本地，或用上述阶段重新生成。
