# 分块加载与 LRU 基线

Train 场景已完成离线分块、视锥选块、CPU 到 GPU 的按需上传，以及按字节限制的 LRU 缓存。实验使用原基线保存的 301 个相机位姿、1280×720 分辨率和同一套 gsplat 1.5.3 光栅化参数。

## 块布局与预算

将最大轴尺度按 0.04、0.16、0.64 个场景单位分组，再沿各组高斯中心跨度最大的轴做递归中位切分。每块最多 2048 个高斯，共 674 块。各块独立打包 float32 属性，原始高斯 ID 保留在索引中。模型载荷合计 231.03 MiB。加载器将块数据读入主存，运行时按块上传，计时中的传输发生在主存和显存之间。

以最大轴尺度的 4.4 倍扩张高斯中心边界，并在居中的针孔相机投影边缘留出 4 像素。扩张量覆盖固定后端的雅可比截断、投影半径上限、最小模糊项和像素取整。视锥半空间测试以整个块的扩张包围盒为单位。 [投影雅可比](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/include/Utils.cuh) · [打包投影实现](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu)

初试使用最多 4096 个高斯的纯空间块，最重一帧会选中全场景。分尺度与细分后，单帧需求从 98.09 MiB 到 215.82 MiB，中位数为 187.13 MiB。用原规划 25%、50%、75% 的载荷比例做预算时，分别有 301、291、189 帧超出容量。因此本轮选择 218 MiB 和 228 MiB。两者仅约束真实驻留的块张量。逐帧打包副本、渲染中间张量和 PyTorch 分配器预留另行记录。

LRU 按帧处理请求，先保护当前帧全部必需块，再淘汰最近最少使用的非必需块。准备阶段校验单帧需求能否放入预算。已实测 173 MiB 会在回放前因 189 帧不可行而退出。每帧使用同一次光栅化完成全局排序与混合。

## 画质核对

对 301 帧逐帧比对全量参考的投影高斯 ID。结果为零遗漏、零额外投影。与原始输入顺序的全量参考相比，最低 PSNR 为 71.66 dB，最大单通道绝对误差为 0.0962。将全量参考重排为与分块一致的高斯顺序后，301 帧的 RGB 值在裁剪到 [0,1] 后全部逐像素相等，确认前述差异来自输入排列。

`reports/streaming/quality-218/worst_pair.png` 并排展示原顺序参考与分块结果中均方误差最大的一帧。逐帧误差和投影集合核对结果位于同目录的 `frames.csv`。

## 性能对照

按需与 LRU 均预热 60 帧，再各回放 3 遍、共计时 903 帧。每遍从空块缓存开始，按需策略在每帧结束后清空已上传的块。帧时间包含选块、上传、输入打包、光栅化和完成等待。画质审计单独执行。全量数据来自前一阶段的同轨迹基线，其模型一次上传后跨三遍保持驻留。

| 策略 | 块预算 | 平均帧时间 | p95 帧时间 | 模型上传量 | 字节命中率 | PyTorch 峰值已分配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全量常驻 | 全量 | 8.14 ms | 10.53 ms | 231.03 MiB，一次 | — | 539.4 MiB |
| 按需加载 | 218 MiB | 49.99 ms | 76.82 ms | 157.44 GiB | 0% | 739.1 MiB |
| LRU | 218 MiB | 10.54 ms | 13.08 ms | 0.874 GiB | 99.44% | 741.7 MiB |
| LRU | 228 MiB | 11.60 ms | 15.16 ms | 0.793 GiB | 99.50% | 745.4 MiB |

LRU 在两个预算下分别实际淘汰 738 和 393 次块，驻留峰值分别为 217.99 和 227.99 MiB。两组都只有每遍的首帧超过 33.3 ms。228 MiB 的上传量略低，当前三遍初测的延迟差异仍需交错重复验证。

两组 LRU 都保留近乎全部活跃块。当前代码每帧还把选中块拼成连续输入，218 MiB 组的平均打包窗口约 2.22 ms，PyTorch 峰值已分配高于全量基线。后续应将常驻块预算和整个渲染进程的峰值显存分别报告。当前场景已验证工作流，后续预取需要用存在更多缓存失效的轨迹或场景测量及时命中、无效传输及尾部延迟。

## 复现

在项目目录执行。分块构建命令使用一个新的输出目录。

```powershell
.\run_python.ps1 -m scripts.spatial_blocks --output data/train/blocks-2048-banded --leaf-size 2048 --scale-bands 0.04 0.16 0.64
$common = @('-m', 'scripts.stream_blocks', '--blocks', 'data/train/blocks-2048-banded', '--trajectory', 'reports/baseline/train-1280x720/trajectory.json')
.\run_python.ps1 @common --policy audit --budget-mib 218 --output reports/streaming/quality-218
.\run_python.ps1 @common --policy demand --budget-mib 218 --output reports/streaming/demand-218
.\run_python.ps1 @common --policy lru --budget-mib 218 --output reports/streaming/lru-218
.\run_python.ps1 @common --policy lru --budget-mib 228 --output reports/streaming/lru-228
```

各策略的 `summary.json` 保存聚合结果，`frames.csv` 保存逐帧时间、需求、命中、上传、淘汰和驻留字节。图像对照在独立审计中生成。
