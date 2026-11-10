# GPU 重叠与上传剖析

Playroom 连续路线、298 MiB 块缓存、16 MiB 预取配额。先对原固定方向预取补计时，再单独交换提交顺序，随后加入批量上传与向量化预测。预取策略和需求块选择保持一致。

## 测量口径

- CPU 时钟分别记录选块、需求提交、拼接提交、预测、预取提交、固定页内存拷贝、渲染提交和两路等待。
- CUDA event 分别包围需求、拼接、预取、渲染阶段。拼接和渲染在同一计算流，预取在独立拷贝流，沿用帧末的两路完成等待。
- 实际重叠采用已安装 PyTorch 附带的 CUPTI 12.4 Activity API。捕获拷贝流上的 H→D 拷贝，再将渲染调用期间发起的 API 与计算流上的 kernel 按 correlation ID 对齐。逐帧求设备拷贝区间和 kernel 区间的交集，并核对拷贝字节数。该采集在计时完成后独立回放。四组采集均无丢失记录或字节差异。 [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)

事件窗口包含命令之间的空隙。报告的真实重叠使用 CUPTI 中实际活动的时间戳。常规计时输出是 `frames.csv`，设备活动核对是 `device_trace.json`。

## 分步结果

| 固定方向预取配置 | CUPTI H→D 次数 | 有真实重叠的上传帧 | 拷贝与 kernel 的交集 / 拷贝总时间 | 初测热态 p99 |
| --- | ---: | ---: | ---: | ---: |
| 原顺序，先预取 | 1394 | 0 / 70 | 0.00 / 53.68 ms | 21.89 ms |
| 先提交渲染 | 1394 | 67 / 70 | 45.72 / 55.27 ms | 15.70 ms |
| 先渲染，批量上传 | 70 | 61 / 70 | 38.05 / 45.03 ms | 16.04 ms |
| 先渲染，批量上传，向量化预测 | 70 | 65 / 70 | 43.70 / 50.43 ms | 16.28 ms |

原顺序在 210 个有预取的计时帧中，有 196 帧在渲染提交之前已经完成预取。先提交渲染后，CPU 预测和提交预取可以在 GPU 已收到当前帧渲染工作时进行。各组 CUPTI 逐帧上传字节完全相同，每次采集合计 538,828,828 字节。

批量路径先把候选块拼进固定页主存，一次 H→D 上传到 16 MiB 的 GPU 暂存区，再在设备上拆成独立缓存分配，以保持逐块淘汰。暂存区单列于显存统计，缓存字节预算仍限制驻留块。批量前后的 CPU 预取提交平均值为 2.84 和 2.29 ms，按有上传的帧统计。向量化把预测视锥合并做矩阵运算，批量配置的预测平均值从 0.58 到 0.56 ms。单独比较批量前后及向量化前后的初测 p99，未看到稳定的额外下降。

## 交错复测

在每条轨迹上轮换三种配置的执行顺序，各测三遍。下表是每遍热态 p99 的中位数，排除每遍首帧。

| 轨迹 | LRU | 先渲染，逐块 | 先渲染，批量及向量化 |
| --- | ---: | ---: | ---: |
| 连续 | 16.92 ms | 16.73 ms | 15.33 ms |
| 折返 | 18.70 ms | 17.93 ms | 16.20 ms |

完整优化版在两条轨迹的三轮复测中均低于同轮 LRU。连续和折返路线每遍热态按需上传量分别从 LRU 的 250.00 和 502.18 MiB 降到 125.27 和 201.91 MiB。固定方向预取仍带来额外投机传输，因此总传输量需要与按需量一起考察。

标量与向量化预测在三条 Playroom 轨迹、三种预取策略共 1491 个测试用例中，预测步数与候选块顺序均一致。连续与折返路线对普通 LRU、渲染先提交、完整优化版逐帧比对了需求字节和投影计数，并各抽 9 帧比对裁剪后浮点 RGB，均一致。

## 复现

```powershell
$common = @('-m','scripts.compare_prefetch','--blocks','data/playroom/blocks-2048-banded','--trajectory','reports/workloads/playroom/continuous.json','--budget-mib','298','--policy','fixed','--device-trace')
.\run_python.ps1 @common --launch-order prefetch-first --output reports/profiling/original-fixed
.\run_python.ps1 @common --launch-order render-first --output reports/profiling/render-first-fixed
.\run_python.ps1 @common --launch-order render-first --batch-upload --output reports/profiling/batch-fixed
.\run_python.ps1 @common --launch-order render-first --batch-upload --vectorized --output reports/profiling/vector-batch-fixed
```

交错复测和画质检查保存在 `reports/profiling/paired.json` 与 `reports/profiling/verification.json`。
