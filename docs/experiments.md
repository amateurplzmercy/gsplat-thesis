# 实验方法与结果

## 测量口径

在 RTX 4060 Laptop GPU 上，以 1280×720 回放固定相机轨迹。Playroom 连续和折返各测 298、326、430 MiB，包含视点跳转的原始拍摄顺序测 375 MiB。Train 连续轨迹以 221 MiB 检查高命中负载。四组对照包括按需 LRU、固定方向预取（`fixed`）、短期预取（`short_only`）和仅用剩余空间的短期预取（`guarded`）。

每组测 16 遍，均衡策略执行位置并交错预算，排除每遍首帧。p99 是把同一条件下的热态帧合并得到的第 99 百分位。组间差使用整轮配对 bootstrap 的 95% 区间。总 H→D 是每遍热态的需求加预取上传。无效预取指被淘汰或回放结束前从未被使用的预取字节。慢帧的阈值为 33.3 ms。

## 主要结果

**Playroom 连续轨迹、326 MiB** 的四策略对照如下。同布局下，`short_only` 与 LRU 在 Playroom 全部七组轨迹和预算中核对的 1041 帧 RGB 逐像素一致。

| 策略 | p99（ms） | 每遍总 H→D（MiB） | 每遍无效预取（MiB） |
| --- | ---: | ---: | ---: |
| LRU | 18.04 | 242.92 | 0 |
| fixed | 15.62 | 595.69 | 350.50 |
| short_only | 15.30 | 245.06 | 1.04 |
| guarded | 16.98 | 242.92 | 0 |

在连续轨迹的三档预算下，`short_only` 相对 LRU 的 p99 均降低，总上传量仅增加 0.32%–4.85%。折返时，326 MiB 同时满足 p99 降低及流量增加不超过 5%，298 MiB 的流量增加超过 5%，430 MiB 的 p99 差区间跨零。

| Playroom 轨迹与预算 | short_only 减 LRU 的 p99 差（ms，95% 区间） | 总 H→D 变化 |
| --- | ---: | ---: |
| 连续 298 MiB | −2.19［−2.71，−1.62］ | +4.85% |
| 连续 326 MiB | −2.74［−3.59，−1.86］ | +0.88% |
| 连续 430 MiB | −3.26［−4.00，−2.46］ | +0.32% |
| 折返 298 MiB | −2.77［−4.47，−2.00］ | +7.17% |
| 折返 326 MiB | −2.76［−3.39，−2.09］ | +3.98% |
| 折返 430 MiB | −2.41［−5.02，+3.89］ | +2.14% |
| 原始顺序 375 MiB | +0.75［−1.83，+6.34］ | +2.86% |

原始顺序中，`short_only` 比 `fixed` 少传 26.24%。相比 LRU，慢帧率差为 +0.39 个百分点［−0.20，+1.14］，仍不足以确认满足 +0.5 个百分点的容忍线。Train 的 LRU 字节命中率约为 99.94%，两种策略总 H→D 都是 14.67 MiB，p99 差为 −1.59 ms［−2.35，+0.50］，尾延迟收益尚不确定。

## 工程取舍

- **先提交渲染。** 在 Playroom 连续 298 MiB 的固定方向预取实验中，仅调整提交顺序，CUPTI 观察到实际拷贝与渲染内核重叠的帧从 0/70 增至 67/70。连续 326 MiB 的固定方向预取消融中，同一改动的组 p99 差为 −4.24 ms［−7.13，−2.87］。
- **采用批量上传与向量化预测。** 原始顺序 375 MiB 的独立 A/B 复验固定向量化和短期预取，两种上传路径的 H→D 相同。逐块减批量的 p99 差为 +0.86 ms［−1.24，+1.91］，上界超过预设的 +1 ms 容忍线。批量路径每遍多 264.31 MiB 的逻辑 D→D，慢帧为 56/3584，逐块为 72/3584。
- **保留 2048 的标准分块布局。** 原始顺序 375 MiB 的 LRU 对照中，将块大小从 2048 改为 512 后，每遍需求上传少约 10%，每帧上传块数中位数从 45 增到 159，p99 从 36.31 ms 升到 70.05 ms。现有逐块需求上传和拼接的开销需要与字节节省一起衡量。

## 复现和核对

按复现清单运行后，`reports/reproduction/candidate-matrix/` 存放主矩阵，`train-221/` 存放 Train 高命中对照，`layout-pair/` 与 `short-pipeline-confirm/` 分别存放布局对照和独立 A/B 复验。每组的 `frames.csv`、`runs.json` 和 `paired_statistics.json` 可用来核对逐帧数据、运行顺序及配对结果。时延区间对应这台测试机上的固定轨迹。 [复现清单](../MANIFEST.md)

CUPTI 设备跟踪可以在数据准备后单独回放。两条命令只改变提交顺序，结果目录中的 `device_trace.json` 记录 GPU 实际活动。

```powershell
$trace = @('-m', 'scripts.compare_prefetch',
  '--blocks', 'data/playroom/blocks-2048-banded',
  '--trajectory', 'reports/reproduction/workloads/playroom/continuous.json',
  '--budget-mib', '298', '--policy', 'fixed', '--device-trace')
.\run_python.ps1 @trace --launch-order prefetch-first --output reports/profiling/prefetch-first
.\run_python.ps1 @trace --launch-order render-first --output reports/profiling/render-first
```
