# Playroom 负载与预取初测

Playroom 已选作第二个真实场景。官方第 30,000 次迭代模型有 2,546,116 个高斯及 225 个原始相机，按范围提取约 499.7 MiB 并通过 ZIP CRC 校验。沿用 Train 的 1280×720、float32、3 阶球谐颜色及空间分块设置，共得到 1569 块，模型张量总载荷 573.05 MiB。模型哈希和原始成员路径见 `data/playroom/source.json`。 [Graphdeco 官方模型合集](https://github.com/graphdeco-inria/gaussian-splatting)

## 轨迹与预算

原始文件名顺序中相邻视角转角的中位数约 41°，用于急转与跳转压力对照。连续路线由 8 个实拍位姿连接而成。相邻锚点最大距离为相机集合直径的 12%，朝向差上限为 60°，路径端点相距约 6.96 场景单位。对位置做线性插值、对旋转做最短旋转插值，每步转角不超过 4°。折返路线沿相同位姿序列返回起点。

| 轨迹 | 帧数 | 逐高斯投影中位 / 峰值 | 块需求中位 / 峰值 | 30 帧内最高块并集 | 实测预算 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 连续跨区 | 91 | 12.0% / 32.6% | 22.0% / 51.2% | 436.3 MiB | 298 MiB |
| 连续路线折返 | 181 | 12.0% / 32.6% | 22.0% / 51.2% | 436.3 MiB | 298 MiB |
| 原始拍摄顺序 | 225 | 10.6% / 43.1% | 20.3% / 64.7% | 573.0 MiB | 375 MiB |

25% 和 50% 档按整数 MiB 向上取整为 144 和 287 MiB。连续路线分别有 34 和 1 帧超出预算，折返有 68 和 2 帧，原始拍摄顺序有 83 和 8 帧。298 MiB 接近连续路线的 293.58 MiB 峰值，且明显低于跨帧并集。75% 档约 430 MiB，三条轨迹均通过 CPU 容量筛选。折返的 LRU 热态 p95 上传量在模拟中从 298 MiB 档的 14.13 MiB 降为 430 MiB 档的 7.95 MiB，GPU 对照采用紧预算。预算筛选数据在 `reports/workloads/playroom/summary.json`。

## 实测方法

各策略使用同一条轨迹、缓存预算和每帧最多 16 MiB 的预取配额。预取只看当前和历史位姿。固定版按恒速外推未来 4 帧。自适应初版依据平移和旋转的一致性，在 0、2、6 帧之间切换。保守版在 0、1、2 帧之间切换。两帧预测时仅保留两个预测视锥的共同候选，准入只用缓存空余空间。两种方向预测与保守版均使用固定页主存和独立 CUDA 流提交真实上传。

每种策略预热 60 帧后测 3 遍。每遍重置缓存，帧结束等待渲染和预取流完成。尾延迟和热态按需传输统计排除各遍第一帧。无效预取量包括预取后未用于任何帧、随后淘汰或在回放结束时仍未使用的数据。CUDA 事件窗口交叠值保存在逐帧日志中。

## 预取结果

| 轨迹 | 策略 | 热态按需上传（MiB） | 无效预取（MiB） | 初测热态 p99（ms） |
| --- | --- | ---: | ---: | ---: |
| 连续跨区 | LRU | 750.0 | 0.0 | 17.66 |
| 连续跨区 | 固定方向 | 375.8 | 1111.6 | 21.34 |
| 连续跨区 | 自适应初版 | 384.1 | 1543.4 | 21.03 |
| 连续跨区 | 保守版 | 735.8 | 1.0 | 17.56 |
| 折返 | LRU | 1506.5 | 0.0 | 20.02 |
| 折返 | 固定方向 | 605.7 | 1657.8 | 50.19 |
| 折返 | 自适应初版 | 652.2 | 2319.5 | 71.22 |
| 折返 | 保守版 | 1491.3 | 1.8 | 18.51 |
| 原始拍摄顺序 | LRU | 17399.7 | 0.0 | 49.92 |
| 原始拍摄顺序 | 固定方向 | 15645.2 | 5532.9 | 118.31 |
| 原始拍摄顺序 | 自适应初版 | 17059.1 | 507.2 | 71.91 |
| 原始拍摄顺序 | 保守版 | 17350.8 | 34.1 | 38.24 |

初测存在运行顺序和系统波动。针对初测中看似胜过 LRU 的折返与原始拍摄顺序，各补做了 4 对交错复测，轮流先运行 LRU 和保守版。折返的每遍 p99 中位数分别为 18.56 和 19.23 ms，原始顺序分别为 40.93 和 45.82 ms。原始顺序的一对复测两组都出现 p99 超过 85 ms 的波动。目前结果支持保守版减少相对固定方向的无效流量，尚无相对普通 LRU 的稳定尾延迟收益。Train 高命中率对照中，初版预取也减少了按需 miss，尾延迟收益同样未出现。

下一阶段建议先减少小块上传的提交开销，再做两项独立消融，区分短预测范围和空余容量准入分别产生的效果。Playroom 的单帧投影比例低于块需求比例，也可评估细块对传输字节和调用开销的取舍。

## 画质与设备

Playroom 全量基线在 RTX 4060 Laptop 8GB 上测完 675 帧，p95 为 11.87 ms，PyTorch 峰值已分配约 973 MiB。各预取计时组的最高 PyTorch 峰值已分配约 1.12 GiB。当前设备完成了本轮全部验证。

连续路线的 91 帧和原始顺序的 225 帧均通过逐帧投影 ID 核对。按相同高斯顺序对齐后，分块和全量参考的 RGB 逐像素相等。三种预取策略在每条轨迹抽取的 9 帧上与普通 LRU 的裁剪后浮点 RGB 完全一致。各策略的逐帧需求与投影计数、块预算和预取字节账本也已核对。

## 复现与文件

在项目目录运行。分块构建使用新的输出目录。

```powershell
python scripts/fetch_graphdeco_scene.py --scene playroom --max-mib 650
.\run_python.ps1 -m scripts.render_baseline --scene data/playroom --output reports/baseline/playroom-1280x720
.\run_python.ps1 -m scripts.spatial_blocks --scene data/playroom --output data/playroom/blocks-2048-banded --leaf-size 2048 --scale-bands 0.04 0.16 0.64
.\run_python.ps1 -m scripts.prepare_workloads --scene data/playroom --blocks data/playroom/blocks-2048-banded --output reports/workloads/playroom
$trials = @(@{Track='continuous';Budget=298}, @{Track='return';Budget=298}, @{Track='capture_order';Budget=375})
foreach ($trial in $trials) {
    $track = $trial.Track
    foreach ($policy in 'lru','fixed','adaptive','guarded') {
        .\run_python.ps1 -m scripts.compare_prefetch --blocks data/playroom/blocks-2048-banded --trajectory "reports/workloads/playroom/$track.json" --budget-mib $trial.Budget --quota-mib 16 --policy $policy --output "reports/prefetch/playroom-$track-$policy"
    }
}
```

初测按策略和轨迹保存到 `reports/prefetch`，每组有 `summary.json` 和 `frames.csv`。交错复测位于 `paired_recheck.json`，采样画质核对位于 `rgb_verification.json`。
