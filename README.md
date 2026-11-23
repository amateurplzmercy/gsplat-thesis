# 3DGS 场景加载与缓存

复现实验从 [MANIFEST.md](MANIFEST.md) 开始。`reproduce.ps1` 按 env、data、main、analyze、demo 五阶段执行。

已完成 Playroom 短预测版七格配对矩阵、Train 高命中对照、跳转工作集诊断和离线回放。连续路线三档预算下，short_only 比 LRU 的组 p99 低 2.19–3.26 ms，总 H→D 多 0.32%–4.85%。原始拍摄顺序相对 fixed 的传输量减少 26.24%，相对 LRU 的慢帧率差仍有不确定性。

题目：**面向3DGS交互式渲染的场景数据加载与缓存优化研究**。

[真实场景与基线](docs/05_真实场景与基线.md) 记录了场景筛选、完整复现命令、统计口径和结果。原始结果在 `reports/baseline/train-1280x720`。

[分块加载与 LRU 基线](docs/06_分块加载与LRU基线.md) 记录了块布局、容量校验、逐帧画质审计和性能结果。逐帧数据在 `reports/streaming`。

[Playroom 负载与预取初测](docs/07_Playroom负载与预取初测.md) 记录了第二场景、三条轨迹、缓存预算、策略对照与交错复测。数据在 `reports/workloads/playroom` 和 `reports/prefetch`。

[GPU 重叠与上传剖析](docs/08_GPU重叠与上传剖析.md) 记录了分阶段插桩、CUPTI 设备活动、提交顺序对照、批量上传与向量化结果。数据在 `reports/profiling`。

[预取策略与预算对照](docs/09_预取策略与预算对照.md) 联合报告尾延迟、总 H→D、无效预取与淘汰字节，并记录 CUPTI D→D 追踪。数据在 `reports/prefetch-study`。

[测量协议修复与配对复测](docs/10_测量协议修复与配对复测.md) 记录电源与运行状态、交错预算、逐帧配对区间、慢帧和消融。短预测版完整矩阵、慢帧率判据和 RGB 对照数据在 `reports/measurement-repair/candidate-matrix`。

[跨场景与跳转负载](docs/11_跨场景与跳转负载.md) 记录 Train 高命中对照、Playroom 需求上传与分块粒度实验，以及 [离线演示](demo/index.html) 的复现命令。

[短预测管线减法](docs/12_短预测管线减法.md) 记录 2×2 对照和独立 A/B 复验。原始顺序复验中，逐块候选的组 p99 差区间上界为 +1.91 ms，论文和演示采用渲染先提交、批量上传和向量化预测。

- `scripts/check_environment.py`：只读环境检测，输出 JSON。默认不导入大型依赖；`--probe-torch` 才实际验证 PyTorch CUDA。
- `scripts/smoke_render.py`：待依赖就绪后运行的 GPU 渲染验证。生成简单高斯场景，调用 gsplat，保存 PPM 图像及 JSON。
- `configs/experiment_plan.json`：待实现的实验参数草案。
- `reports/environment.json`：安装前环境快照。
- `reports/environment_cuda.json`、`reports/cuda_primitives.json`：安装后 GPU 和预取接口检查。
- `reports/smoke/synthetic.png` 与 `result.json`：真实 CUDA 后端渲染的验证图像与记录。
- `requirements-lock.txt` 包含 20 个运行依赖的固定版本与已核对散列。

在本项目目录执行：

```powershell
.\run_python.ps1 scripts/check_environment.py --probe-torch --output reports/environment_cuda.json
```

完整复查环境和渲染：

```powershell
.\verify_environment.ps1
```

该命令需要真实 GPU 与 gsplat CUDA 后端，会检查依赖、GPU 运算和图像输出。

## 实现顺序

1. 验证 CUDA 后端实际渲染并锁定依赖。
2. Train 预训练模型的全量渲染和固定轨迹回放。
3. 离线空间分块、保守边界选择和真实 CPU→GPU 加载。
4. Train 场景固定字节预算、LRU 淘汰及帧时间和画质对比。
5. 预取消融、短预测版矩阵、跨场景高命中对照和离线可视化演示。

