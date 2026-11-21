# AvgPool2D · CCPC 厦门站

用 Ascend C 实现三维输入 `[B,H,W]` 的二维平均池化。输入和输出支持 FP16、FP32，窗口无填充，输出尺寸向下取整。[TensorOJ 题目与提交记录](https://tensoroj.cn/cann/ccpc/onsite_2026_xiamen/avgpool2d)

## 当前交接基线

**新 agent 从 [HANDOFF.md](HANDOFF.md) 开始。** 已验证的冲分基线为 `0c9e873`，OJ 提交 1456578，Pass，30.00 分。交接文档包含十项原始耗时、榜首快照和下一轮实验建议。

| Git 提交 | 已记录的 OJ 结果 | 主要变化 |
| --- | --- | --- |
| `18da943` | Pass，16.75 分 | 首版通用池化 |
| `3dfceee` | Pass，20.72 分 | 小窗口使用跨行搬运和 Gather |
| `448d5ac` | 已通过本地目标编译和官方 CPU 样例 | 为 K=S=2 加入 FP32 成对归约 |
| `ad67bc7` | 已通过本地目标编译和多核功能验证 | 精简小窗口分核与坐标计算 |
| `0c9e873` | Pass，30.00 分 | 合入队友的多行计算，调整短行的分核边界 |
| `fef3583` | Pass，总分待补；相比 `0c9e873` 约 28.69 → 28.64 | 将 stride=1 行批从 8 增至 24，作为单变量 A/B |

评分采用各点相对历史最优耗时计算，同一耗时对应的分数会随基准变化。比较两版时，优先对齐测试点编号和微秒数。第 1、2 点现为 2.54、3.03 µs，隐藏形状仍待确认。[比赛计分规则](https://tensoroj.cn/cann/ccpc/onsite_2026_xiamen/scoring-rules)

## 内核与验证范围

[核函数与 Host 入口](kernel.asc) 在 `run_kernel` 里选择实现。适合批量处理的小窗口走 `strip_pool`，每块最多 8 行、128 列；短行按输出行和 32 字节搬运边界分核。`small_pool` 处理其余小窗口，K=S=2 使用成对归约；更大的窗口走通用路径 `Pool::run`。优化某一路径时，保留其他分支作为对照。

[样例程序](main.asc) 运行 FP32 输入 `[2,16,63]`、K=S=2。[样例脚本](sample.py) 生成输入并和 [NumPy 参考实现](reference.py) 比对。30 分版在交付前通过目标编译、官方样例（496 个输出全部匹配）以及另外 14 组 CPU 调测，覆盖两种精度、奇数宽度、跨块、多核和回退路径。14 组属于该次专项验证，下面的一键命令只跑官方样例。当前没有可用的核内性能采样数据，真实耗时由 OJ 或昇腾设备验证。

## 在当前 Windows 开发机上验证

在仓库根目录的 PowerShell 中运行：

```powershell
wsl -d CANN-Dev -u root -- avgpool-check
```

这台机器的 WSL2 Ubuntu 22.04 位于 `D:\WSL\CANN-Dev`，已安装 CANN 9.0.0。发行版内的 `/usr/local/bin/avgpool-check` 会加载环境，分别编译 dav-2201 的 NPU 和 CPU 目标，再执行 CPU 样例。成功时末尾显示 `mismatched=0/496` 和 `PASSED`。构建产物在发行版的 `/root/.cache/avgpool2d/`。这个命令是本机安装的工具，新克隆的仓库可使用下方步骤。

## 在另一台 Linux 开发机上验证

准备 CANN Toolkit 9.0、CMake、C++ 构建工具和 Python 3。在仓库根目录执行，环境脚本路径按实际安装位置调整：

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -m pip install -r requirements.txt
cmake -S . -B build-npu -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build-npu --parallel 4
bash run.sh cpu
```

在有兼容昇腾设备的机器上，`bash run.sh npu` 会进一步运行并校验 NPU 样例。`run.sh` 支持 `DEVICE_ID`、`BUILD_JOBS`、`PYTHON` 和 `NPU_ARCH`。仓库的 [GitHub Actions 工作流](.github/workflows/cann.yml) 需要有 `linux`、`cann-9` 标签的自托管 runner，可用性需在对应 GitHub 仓库检查。

## 送评与记录

按 [开发约定](AGENTS.md) 验证并提交本地 Git 后，将该提交的 `kernel.asc` 全文放入 TensorOJ 的同名编辑器，由用户送评。收到结果时记录提交哈希、OJ 提交 ID、时间、状态、总分和按原顺序排列的十项微秒数。交给另一台机器时，核对 Git 历史里含有上述内核提交及本交接说明。
