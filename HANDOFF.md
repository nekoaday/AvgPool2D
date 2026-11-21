# AvgPool2D 冲分交接

**接手目标**　在保持 FP16、FP32 全部测试点通过的前提下，继续提高 TensorOJ 队伍成绩，目标 100 分。请从 Git 提交 `0c9e873` 的 `kernel.asc` 继续；该版 2026-11-21 19:25:39（北京时间）送评，**Pass，30.00 分，提交 ID 1456578**。后续文档提交沿用这份内核。[该次评测详情](https://tensoroj.cn/cann/ccpc/onsite_2026_xiamen/avgpool2d/submission/6b017fb385ad43bbe0eafe48)

## 题目和代码入口

输入 `x[B,H,W]`。每个平面上，边长 K 的正方形窗口按行列步长 S 移动，输出窗口内 K² 个数的平均值。padding=0，输出高宽分别为 `(H-K)//S+1`、`(W-K)//S+1`。输出与输入同类型；题面给出的绝对和相对误差阈值为 FP32 的 1e-4、FP16 的 1e-3。公开样例为 FP32、[2,16,63]、K=S=2，输出 [2,8,31]。各 OJ 点的具体形状和 K/S 尚未确认。[比赛题面](https://tensoroj.cn/cann/ccpc/onsite_2026_xiamen/avgpool2d) · [基础教程](https://github.com/SugarYaoai/ascend-lecture)

直接读 [kernel.asc](kernel.asc) 的 `run_kernel`。Host 根据数据类型、形状、K/S 和可用核数选路。30 分版合入了队友的多行池化实现，并调整短行的分核边界。

| 分支 | 工作方式 |
|---|---|
| `strip_pool` | 适合多行处理的 S=1 小窗口，以及 S=2、K=2/3 的部分形状。最多一起处理 8 行、128 列；合并搬运与写回。短行按整行和 32 字节边界分核。 |
| `small_pool` | 其他满足小窗口条件的形状。K=S=2 用上下两行相加和 PairReduceSum；其余利用 Gather 做横向求和。 |
| `Pool::run` | 通用回退，可分段处理宽窗口。此前保留了标量横向归约；是否命中某个慢点需实验确认。 |

## 对照数据与下一轮实验

下面全部为微秒，列序与 OJ 一致。榜首行是用户提供的 19:18:23、74.93 分提交快照；每个测试点的历史最快值可能由不同提交创造。

| 版本 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 本队 30.00，`0c9e873` | 2.54 | 3.03 | 8.33 | 12.19 | 13.60 | 12.80 | 10.99 | 10.33 | 15.25 | 13.56 |
| 榜首快照 | 2.24 | 2.39 | 1.93 | 3.99 | 5.07 | 3.05 | 5.22 | 3.93 | 3.28 | 3.74 |

每点按 `100 / (1 + log(t/TBest) / log(1.5))` 计分，各计分点取平均。队伍取历史提交的最高分；历史最快耗时刷新或赛后纳入隐藏点时，总分可变化。比较改动请优先看同点微秒数并保留原始回执。[官方规则](https://tensoroj.cn/cann/ccpc/onsite_2026_xiamen/scoring-rules)

**下一位 agent 的第一轮**：保住第 1、2 点（分别约为榜首快照的 1.13×、1.27×）。优先调查第 9、3、6 点（约 4.65×、4.32×、4.20×）。先选一个可单独切换的改动，例如只改变 stride=1 多行路径每核处理的行数，其他分支沿用基线。用本地用例验证这条路径和尾块，再送 OJ 看哪些点响应。一次结果只能支持该改动的总体效果，无法从耗时反推隐藏形状。若某点明显提速，下一轮再收窄 Host 的启用条件，降低其他点回退的风险。

## 在这台电脑上开发

Windows PowerShell 中运行：

```powershell
Set-Location 'C:\Users\Neko\Documents\GitHub\AvgPool2D'
git status --short --branch
git log -1 --oneline -- kernel.asc
wsl -d CANN-Dev -u root -- avgpool-check
```

已安装的 WSL2 发行版叫 `CANN-Dev`，位于 `D:\WSL\CANN-Dev`，系统为 Ubuntu 22.04 x86_64。CANN Toolkit 9.0.0 环境脚本是 `/usr/local/Ascend/cann/set_env.sh`。本机有毕昇编译器，无昇腾卡。

`avgpool-check` 是 WSL 内的 `/usr/local/bin/avgpool-check`。它编译当前工作树的 dav-2201 NPU 目标和 CPU 调测目标，生成样例、运行 CPU 版内核，并调用 [sample.py](sample.py) 对比 [reference.py](reference.py)。成功标志为 `mismatched=0/496` 和 `PASSED`；构建缓存位于 `/root/.cache/avgpool2d/`。30 分版另通过 14 组 CPU 专项用例，覆盖 FP16/FP32、非对齐行宽、跨 128 列、多核和回退路径；这些用例独立于一键命令。换机时按 [README 的 Linux 步骤](README.md#在另一台-linux-开发机上验证) 重建环境。

CPU 调测用于查正确性，dav-2201 编译用于查目标指令兼容性。此前试过 `msprof op simulator`，在设备上下文初始化阶段失败，尚无可信的流水耗时。当前以 OJ 做性能 A/B。此前 V3 就曾在 OJ 因 AI Core 的无符号整数转 float 而编译失败，单跑 NumPy 参考模型覆盖不到这类错误。

## 交付节奏

1. 遵守 [AGENTS.md](AGENTS.md)。先读工作树状态，保留用户文件；根目录的模板 ZIP 是用户下载的空工程。
2. 一轮只验证一个优化假设。跑本地检查，再针对改动的 dtype、边界或分核路径补测。每个待送评版本先提交本地 Git，给出短哈希。
3. 用户偏好亲自送评。交付该提交的 `kernel.asc`；在这台 Windows 机器上可将全文放进剪贴板，省去浏览器逐步操作。远端推送按用户要求执行。
4. 把回执的 OJ ID、时间、状态、分数及十项耗时关联到 Git 提交，记录哪些点改善或回退。继续维护此文档里的交接基线。
