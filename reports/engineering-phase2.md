# 第二阶段工程验证报告

本报告只记录实际执行的工程验证，不是负迁移实验结论。

## 实际命令与结果

```bash
.venv/bin/python -m eeg_transfer.bnci --report reports/data-audit-S01.json
.venv/bin/python -m eeg_transfer.experiment prepare --config configs/real-smoke.json --out results/real-smoke-plans
.venv/bin/python -m eeg_transfer.experiment run --plans results/real-smoke-plans/plans.json --group A2 --out results/real-smoke-A2 --evaluate-test
.venv/bin/python -m pytest -q
```

原始文件使用官方链接 curl -L --fail 下载；上述bnci命令在下载完成后进行审计与适配。运行器均返回退出码0。最终pytest：**16 passed in 22.88s**；无skip（S01文件存在）。原始pytest输出见 pytest-phase2.txt。

环境：Python 3.12.14，PyTorch 2.15.0+cpu，NumPy 2.5.4；device=cpu，CUDA不可用。依赖版本见requirements-lock.txt和run/environment.txt。

模型保持128维、两层Attention，窗口128点；参数371200。训练batch2，共4次更新（22ch两次、8ch两次），实际目标元素4096。实际训练环节耗时0.219秒，不包括导入、准备、评价与日志快照，不能代表正式实验成本。

每次更新的loss和梯度范数均有限；22ch可见16/目标6，8ch可见6/目标2。日志包含事件ID/哈希/seed/目标元素数/耗时；具体样本键、窗口与目标电极在同目录plans.json。归一化只fit218个T训练trial。

## 冒烟评价输出（不能用于科学结论）

每split在训练前固定取2trial。random25含3个mask seed、每trial7个128点窗口；outside14为一个确定性mask，评价协议分别报告。仅执行A2，未进行A0–A3性能比较，也未调参。

| 评价 | 目标元素数 | RMSE (V) | 标准化RMSE |
|---|---:|---:|---:|
| validation_random25 | 32256 | 9.3816253e-06 | 0.94261936 |
| validation_outside14 | 25088 | 9.4498555e-06 | 0.95164811 |
| test_random25 | 32256 | 9.6068513e-06 | 0.96272862 |
| test_outside14 | 25088 | 9.4942348e-06 | 0.95364735 |

所有22个电极逐通道RMSE/Pearson、种子级和受试者级输出见results/real-smoke-A2/evaluation.json。外14ch协议中的8个输入电极不作为目标，其指标为null，不能将这些空值视为0。

## 验证覆盖

- 原有9项：划分、NPZ/MAT规范schema、双mask、隐藏通道/目标扰动、有限梯度、控制数量、合成重复性与分类探针。
- 新增7项：trial/window隔离与held-out统计隔离；事件配对与固定评价计划；目标元素加权RMSE及Pearson精确值；14ch协议无梯度/无权重变化且8ch训练拒绝该监督；真实官方MAT映射/微伏转换/元数据/哈希；真实8ch隐藏值及目标泄漏扰动；真实短程训练日志/配置/划分/初始化/评价mask回放。
- 真实日志回放测试独立执行两次4步、dim32窗口32验证，除耗时外逐更新日志完全相同。该较小测试模型与正式各组共享架构的约束无冲突；真正smoke运行使用dim128。

## 配对计划与预算

正式配置仅prepare，没有执行run；data/plans/rq1-S01/plans.json约32MB，位于忽略目录。其哈希和各组预算见rq1-S01-budgets.json。A1/A2/A3的高导联事件分别与A0前750/500/250个完全一致；exposure保留这些事件；extra保留A0全部1000个高导联事件并插入低导联更新。

| 组 | 总更新 | 22ch | 8ch | 目标元素数 |
|---|---:|---:|---:|---:|
| A0 | 1000 | 1000 | 0 | 12000000 |
| A1 | 1000 | 750 | 250 | 10000000 |
| A1_exposure | 750 | 750 | 0 | 9000000 |
| A1_extra | 1333 | 1000 | 333 | 13332000 |
| A2 | 1000 | 500 | 500 | 8000000 |
| A2_exposure | 500 | 500 | 0 | 6000000 |
| A2_extra | 2000 | 1000 | 1000 | 16000000 |
| A3 | 1000 | 250 | 750 | 6000000 |
| A3_exposure | 250 | 250 | 0 | 3000000 |
| A3_extra | 4000 | 1000 | 3000 | 24000000 |

混合组同预算但高导联曝光和监督元素数不同；exposure/extra总计算不同。当前dense22slot模型不会按可用通道数量线性降低计算。训练墙钟时间和目标数均记录，不将所有对照当等计算组。

## 发现的问题与剩余工作

1. 原有规范MAT接口不能读取官方结构：新增bnci适配器，不改变旧接口语义。MAT trial起点不能直接当cue：固定+2秒并以原数组索引测试。
2. 第一阶段随机计划不能保证高导联配对：新计划使用独立高/低事件流和显式mask，测试逐事件完全一致。
3. 第一阶段仅有batch指标：新增全目标元素充分统计汇总，避免平均batch RMSE偏差。
4. 原模型temporal全局平均池化可能限制波形时序信息；按任务保留，没有通过测试集调参。任何架构改动应另行预注册并重跑全部组。
5. 仅S01；其他受试者数据、跨受试者差异估计、正式多种子置信区间未完成。当前评价已有by_subject输出，但单人结果不能估计人群差异。
6. GDF与其他官方MAT受试者、CUDA路径、中断精确恢复均未验证；下载仅做本地哈希追踪，无独立官方checksum。
7. 不实现RQ2/RQ3、不运行分类、不运行正式大规模训练。无任何负迁移结论。

## 主要变更文件

新增：eeg_transfer/bnci.py、plans.py、reconstruction.py、experiment.py；tests/test_phase2.py；configs/real-smoke.json、rq1-S01.json；reports数据审计/来源/工程/测试/预算记录；results/real-smoke-plans与real-smoke-A2实际产物。

修改：data.py（来源元数据随划分/NPZ保留）、train.py（CUDA确定性环境配置）、configs/protocol.json、README.md、requirements-lock.txt、.gitignore。模型和原始mask/loss算法未修改。
