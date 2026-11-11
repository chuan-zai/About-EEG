# 异构导联 EEG 混合预训练重建研究

> 2026-11-07 当前云环境恢复：仓库位于 `/workspace/About-EEG`。本次恢复记录见 [恢复验证报告](reports/resumption-2026-11-07/engineering.md)。下文第三阶段数据、测试和锁定状态是历史记录；本环境以该恢复报告为准。正式训练尚未启动。

当前状态（第三阶段）：九名受试者数据与27个配对计划包已审计；V2通过S01 T-only三种子学习门槛，38项测试通过，协议已锁定。270个正式运行尚未启动。前两阶段说明和历史V1命令保留在下文，当前正式启动请使用末尾第三阶段V2命令。

本仓库在第一阶段工程上继续开发。当前 RQ1 是：混合 8ch/22ch 自监督训练是否损害独立完整 22ch 数据上的 masked reconstruction 泛化？RQ2 是可能下降与梯度干扰的关系，RQ3 是梯度保护是否有效；后两者尚未实现。不预设负迁移存在，分类探针保留但不再是本阶段主评价。

## 环境与已有验证

```bash
cd /workspace/eeg-transfer
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

`requirements-lock.txt` 是实际环境版本快照，editable 路径仅适用于当前工作区。当前 PyTorch 为 CPU 版本；真实数据已下载 S01 T/E 两个 Session，未运行正式 RQ1 实验。测试结果和限制见 `reports/engineering-phase2.md`，数据审计见 `reports/data-audit-S01.md` 及对应 JSON。CPU 足以完成短程验证；device=auto 自动检测 CUDA，CUDA 路径尚未实测。

## 数据下载与导入

来源为 BNCI Horizon 官方发布页：<https://bnci-horizon-2020.eu/database/data-sets>。使用其 A01T.mat/A01E.mat，不是自行制作或猜测标签的数据。网站列出 CC BY-ND 4.0，须署名；原始和派生数据均不纳入 Git，不在本项目中发布数据副本。引用 Tangermann et al. 2012, DOI:10.3389/fnins.2012.00055。来源文档、MOABB 通道/单位解析源码快照及哈希保存在 `reports/sources/`；未安装 MOABB，运行时通过 scipy 读取官方 MAT，MNE 的 GDF 接口继续保留。

```bash
# 明确下载一个受试者；已存在文件不会重复下载
.venv/bin/python -m eeg_transfer.bnci --subject 1 --download \
  --output data/processed/S01.npz --report reports/data-audit-S01.json
```

`bnci.py` 是新增的官方 MAT 适配器。`data.load_mat` 仍只读取第一阶段的规范化 schema，不能直接用来读官方结构；`load_npz` 已兼容元数据。GDF 接口仍需人工核验 channel_map，未在本阶段获得 GDF 实测。

MAT 每个 run 的 X 为 [samples,25] 微伏；前22列的实际电极名由官方布局与 MOABB 映射核验，后三列为 EOG，明确排除。`run.trial` 是 MATLAB 1-based 试次起点，cue 位于 +2 秒。固定提取 `[2,6)` 秒，250 Hz、1000点，乘 1e-6 转伏特。无新增滤波、重采样、重参考、去趋势或增强；原采集已包含 0.5–100 Hz 带通和 50 Hz 陷波。无 trial 的 EOG 校准 run 不参与训练。

默认在每个 Session 按官方 artifacts 标记剔除坏试次；保留原文件、被剔除的键/原因及已接收试次的 run、原 onset、cue sample、标签和 artifact 元数据。非有限 EEG 或越界 epoch 被剔除并记录，异常幅值和常数通道只审计，不用测试表现决定阈值。当前 MAT 适配器固定按本数据集 25 列、250 Hz 验证，不作为通用 EEG 解析器。

逐受试者：T 随机留20%试次验证（固定 split_seed=42、不按标签分层），E 全部最终测试。先划分 trial，再窗口化；每个窗口以 subject/session/trial/start 唯一定位。正式配置采用 250 点不重叠窗口，每 trial 恰好4个；smoke 为128点、步长128，末尾不足窗口丢弃。Standardizer 仅 fit 训练 trial，每电极独立 mean/std；8ch 转换先排除其他电极，不做跨通道统计。统计可使用训练 trial 全部1000点，不使用验证/测试样本。

## 模型与 mask

维持原 Temporal CNN、Channel Embedding、两层 Spatial Attention、Channel-query Decoder，dim=128。模型是固定窗口长度；所有比较组必须使用相同 time/dim/layers。`signal/target` [B,22,T]，`channel_ids` [B,22]，availability [B,22]，reconstruction [B,22,T]。重建采用整电极遮蔽：22ch 每样本遮6个电极、可见16个；8ch 每样本遮2个、可见6个。

固定8电极为 Fz、FC3、FC4、C3、Cz、C4、CPz、Pz，覆盖双侧中央区及中线，不声称最优。模拟8ch的其他14个电极在标准化前移除，既不进入输入也不进入损失。训练事件不含标签，标签仅在源数据与元数据中保留。masked MSE 只覆盖 availability AND reconstruction；每次更新记录 MSE、L2梯度范数、每样本可见/目标电极数、目标元素数、耗时、event ID、seed 和哈希，具体 trial/start/targets 由随运行保存的 plans.json 还原。

模型内部再次清除不可用/被遮蔽电极，并使用 attention key padding 隔离；query 只含电极身份，没有原信号残差路径。原模型的 temporal 全局平均池化丢失了时序相位信息，可能限制逐点波形重建；本阶段按要求保留架构，不依据短程测试结果调模型。未来若修改，应形成新的统一架构版本并重做全部对照。

## 固定重建评价

`reconstruction.evaluate` 使用 no_grad、eval 模式，不更新权重、梯度或归一化。所有组复用同一个计划包内的试次/窗口/mask，计划在训练前生成并保存。随机25%整电极遮蔽的固定种子为1101、2202、3303，每个窗口每个种子6个目标电极。

主指标 **RMSE_V = sqrt(所有目标元素伏特误差平方和 / 所有目标元素数)**，汇总包含所有预定 trial/window/seed，绝不是先算batch RMSE再平均。另报标准化RMSE、所有22电极的RMSE和Pearson、按mask seed和受试者的汇总。Pearson 按每电极所有目标值池化后计算，零方差/无目标返回 null。多mask种子共享相同测试试次，不应当作独立受试者估计统计置信度。

独立 outside14 协议：8个固定电极全部可见，其余14个全部遮蔽，直接由主模型推理。为了允许完整22ch测试真值评分，评价 batch 的 availability=22ch，reconstruction=其余14ch；内部 visible 恰好8ch。该协议使用一个确定性mask，无需重复随机seed，单独报告。它无需拟合探针，和第一阶段未实现的独立学习探针不同；该14ch真值绝不用于模拟8ch训练。两种评价不可混合，重建性能不代表运动想象分类性能。

## 配对训练计划及正式运行命令

先生成计划，不训练。当前正式 S01 计划已保存在被 Git 忽略的 `data/plans/rq1-S01/`。计划包含全部组的 trial、start、显式目标电极、montage、event seed、顺序、划分、数据哈希、固定评价计划和预算。

```bash
# 如需重建计划，选择一个尚不存在的输出目录
.venv/bin/python -m eeg_transfer.experiment prepare \
  --config configs/rq1-S01.json --out data/plans/rq1-S01-new

# 正式训练需人工独立启动；本阶段没有执行下列命令
.venv/bin/python -m eeg_transfer.experiment run \
  --plans data/plans/rq1-S01/plans.json --group A0 --out results/rq1-S01-A0
```

将 group 分别换为 A1/A2/A3/A1_exposure/A2_exposure/A3_exposure/A1_extra/A2_extra/A3_extra，输出目录分别命名。默认只评价 validation；协议和超参锁定后，在最终独立启动时增加 `--evaluate-test`。不要根据测试结果反复改配置。固定预算取最终 checkpoint，不进行基于测试或验证的自动选模。初始配置不代表已完成调优；正式多受试者、多种子实验还需预注册统计与预算。

混合比例单位是**更新数**，整batch使用同一montage，A0/A1/A2/A3分别为100/75/50/25% 22ch；所有组同初始化、优化器、batch与总更新数。目标数差异会导致监督量不同，已明确记录。高导联事件来自统一的有序序列，A1/2/3复用A0的前缀。曝光匹配对照仅保留对应混合计划的完全相同22ch事件，顺序一致；额外8ch对照保留A0全部22ch事件且顺序一致，插入额外8ch事件。使用独立随机流固定trial/window/mask，避免因插入低导联更新改变后续高导联抽样。

当前正式预算1000更新/batch8/time250：A0–A3总更新数相同；曝光匹配组更新更少；extra组更新更多。dense模型即使8ch也计算22个slot，不能把8/22直接作为FLOPs比例。`budgets.json` 保存目标元素数和输入slot数，`runtime.json` 记录实际训练耗时；不同预算组不能被称为等计算比较。性能下降可能来自完整导联曝光减少、监督量和优化差异，不能直接称作梯度污染。

## 重跑真实数据短程验证

```bash
.venv/bin/python -m eeg_transfer.experiment prepare \
  --config configs/real-smoke.json --out results/my-smoke-plans
.venv/bin/python -m eeg_transfer.experiment run \
  --plans results/my-smoke-plans/plans.json --group A2 \
  --out results/my-smoke-A2 --evaluate-test
```

smoke仅4次更新、batch2、每个评价split预先取2个trial。这里显式执行测试评价只为校验管线；该输出不能用于调参或负迁移结论。每个run保留配置、完整计划、数据/计划/初始化权重哈希、归一化、权重、优化器、依赖快照、逐更新日志、分指标评价及预算。输出目录必须不存在以避免覆盖。未实现训练中断后的精确断点恢复。

## 文件结构与后续工作

- `data.py`：Trials、原有加载器、划分、归一化、窗口；`bnci.py`：官方MAT下载/适配/审计。
- `masking.py` / `model.py`：双mask与模型；`train.py`：第一阶段合成冒烟和随机种子。
- `plans.py`：配对事件/固定评价计划；`reconstruction.py`：独立重建评价；`experiment.py`：单受试者真实运行器。
- `evaluate.py`：保留分类探针和未来梯度接口；`configs/`：合成、真实smoke与正式RQ1配置。
- `tests/`：原测试及第二阶段隔离、配对、指标、真实数据与日志回放测试；真实文件缺失时3个集成测试显式skip。
- `reports/`：来源、审计、实际测试和工程报告；`results/`：小规模真实/合成运行产物；`data/`：忽略的原始数据、派生数据和较大正式计划。

仅完成S01工程接入。下一步应预先锁定正式预算和统计方案，验证其他受试者的元数据与导入异常，逐个建立配置与计划；正式多随机种子须保持split_seed/evaluation_seeds不变、只改变训练seed并成对比较。汇总以受试者为单位，不能用窗口重复当独立样本。RQ2/RQ3、分类实验、全9人正式训练均未执行。

## 第三阶段：时间token、区域分析和九人正式计划

前两阶段代码接口与报告保留，`model.py` V1未变。V2在`model_v2.py`，每5点构成有序时间patch，每个时间位置执行通道注意力，保持原有双mask与身份query约束。旧配置默认V1，新正式配置统一V2。分类模块仅保留，不是本阶段终点。

第三阶段资料：

- `reports/phase3/model-validation.md`：S01 T-only三种子学习判据、基线、真实参数/FLOPs/CPU耗时和结果。
- `reports/phase3/data-audit-all.md` 与 `data/Sxx.json`：九人T/E审计、来源/校验和和坏trial。
- `reports/phase3/statistical-protocol.md`：受试者级推断、多重比较、区域选择性与优化器状态说明。
- `reports/phase3/protocol-lock.json`：是否已满足锁定条件及固定配置。
- `reports/phase3/rq1-manifest.json` / `commands.txt`：270个正式运行清单/命令；本轮未执行这些正式训练。

区域指标新增于`reconstruction.py`和`regions.py`：random25内分inside8/outside14，按各区域目标元素汇总，再与完全相同评价计划的A0配对；outside14仍独立。计划检查所有22电极的评价覆盖，覆盖次数不是独立样本数。预定受试者级主要比较、bootstrap CI和Holm方法实现于`statistics.py`和`summarize.py`。

```bash
# 已有九人数据，不重复下载；再次执行会核验已有文件
.venv/bin/python -m eeg_transfer.dataset_suite

# 若要重新生成27个计划包，请使用尚不存在的目录；只生成，不训练
.venv/bin/python -m eeg_transfer.suite --out data/plans/rq1-v2-new

# 正式单次运行：须在protocol-lock.json显示ready后由人工启动
.venv/bin/python -m eeg_transfer.experiment run \
  --plans data/plans/rq1-v2/S01/42/plans.json.gz \
  --group A0 --out results/rq1-v2/S01-seed42-A0 --evaluate-test

# 依完整预定清单汇总；未全部完成时只报告缺失，不输出推断结论
.venv/bin/python -m eeg_transfer.summarize \
  --manifest reports/phase3/rq1-manifest.json --out results/rq1-summary.json
```

S01–S09、seed42/43/44、10组由完整清单固定；运行其他组应逐项使用commands.txt，不根据E表现改seed/超参。原`configs/rq1-S01.json`和`data/plans/rq1-S01`是第二阶段V1候选历史计划，保留供追踪；新的正式研究应使用`configs/rq1-suite.json`及新V2计划。不能在一次分析中混合两种版本。

当前不支持精确断点续训；失败从相同事件计划/seed在新目录重启，不覆盖旧记录、不更换seed、不挑最好重跑。status标记完成状态；优化器与权重保存在最终checkpoint。原始/派生/大计划均在忽略目录；工作区没有有效Git索引，因此历史提交状态无法检查，忽略规则已在临时Git环境验证。架构验证与单元测试不是负迁移结果，正式全量RQ1尚未执行。

## Git仓库与云端数据

仓库地址：https://github.com/chuan-zai/About-EEG 。Git保存源码、配置、测试、数据溯源、协议与小规模运行日志；不包含原始/预处理EEG、正式大计划、虚拟环境或模型二进制。云端现有文件均保留，未因Git接入删除。克隆仓库后若需真实训练，须按数据接入与计划生成步骤准备本地文件；不能把仓库中保留的历史审计路径视为当前数据已存在。第三阶段报告中无有效Git索引的描述属于接入前的历史状态。

## 新环境恢复计划（保留历史清单）

计划生成器新增 `--reports` 和 `--results-root`；如果报告目录已有清单会拒绝覆盖。

```bash
cd /workspace/About-EEG
.venv/bin/python -m eeg_transfer.suite --out data/plans/rq1-v2-restored \
  --reports reports/resumption-2026-11-07/suite --results-root results/rq1-v2-restored
```

已生成时不要重复执行。正式启动命令见新目录的 `commands.txt`，本轮只恢复计划并运行短程工程验证。

## 项目Web操作台

```bash
.venv/bin/python -m eeg_transfer.console
```

浏览器打开 `http://127.0.0.1:8765`，输入终端显示的访问令牌。操作台覆盖数据审计/下载、只读协议与配置、配对计划、单次训练、测试、任务停止、实时日志/loss及结果汇总。不会自动启动全量正式实验。完整的安装、逐页操作、下载审计、冒烟验证、远程GPU部署、SSH转发及故障处理，见 [操作台详细使用指南](docs/CONSOLE.md)。

## 2026-11-11：原始数据Git LFS与完整操作指南

[操作台详细指南](docs/CONSOLE.md)覆盖首次启动、逐页面操作、数据准备、计划、任务启停和故障排查。[数据获取说明](docs/DATASET.md)给出18份官方MAT的Git LFS下载、许可署名、SHA校验及派生数据生成步骤。先执行 `git lfs install`、`git lfs pull origin work`；ZIP归档可能只有LFS指针。后文旧章节中“原始数据不入Git”的说明属于历史策略，本次仅对官方未修改MAT改用LFS，派生数组/权重/大计划仍不提交。
