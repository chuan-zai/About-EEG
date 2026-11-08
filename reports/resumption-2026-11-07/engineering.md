# 当前云环境恢复与继续开发（2026-11-07）

已阅读用户原始提示词（包含三个阶段），继续既有重建研究，不重建模型。工作目录 `/workspace/About-EEG`；从 origin/work 取回提交 `f2c52a2ba3a51501fe25973cd910fe67463636f7`。本轮公网策略 unrestricted/enforced，官方URL通过302重定向至TU Graz并返回200，解除了历史403阻碍。

## 数据恢复

沿用官方BNCI2014_001 MAT适配器、来源和许可记录。18个T/E文件均从历史审计中的官方URL下载，全部SHA256与历史值一致。九人预处理NPZ的字节SHA256也全部一致；每个Session的run、trial、通道、采样率、单位和坏段处理以及训练/验证/测试划分均通过恢复脚本比较。训练归一化按每人Session T训练trial重新拟合，并由真实数据测试核验。详细路径迁移、原始/派生SHA和比较结果见 `data/status.json` 与 `data/S01.json`–`S09.json`。历史报告和哈希未修改。

数据单位为V，250Hz，22EEG（排除3EOG），cue对应MAT试次起点+2秒，固定[2,6)秒。维持T训练/验证、E测试；不依据E调整预处理或架构。来源 https://bnci-horizon-2020.eu/database/data-sets/001-2014/ ，重定向至 https://lampx.tugraz.at/~bci/database/001-2014/ 。许可/署名要求沿用历史CC BY-ND 4.0记录，不分发数据副本。

## 本轮修改

- `eeg_transfer/suite.py`：增加独立报告目录、结果根目录；生成前拒绝覆盖已有清单/命令。解决迁移重建计划会覆盖历史清单的问题，不改变抽样、模型、超参和预算。
- `tests/test_restoration.py`：两项历史清单保护测试及一项恢复清单/源文件/评价mask/22ch配对完整性测试。
- README：标注历史与当前状态差别，补充新环境恢复命令。
- 新记录均保存在此目录；未改旧审计、学习gate、模型V1/V2、配置或协议锁。

## 短程验证

本环境S01 V2 dim64/layers2/patch5，250点窗口、batch4、A2四次更新。仅评价预先限定的1个T验证trial，未调用最终测试评价。全部loss及梯度有限，状态complete，优化器4步。运行记录见 `smoke/`，配置见 `smoke-config.json`，checkpoint保存在忽略目录 `results/rq1-restored-smoke/`。该冒烟仅验证工程，不用于比较模型或支持负迁移结论。原测试中的小规模E评价属于既有固定工程校验，不用于模型选择。

## 正式实验范围与限制

保持V2锁定科学方案、9人×3配对seed×10组；仅准备270个运行，正式训练启动数为0，不实现RQ2/RQ3。新环境没有恢复历史模型权重，也没有重跑720步架构学习验证；历史T-only gate仅作为保留的既有证据，不能声称本轮重新训练复现。CPU环境，CUDA未验证；精确断点恢复仍不支持。

历史协议锁中的源码SHA反映原快照，本轮仅suite编排代码改变，模型/科学配置不变；实际每次训练manifest会保存当前全部源码SHA。正式启动前应使用本轮新清单，不使用历史gzip文件哈希或旧绝对路径。数据及大计划受Git忽略，已检查tracked文件无data/或模型权重。

## 已恢复计划与启动方式

新清单 `suite/rq1-manifest.json`、`suite/commands.txt`，大计划位于 `data/plans/rq1-v2-restored/`。270项的派生数据hash与历史一致，所有评价mask哈希与历史一致；新gzip包字节hash不同，已在 `plan-migration.json` 显式记录，没有改写历史hash。恢复专项3项测试已实际通过（13.07秒）。

以下为日后独立启动的单个正式运行示例，本轮未执行：

```bash
cd /workspace/About-EEG
.venv/bin/python -m eeg_transfer.experiment run \
  --plans data/plans/rq1-v2-restored/S01/42/plans.json.gz \
  --group A0 --out results/rq1-v2-restored/S01-seed42-A0 --evaluate-test
```

科学方案可以继续沿用历史锁定版本；本轮没有依据E评价更换架构/参数，也没有新的学习能力或负迁移结论。迁移后的文件校验依据本轮清单，CPU/GPU数值不能保证逐位一致。

## 最终实际测试结果

`.venv/bin/python -m pytest -q -ra --junitxml=reports/resumption-2026-11-07/pytest-final.xml`：**40 passed, 1 skipped in 29.44s**，0失败。跳过的是硬编码历史 `data/plans/rq1-v2/` 和历史清单路径的旧正式计划集成测试；没有把新文件放入旧路径冒充历史包。新恢复专项测试实际核验本轮27包/270运行、数据SHA、历史评价mask相等及所有22ch配对关系并通过。九人真实数据映射、单位、trial隔离、归一化与S01短程运行测试均通过。完整输出见 `pytest-final.txt` / XML。

`pip check` 无依赖冲突，`git diff --check` 通过。正式结果汇总仍为0/270，不完整，不输出推断统计；见 `formal-results-status.json`。未提交/推送Git改动，数据和计划仅留当前工作区。
