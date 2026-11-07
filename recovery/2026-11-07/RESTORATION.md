# EEG 项目恢复审计（2026-11-07，本环境实测）

## 当前恢复程度

从附件 ZIP 原样恢复至 /workspace/eeg-restoration/eeg-transfer，没有从零重写。已阅读交接、README、第三阶段工程/学习验证/九人审计/统计协议/协议锁、配置、全部项目 Python 源码与三份测试文件，并核对第二阶段和来源材料。未修改历史源码、测试、报告、协议锁或运行记录。新增内容仅在 recovery/2026-11-07，及安装生成的 .venv / 包元数据 / 缓存。

HANDOFF_MANIFEST 的112个文件SHA256全部匹配；协议锁19个源码及3个配置/清单SHA256全部匹配；V1原始SHA检查通过。测试后再次核验112文件均未变。证据见 integrity.json、post-test-integrity.json。

交接manifest与ZIP comment均声明提交1f670d36b3fa9d1d97c5a14f2904375bcdece075。这是相互一致的快照来源声明，不是Git提交对象验证：附件未提供bundle或.git；现有 /workspace/About-EEG 只有无提交的Git目录，未修改该目录、未假设远端已上传成功，也未伪造历史。

## 环境与测试

独立 .venv；Python3.12.14，torch2.15.0+cpu，numpy2.5.4，scipy1.18.1，pytest9.2.0，mne1.13.2，scikit-learn1.9.1。CUDA不可用，CPU可运行。依照pyproject安装依赖，未照搬旧editable绝对路径；pip check通过。安装日志、完整freeze、environment.json均保留；不承诺与旧环境逐位复现。

实际命令（项目根目录）：

```bash
.venv/bin/python -m pytest -q -ra --junitxml=recovery/2026-11-07/pytest.xml
```

结果：**23 passed, 15 skipped in 5.28s；0 failed，0 errors，退出码0**。见pytest.txt、pytest.xml、pytest-exit-code.txt。共38项收集，但不是38项通过。

跳过分布：第二阶段S01真实数据3项；第三阶段S01–S09数据/单位/划分/归一化9项；压缩V2真实运行器1项；真实数据前提下协议锁拒绝变更1项；正式27包计划完整性1项。具体原因与位置见pytest.txt。没有改skip条件或用合成文件冒充真实数据。

通过范围包括合成管线、V1/V2屏蔽安全性/有限梯度、区域元素汇总、配对事件和固定评价计划、受试者统计与不完整汇总处理、历史T-only gate证据检查。gate证据检查通过只证明所保留记录符合测试断言，并非当前重新训练或独立重现720次学习更新。原测试中的合成E和固定分类探针fixture用于工程回归，不用于架构选择，未运行真实E评价。

另外实际运行 summarize，只输出新路径 current-run-summary.json：0/270正式结果可用，complete=false，statistics=null。未完成的正式结果从未历史生成，不属于传输丢失。

## 缺失与阻碍

missing-files.json列出63项：18份T/E原始MAT、9份处理后Sxx.npz、9份训练归一化、27份正式gzip计划。另缺历史checkpoint.pt、第二阶段正式S01大计划、原始description.pdf及离线Git bundle/历史。来源提取文本、网页、MOABB源码快照、九人历史审计、小规模日志、V2 gate记录、270运行清单和锁均在。

历史审计引用 /workspace/eeg-transfer，当前根目录为 /workspace/eeg-restoration/eeg-transfer。dataset_suite读取旧审计中的绝对路径核验，迁移数据后会有路径阻碍。没有批量改写历史路径或hash，也没有建立虚假旧路径。配置多数为项目根目录相对路径，运行时应先cd到根目录。

历史ready_for_formal_run=true保持作为历史证据；本环境当前不具备真实数据/计划/权重的完整恢复条件。尤其实际运行器的锁检查只核对ready及架构/超参，不自动核验锁内全部代码哈希；本轮已另外做完整哈希核对，后续仍须做新环境启动前核验。

## 下一步（尚未执行）

1. 优先接收原工作区的Git bundle及私有data/、历史checkpoint，恢复后逐文件核对旧审计SHA；若只能重建数据，先取得明确联网下载授权。
2. 为新根目录单独建立路径映射与迁移核验记录；如需适配路径解析，在新工作副本修复，不覆盖原审计。
3. 真实数据到位后检查来源/单位/通道/坏trial、T内split和训练统计，再重跑15个跳过测试。模型选择仅允许T训练/验证，E只按锁定协议最终评价。
4. 优先恢复原配对计划；若需重新生成，另行明确授权，在副本与新版本输出路径中运行。suite生成器会写固定reports/phase3路径，不可直接运行覆盖历史；序列化字节变化与事件/信号语义变化必须分别核对，保留旧hash及锁。
5. 完成这些核验后另行报告正式运行准备状态；270个正式运行仍须独立启动指示。

本轮没有下载EEG数据、重建正式计划、重跑学习gate、启动正式训练、改变模型/预算/seed或实现RQ2/RQ3/梯度保护。没有负迁移结论。
