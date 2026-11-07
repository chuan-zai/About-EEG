# 第三阶段工程交付与实际执行记录

本报告延续前两阶段。所有原有报告、配置、S01文件、历史计划和实验产物保留；V1 model.py字节级不变。未执行全9人正式RQ1训练，没有负迁移或区域损害的科学结论。

## 初始审计

原16测试复跑：16 passed in 18.06s。S01原始/派生SHA256和第二阶段正式计划SHA256与旧审计一致；random25按目标元素汇总，outside14恰好8可见且无梯度。审计记录见initial-audit.json和initial-tests.txt。

/workspace/.git为空，不是可用Git仓库；无法检查历史tracked index或提交，不声称已证明历史数据未提交。本轮没有执行提交或初始化实际工作区Git；临时Git环境的check-ignore确认data/raw、data/processed、data/plans及results/rq1-v2被排除，记录见git-ignore-check.txt。实际745MB原始、183MB派生EEG及15MB正式大计划均位于data/。小规模历史记录未删除。

## 实际执行

1. `python -m eeg_transfer.dataset_suite`：九人T/E均ready，S01跳过下载且核对旧hash。每受试者独立训练归一化已保存；无部分失败。全部坏trial键/原因、SHA256和来源见data/Sxx.json；汇总data-audit-all.md。
2. `python -m eeg_transfer.validation`：S01 T-only，V1/V2各3seed各120更新，共720次有限验证更新。未读取E来选择模型。V2全部3seed通过预先门槛；所有loss/梯度有限。详细参数、CPU耗时、不完整FLOPs估计和基线见model-validation.md；原始记录在results/v2-validation，日志phase3-learning.log。
3. `python -m eeg_transfer.suite`：9人×3seed的27个gzip包、270个正式运行描述生成完毕。生成过程中逐一检查22ch事件配对和跨seed相同划分/评价mask。未执行commands.txt中的正式训练命令。
4. `python -m eeg_transfer.summarize ...`：实际检查0/270正式结果时输出complete=false、statistics=null，仅缺失清单；见pre-run-summary.json。不会把未完成结果或测试fixture汇成科学结论。
5. pytest除了旧16项，新增V2、区域、基线、统计、九人原始MAT单位比对、压缩V2短程运行/优化器步骤、不访问测试、协议锁拒绝变更、汇总完整性、27个实际大计划完整性、T-only学习gate证据检查。

原有第二阶段16项测试包含在临时目录对E单trial的管线回归，按本轮要求原样复跑；这些临时数值不进入架构gate或任何模型选择。架构学习验证脚本单独只读A01T.mat，E的数据质量审计和计划生成也不作为选择依据。

## 评价与统计

random25主终点仍为E完整22ch、3个固定maskseed、每window遮6电极，按全部目标元素平方误差汇总RMSE_V。新增inside8/outside14同样按各区域目标元素计算，绝不是区域/批次RMSE算术平均。逐电极Pearson/误差保留，外14ch单独协议不混入主终点。正式计划中每个测试电极在mask/window覆盖计数上至少666次（门槛100），不是666个独立样本。

全部组以同一subject/seed初始化，严格复用配对的22ch trial/window/显式mask。预算记录总更新、22/8更新、目标元素、slot数量；运行日志新增optimizer_step/LR，manifest记录AdamW参数与独立状态，final status记录每参数实际步数。无scheduler；更多低导联更新会改变moment/bias-correction/weight-decay次数，不能直接解释为梯度污染。受试者级平均、bootstrap CI、Holm多重比较和区域选择性定义见statistical-protocol.md。

## 恢复与锁定

不实现精确中断恢复。输出目录必须不存在；失败保留原日志，使用同一plan/seed从头在新目录重跑，手工记录替代路径。完整结果标记complete，权重/状态原子写入；汇总缺少任一预定运行时不做推断。不得按最终测试结果选择重跑、seed或模型。

锁定依据：T-only三seed预定能力门槛、安全/协议测试、九人审计和27个配对包完整性；正式协议架构/超参在protocol-lock.json固定，运行器拒绝未ready或不匹配配置。GPU/其他机器性能不保证与CPU逐位一致。协议锁定表示适合开始预定正式实验，不保证存在负迁移、统计显著性或区域差异。

## 主要新增与修改

最终自动测试：**38 passed in 171.15s，无skip**；完整原始输出见tests.txt。另以collect-only验证新增集成测试在数据/计划不存在时具备skip条件。解析计算估计已实际执行并核对FLOPs=2×主要算子MAC总和；结果见compute-estimates.json。

新增模块：compute.py、model_v2.py、models.py、dataset_suite.py、validation.py、regions.py、statistics.py、summarize.py、suite.py。

修改模块：experiment.py（模型工厂、gzip、配置锁、原子最终保存、optimizer/LR状态及代码hash）、reconstruction.py（区域充分统计）、plans.py（覆盖审计）。原V1、mask和masked loss实现不变。

新增测试：tests/test_phase3.py。新增配置：configs/v2-validation.json、rq1-suite.json。README追加第三阶段运行方式，原配置留作历史V1版本。新增reports/phase3内来源审计、报告、协议、清单、测试和学习日志；数据/正式计划在忽略目录，实际学习产物在results/v2-validation。

## 尚未完成

正式九人三seed的A0–A3及所有控制结果、区域损害结论、真实E主要指标和跨受试者统计尚未产生。CUDA、精确断点恢复和额外人工伪迹标注未验证/实现；统计N=9的CI与sign-flip假设有明确局限。分类不运行，RQ2/RQ3不实现。其他受试者结构和数据已审计，但没有据E性能调整任何规则。

最终锁定判断：**可以锁定当前V2重建RQ1协议**。protocol-lock.json已标记ready=true，27个配置与固定架构/超参逐项匹配。锁定不表示存在负迁移；270个正式运行均未启动。候选配置原status字段保留用于溯源，最终ready状态以独立protocol-lock.json为准。
