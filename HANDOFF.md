# EEG项目：交给另一位AI的恢复说明

目标是恢复和继续既有项目，不是根据聊天描述重新发明实验。首先解压源码并审查实际文件；不得把本说明中的历史通过情况当作新环境实测结果。

## 文件与历史状态

源提交：1f670d3（完整hash见HANDOFF_MANIFEST.json）。源码ZIP包含代码、配置、测试、来源快照、9人审计、锁定协议和小规模运行日志；不含EEG数据、模型权重、虚拟环境及正式大计划。Git bundle另提供同一提交的离线Git历史；没有依赖GitHub上传成功。2026-11-07连接了公开仓库 https://github.com/chuan-zai/About-EEG ，但推送403，不能假设远端已有文件。云端原始文件仍保留于/workspace/eeg-transfer。

历史结果：原16测试复跑通过，第三阶段38测试通过；S01–S09两Session已在原工作区审计；V1保持原样，V2通过S01 Session T-only三seed学习能力门槛。正式计划9人×3seed×10组=270个运行，正式训练尚未启动。没有负迁移结论；RQ2/RQ3没有实现。

## 必须先阅读

1. eeg-transfer/README.md
2. reports/phase3/engineering.md、model-validation.md、data-audit-all.md
3. reports/phase3/statistical-protocol.md、protocol-lock.json
4. configs/v2-validation.json、rq1-suite.json
5. eeg_transfer下全部代码和tests下全部测试
6. reports/engineering-phase2.md、data-audit-S01.md及reports/sources

第三阶段报告中“没有Git索引”是接入Git前历史事实。README末尾说明了后续Git接入。源提交内不含Git认证信息；不要请求或在聊天中接收Token/密码/私钥。

## 科学约束

研究8/22ch混合自监督是否损害完整22ch独立EEG重建及是否有区域选择性；分类不作为主要终点，不预设负迁移存在，不实现RQ2/RQ3。保留V1，正式统一V2。8ch固定Fz/FC3/FC4/C3/Cz/C4/CPz/Pz；其余14ch不能进入该训练样本的输入或监督。不能使用跨通道归一化泄漏隐藏信号。

官方BNCI2014_001 MAT为25列（22EEG+3EOG），250Hz、uV转V。MAT trial为1based试次起点，cue在+2s，提取[2,6)秒，1000点。按官方artifact标记固定剔除，保留标签仅供未来使用。每人T划分训练/验证，E最终测试；先按trial分，再切窗，归一化只fit训练。

模型V2 dim64、两层attention、patch5；window250、stride250、batch4、AdamW lr=.001、wd=.01、无scheduler、基本1000步。配对训练seed42/43/44；split_seed42；固定评价maskseed1101/2202/3303。详细固定值以锁定协议和配置为准。

主评价为random25下全部被遮蔽目标元素的RMSE_V，按平方误差总和/元素数开方；inside8/outside14区域采用同样元素汇总。outside14是独立协议：8个可见、14个目标，不能与random25混合。所有组使用相同评价trial/window/mask，22ch训练事件严格配对。先在受试者内平均配对seed差，再按9名受试者统计；不能把窗口/maskseed当独立受试者。主要3个比较Holm校正，CI与探索区域比较按既定统计协议。

## 新环境恢复

建议Python3.12；从ZIP解压后在eeg-transfer目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q
```

依赖快照仅用于参考。requirements-lock.txt里有旧editable绝对路径和平台相关版本，不要盲目原样安装；记录新环境版本。CPU/GPU、依赖版本差异可能影响数值，不能承诺逐位复现。缺数据或大计划时相关集成测试会skip，必须报告实际pass/skip，不能照抄“38通过”。模型学习验证已有报告/日志，不需先重跑720个短程更新。

若选择Git bundle，可执行：

```bash
git clone eeg-transfer.bundle eeg-transfer
```

请勿把ZIP的代码和bundle克隆到同一已存在目录。新clone的origin可能指向本地bundle，按实际需要另行设置远程。不要假设GitHub写入权限已修复。

## 缺少的数据、权重与计划

源码包不是完整云盘备份。若用户需要精确恢复原运行产物，还需原工作区的data/及被忽略的checkpoint.pt；可通过单独私有数据包传递，不提交Git。没有数据时，只有在用户允许联网下载后才调用官方适配器；下载规则和许可见原审计（CC BY-ND 4.0，需署名，处理后数据分发须核对许可）。

重要路径陷阱：历史审计含/workspace/eeg-transfer/...绝对路径。若接收已有数据而迁移路径不同，dataset_suite会可能因旧绝对路径核验失败；不要通过批量替换历史报告来掩盖问题。保留原报告，新增迁移映射/核验记录，只在新的工作副本中修复路径解析。正式运行配置多数使用项目根目录相对路径。

若重新下载/转换数据或重建gzip计划，文件字节SHA可能变化（环境、序列化和gzip时间等），即使内容语义相同。必须区分源文件/信号内容/事件计划的差异；不能忽略校验错误或直接改旧hash为新值。保留原审计、清单和锁，在新路径输出新的配置/审计/计划/锁修订，并检查划分和mask是否一致。原suite生成器会写固定reports/phase3路径，所以在副本中运行或先扩展显式输出选项，防止覆盖历史记录。

可在用户明确允许后重建计划，但不可自动启动9人正式实验。恢复时优先检查缺失项和路径；不得通过改模型/预算/seed来让历史结果“匹配”。

## 验收与下一步

1. 核对源码提交和文件manifest；验证V1和各模块的SHA与protocol-lock记录一致。
2. 安装依赖并实际运行pytest，报告pass/skip/fail及缺失数据。
3. 核验数据来源/原始SHA、通道、单位、trial、坏段、split和训练统计。
4. 恢复或生成新版本配对计划，检查22ch事件、跨组固定评价mask和22电极覆盖。
5. 检查架构锁、超参、统计方案和新环境的限制；不因复制旧ready=true就宣称新环境已经验证。
6. 如需模型/加载器改动，保留旧实现及测试，新增版本并重新验证；不可用E选择架构。
7. 未获用户独立启动指示，不执行270个正式训练；不伪造结果、不自动实现梯度保护。

恢复后先给出实际审计结论、可运行程度、阻碍和明确下一步，不只给一个重写计划。

## 可直接复制给另一位AI的指令

请恢复附件中的EEG重建研究项目，而非从零重写。先阅读HANDOFF.md、实际源码、README、第三阶段报告和锁定协议；核对源提交及manifest，检查当前环境并实际运行可执行测试。保留历史代码、来源、报告、锁与运行记录。只使用T训练/验证作架构选择，不以E调参，不预设负迁移。源码包未包含真实数据/模型权重/正式大计划，请明确报告缺失项；未经授权不要下载数据或启动正式全量训练。路径、数据或依赖变动必须保留旧审计并建立新迁移/验证记录，不可直接修改旧哈希或照抄历史38项通过。完成后汇报实际测试结果、当前恢复程度、未解决问题和下一步建议。
