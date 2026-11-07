# 本轮继续恢复记录

## 目标与范围

检验混合8ch/22ch自监督训练对独立完整22ch重建的影响与区域选择性。保留V1，统一V2；T内训练/验证，E仅最终测试；9人×3配对seed×10组。主终点是目标元素汇总RMSE_V，受试者内平均seed配对差，再按9人等权统计，主要三比较Holm校正。不预设下降，不从RQ1推断梯度污染，不实现RQ2/RQ3。

用户本轮授权下载缺失数据，同时明确不启动全量训练；随后指示网络受阻时继续本地工作。正式训练启动数0；真实E评价数0。

## 下载尝试与未解决项

官方地址 https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat 实测CONNECT 403。来源快照 reports/sources/moabb_bnci_base.py.txt:21 中的另一个地址 https://lampx.tugraz.at/~bci/database/001-2014/A01T.mat 也实测CONNECT 403。两份响应保存在network-probe.txt及mirror-probe.txt。运行环境允许域名不含两者，代理规则已生效；没有尝试绕过网络策略。另做了网页检索，未找到足以核验数据身份的其他下载源。未下载真实MAT或用其他数据替代。

inventory/status.json逐项列出18份缺失原始MAT的官方URL、旧绝对路径、当前绝对路径、期望SHA256；仍缺9份处理后NPZ、9份训练归一化及27个正式计划包。历史模型权重与Git bundle也未补齐，无法从这次文字交接中恢复。

## 已完成的本地工作

- 阅读并归档新交接文件，继续使用原项目实现。
- 112个历史文件仍全部匹配manifest。19个源码锁哈希、正式架构/超参、配对seed和270项运行网格核验通过，见preflight.json。
- 实际构建既定V2：98053参数；主要算子解析前向估算909004800 FLOPs/batch4（不是完整硬件计数）。未改模型、预算或seed。
- 新增restore_data.py，默认仅生成盘点；显式--download才下载。使用原适配器，原始文件SHA必须与历史审计一致才能转换；旧审计不改，已有派生数据不覆盖。转换后分别记录处理文件字节SHA与trial/split/单位/通道/坏段语义是否匹配，保存T训练归一化。字节SHA不同不能被语义一致掩盖。
- 已实际运行脚本盘点、CLI帮助和语法编译。下载/转换分支因网络及数据缺失尚未实测，不能宣称已恢复真实数据。
- pip check通过，当前完整依赖快照保存于requirements-freeze.txt。
- 原测试集本轮实跑：**23 passed, 15 skipped in 4.80s，0 failed，退出码0**。原始结果见pytest.txt及pytest.xml。15项跳过原因仍为真实数据和正式计划缺失，没有改测试条件。
- summarize实际执行，当前0/270正式结果，complete=false、statistics=null，见run-summary.json。历史正式实验也未执行，结果缺失不是丢失已有成果。

## 后续执行命令

在网络可访问官方来源的环境中：

```bash
cd /workspace/eeg-restoration/eeg-transfer
.venv/bin/python recovery/data-restoration-2026-11-07/restore_data.py \
  --download --out recovery/download-audit-01
.venv/bin/python -m pytest -q -ra \
  --junitxml=recovery/download-audit-01/pytest.xml
```

输出目录必须全新；脚本遇到下载/哈希/语义错误会保留记录并退出，不训练。脚本拒绝覆盖已有派生文件，部分恢复后需要先检查日志与文件再决定后续动作。若已有MAT被私下传入data/raw，相同命令只核验存在文件，不重复下载。

若处理NPZ字节hash不同，原集成测试可能失败，应如实记录、逐数组核验并在新的审计/测试版本处理迁移，不能修改旧hash求通过。恢复数据后也不要直接运行dataset_suite，它存在旧绝对路径与固定历史输出；suite也会覆盖固定reports/phase3文件。正式计划重建应在隔离副本、新清单和新锁修订中进行，未在本轮自动执行。

当前正式运行就绪判断仍为false；下一步是获得可访问的真实数据后完成数据集成核验，再单独处理计划恢复与完整性验证。原ready=true仅保留作历史证据。
