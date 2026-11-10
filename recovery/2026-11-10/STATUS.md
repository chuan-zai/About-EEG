# 2026-11-10 下载尝试与当前实例状态

用户授权自行从网站下载BNCI2014_001并push Git。本次保留既定研究范围：数据不入Git，不启动全量正式训练，不用E选模型。

## 实际结果

当前实例实时策略仍为restricted/enforced，observed revision=1，官方bnci-horizon-2020.eu和lampx.tugraz.at均未获准。实际运行restore_data.py --download后，官方请求在CONNECT阶段返回403，退出码1。Graz地址也重新实测CONNECT 403。原始18份MAT当前存在0份。没有下载成功、转换数据或执行正式训练。

下载异常、S01期望哈希与路径见download.log、download-attempt/status.json；九人18文件的URL/历史SHA/存在性见data-inventory.json；镜像响应见mirror-probe.txt。该错误来自当前代理连接，不是数据集不存在或许可拒绝。

## 远端同步与实测

先快进同步远端新增提交d5ce896及17c1c0a，保留其控制台、数据恢复报告和新测试。远端报告记载另一个实例成功恢复18文件，但Git不携带真实EEG或大型计划，不能认定当前实例已有数据。

对同步后的当前代码实际运行pytest，显式PYTHONPATH=/workspace/About-EEG避免旧editable工作区干扰：**32 passed, 15 skipped, 1 failed in 5.61s**。失败为tests/test_restoration.py::test_restored_suite_matches_historical_scientific_plan：已存在远端清单，但忽略目录data/plans/rq1-v2-restored/S01/42/plans.json.gz未传输，触发FileNotFoundError。15项跳过因真实数据/旧正式计划缺失。没有改变测试条件或伪装通过。原始日志/XML和退出码均保留。

## 后续

网络策略实际放行官方及重定向目标后，使用全新审计目录执行：

```bash
cd /workspace/About-EEG
PYTHONPATH=/workspace/About-EEG /workspace/eeg-restoration/eeg-transfer/.venv/bin/python \
  recovery/data-restoration-2026-11-07/restore_data.py --download \
  --out recovery/download-audit-next
```

当前借用既有虚拟环境；换机器应按pyproject重新安装。随后核对18个原始SHA、九人trial/split/通道/单位/训练统计，恢复正式计划到独立新输出并记录哈希迁移，再跑集成测试。不能覆盖历史锁和清单。所有大型EEG数据、权重和正式计划继续被Git忽略。
