# 第一阶段实际验证

环境：Python 3.12，PyTorch 2.15.0+cpu；CUDA 不可用。无真实 EEG 数据下载。

执行 `.venv/bin/python -m pytest -q`：9 passed in 12.89s。
执行 `.venv/bin/python -m eeg_transfer.train --out results/smoke`：退出码 0。
默认模型 362944 个参数；CPU、batch=4、4 次更新（2 次22ch，2次8ch），完成前向、有限 loss、反向与优化器更新。具体日志见 smoke/train.jsonl；不作性能解释。

测试还覆盖标准 schema MAT 往返；真实竞赛 MAT 与 GDF 文件尚未验证。无 CUDA 验证、真实分类评价、正式 RQ1 或 RQ2/RQ3 结果。9项计数包含8/22ch参数化测试。

文件索引：data.py 数据/划分/统计；masking.py 导联与双mask/来源元数据；model.py 通道感知模型；train.py 可重复冒烟与调度；evaluate.py 重建评价/固定分类探针/后续接口；configs 两份配置；tests/test_core.py 关键测试；README.md 协议和限制；pyproject.toml 与 requirements-lock.txt 依赖；results/smoke 为实际合成运行产物。
