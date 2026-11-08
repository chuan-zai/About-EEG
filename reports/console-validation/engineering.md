# 项目操作台工程验证

在既有EEG工程内实现Python标准库HTTP服务与无构建前端。数据台账、历史审计、只读配置/协议、清单筛选、单个正式训练、下载/审计、生成配对计划、测试、冒烟、任务停止、实时尾日志、loss及梯度、原始JSON结果与清单汇总均接入现有模块；无演示指标。

本轮新文件：eeg_transfer/console.py、console_worker.py、web/index.html/style.css/app.js、tests/test_console.py、docs/CONSOLE.md。修改pyproject包含前端包资源，README补充启动说明；既有模型、预处理和科研参数未修改。suite.py历史覆盖保护为前一轮保留改动。

## 实际验证

- 完整pytest：45 passed, 1 skipped in 29.07s；见pytest.txt/XML。唯一skip为历史硬编码计划路径，新恢复计划由专项测试覆盖。
- 随后新增计划篡改/已有输出拒绝与进程停止所有权测试；操作台最终专项：7 passed in 0.81s，见api-tests-final.txt。新增两项后未重复运行全部科研测试。
- 实际Console.launch执行S01 V2/A2四次更新，状态complete，loss及梯度记录完整；仅validation_random25/validation_outside14，未执行最终测试评价。见smoke.json，原运行产物位于data/console与results/rq1-console-smoke-*。
- 访问令牌401、文件路径越界/符号链接、任意命令拒绝、单任务互斥、任务持久状态/中断标记、正式单项输入确认、计划SHA、已有输出保护、只停止本服务拥有进程组均有实测覆盖。
- node --check、Python编译及git diff --check通过。
- 托管浏览器预览尝试失败：relay could not open browser port 4173: timed out waiting for relay readiness；sites-preview status显示stopped。未绕过预览基础设施，浏览器视觉/点击QA未验证。

## 运行与限制

在项目根目录执行 `.venv/bin/python -m eeg_transfer.console`，浏览器访问http://127.0.0.1:8765并输入终端令牌。操作台默认仅本机监听。朋友GPU主机运行相同服务后，可通过SSH端口转发访问；训练在该主机执行，云端不自动获得远程CUDA设备。

本轮未连接朋友GPU、未启动任何正式RQ1实验、未部署公开托管站点。服务面向单用户单主机，非公网多租户服务；运行边界和中断处理详见docs/CONSOLE.md。不支持精确续训，重启服务不会自动恢复训练。
