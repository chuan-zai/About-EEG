# 当前恢复状态

本提交接续远端原有下载脚本，恢复附件源码快照及其历史记录；没有恢复缺失的离线Git历史，也没有重写为原提交。附件声明的源提交为1f670d36b3fa9d1d97c5a14f2904375bcdece075，逐文件SHA清单见HANDOFF_MANIFEST.json。原远端README另存recovery/remote-bootstrap/README.md，原下载脚本保留。

README.md和reports/phase3中的38通过、ready=true及绝对路径是原环境历史记录，不代表当前环境已完成核验。恢复后实际两次pytest均为23通过、15跳过、0失败；最近一次耗时4.80秒。完整结果见recovery/data-restoration-2026-11-07/pytest.txt。

当前目标是8/22ch混合自监督训练对完整22ch重建泛化与区域误差的影响。不预设负迁移；T训练/验证、E最终测试；保留V1，正式统一V2。不运行全量正式训练或RQ2/RQ3。

仍缺真实EEG、历史权重、正式大计划及原Git bundle。官方与Graz地址在当前实例仍返回CONNECT 403，详见recovery/data-restoration-2026-11-07/NETWORK-RECHECK.md。数据恢复命令、路径迁移说明和下一步见同目录PROGRESS.md。data/、虚拟环境、模型二进制和正式运行目录均由Git忽略。
