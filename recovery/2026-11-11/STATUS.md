# 2026-11-11：操作指南核对与官方原始数据LFS入库

先快进同步远端86045db/cb5c5a4，保留11月10日另一实例的403/缺数据记录和已完善的详细操作指南。本实例网络unrestricted/enforced，来源页面实际可访问。按用户本次明确要求，将官方未修改原始数据纳入Git LFS；不沿用旧“所有数据不入Git”的限制。派生NPZ、归一化、大计划、模型和任务日志继续忽略。

本实例保留此前官方下载的18份MAT（9人T/E），总779,873,919字节。逐文件核对大小及SHA256，18/18与历史官方记录一致；未重复下载已有完整文件，未修改信号字节。来源/许可页已在线重新获取，HTML仅规范行尾空白，原响应和保存文件哈希另记。署名、CC BY-ND 4.0链接和许可方见docs/DATASET.md。

Git LFS客户端3.6.1：先上传1个对象验证可用，再将18个SHA对象显式上传至origin；两次命令均退出0。本次Git条目为LFS指针，不是仅提交URL，实际数据对象已上传。Git推送和远程取回验证在提交后执行，最终以终端实际结果为准。

实际校验：scripts/verify_dataset.py输出18/18 files verified。完整pytest为47 passed, 1 skipped in 36.88s；唯一skip仍为旧历史大计划路径，新恢复清单测试实际通过。派生数据/计划/任务状态的check-ignore通过。没有正式训练或新的科学结论。

交付：docs/CONSOLE.md详细指南补充LFS政策与本轮状态；docs/DATASET.md给出Windows/Linux获取、SHA校验、预处理和许可；scripts/verify_dataset.py不依赖第三方包；README增加当前入口。历史报告中的“数据未入Git”“403”均保留为当时事实，不能误当本实例当前状态。
