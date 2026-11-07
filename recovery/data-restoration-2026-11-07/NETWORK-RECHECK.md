# 公网设置后重新核验

2026-11-07 08:37 UTC，用户告知已开放公网后，使用新请求重新核验。

- 实时环境状态：spec_revision=1，observed_spec_revision=1，observations_current=true。
- 当前实际策略仍为restricted/enforced，allowed_hosts为空，仅启用package_managers预设。
- 官方bnci-horizon-2020.eu及来源快照中的lampx.tugraz.at均不在effective_allowed_hosts。
- 两个实际MAT URL均返回CONNECT tunnel failed, response 403，未下载数据。
- 原始提示词已阅读，后续阶段的重建主终点与T/E隔离约束继续适用；下载已获授权，全量训练仍未授权启动。

这表示当前运行实例尚未体现用户所述公网设置，不能据此断定设置未在其他位置保存。需要让设置应用到此运行实例后再用新连接核验。未重复运行无变更的测试，也未将此前23通过/15跳过当成本轮新结果。
