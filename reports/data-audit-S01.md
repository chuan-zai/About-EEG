# S01 真实数据审计

审计日期：2026-11-05。来源为 BNCI Horizon 官方公开 MAT；源文件无明确语义版本号，以本地 SHA256 固定版本。未找到独立发布的校验和，本地哈希用于复现和完整性追踪，不声称是第三方认证。

许可：官方页面列出 CC BY-ND 4.0；需署名并引用 Tangermann et al. (2012), DOI:10.3389/fnins.2012.00055。原始及派生数据均位于 Git 忽略的 data/，未分发数据。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| A01T.mat | 42806453 | 054f02e70cf9c4ada1517e9b9864f45407939c1062c6793516585c6f511d0325 |
| A01E.mat | 43772146 | 53d415f39c3d7b0c88b894d7b08d99bcdfe855ede63831d3691af1a45607fb62 |

官方链接：

- https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
- https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat

MOABB 源码 snapshot 证实 channel 顺序和 uV→V；官方说明证实250Hz、trial onset后2秒cue、4秒MI。源文档、源码和哈希见 sources/manifest.json。未安装MOABB或触发其自动下载。

**通道顺序：** Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz

原始25列中仅前22列EEG按已核对映射导入；EOG1/EOG2/EOG3完全排除。单位uV乘1e-6成为V；无额外滤波、基线减法或重参考。每trial采样范围为MAT trial-1+500 至 trial-1+1500（右端不含），1000点。

| Session | 有标签trial | 官方坏trial剔除 | 接收 | 标签0/1/2/3计数 |
|---|---:|---:|---:|---|
| T | 288 | 15 | 273 | [69, 69, 68, 67] |

T：原始各run非有限值共0；接收epoch非有限值0；常数电极trial组合0；峰值超过1mV的trial为0；范围 [-7.6513672e-05, 7.1289061e-05] V。

| E | 288 | 7 | 281 | [71, 70, 69, 71] |

E：原始各run非有限值共0；接收epoch非有限值0；常数电极trial组合0；峰值超过1mV的trial为0；范围 [-5.7080077e-05, 7.8417972e-05] V。

每Session前三个无trial的run为EOG校准，不使用；六个任务run各48trial。剔除基于官方artifact标记，与重建成绩无关。完整的run、剔除原因、类计数、原文件路径、派生文件哈希与划分键见JSON。原始标签保留为1..4元数据及0..3数组，不进入SSL事件或主要评价。

划分：split_seed=42，T内不按标签随机trial划分，训练218、验证55；E最终测试281。先划分trial，再生成窗口。训练归一化保存于run/normalization.npz，只fit训练trial。smoke测试评价预先限定2trial，用于工程验证，不能用于调参。

局限：仅S01，未下载其余8名受试者；未验证GDF文件；没有额外人工伪迹审阅；不将通过单位/范围检查视为完整生理质量认证。
