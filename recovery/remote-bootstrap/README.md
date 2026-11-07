# About EEG — BNCI2014-001

目标数据集为 **BNCI2014-001**（官网编号 **001-2014**），亦称
**BCI Competition IV dataset 2a**。包含 9 名受试者、每人训练 T 和评估 E
两次记录，共 18 个 MATLAB 文件；22 个 EEG 通道和 3 个 EOG 通道，250 Hz，
四类运动想象：左手、右手、双脚、舌头。

## 下载

需要 Python 3.8+，无第三方依赖：

```bash
python3 scripts/download_bnci001.py
```

文件保存到 `data/BNCI2014-001/`，保持原始文件不变。脚本检查 MATLAB 文件头、
最低文件大小，并为全部 18 个文件生成 `manifest.json`（字节数、来源 URL、SHA-256）。
已有文件校验通过后跳过；下载失败不会留下冒充完整文件的临时文件。
这些是本地完整性记录，并非发布方提供的独立校验和。

## 来源与许可

- [官网数据集列表，第一项](https://bnci-horizon-2020.eu/database/data-sets)
- [原始实验说明 PDF](https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf)
- [实际下载目录](https://lampx.tugraz.at/~bci/database/001-2014/)
- 数据许可：[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)
- 数据提供方：Institute for Knowledge Discovery, Graz University of Technology。

使用和再分发原始数据时应保留来源与署名，遵循数据许可并引用官网关联论文。
本仓库的下载脚本不改变原始数据。官网 PDF 描述原竞赛的 GDF 格式；
本脚本获取的是 BNCI 官网提供的 MATLAB 版本。

## 当前数据状态

原始数据尚未下载：云环境网络白名单阻止访问官网及实际下载主机。
需要放行 `bnci-horizon-2020.eu` 和 `lampx.tugraz.at` 后运行上述命令。
