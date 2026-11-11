# BNCI2014_001：数据获取、署名与本地准备

2026-11-11起，本仓库按用户要求通过 **Git LFS** 保存18份未修改的官方MAT文件：S01–S09，每人T/E两个Session。总大小779,873,919字节（约744MiB）。这些原始文件此前已从官方URL下载，本轮逐一重新核验，全部与历史官方下载的SHA256相同，不重复下载相同文件。

## 1. 来源和许可

- 数据集：BNCI2014_001，即BCI Competition IV dataset 2a / Four class motor imagery。
- 官方发布页：https://bnci-horizon-2020.eu/database/data-sets
- 文件URL：https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat （其他编号同样规则）。官方地址可能重定向至 https://lampx.tugraz.at/~bci/database/001-2014/ 。
- 许可方：Institute for Knowledge Discovery, Graz University of Technology。
- 官方许可：**Creative Commons Attribution–NoDerivatives 4.0 International (CC BY-ND 4.0)**，https://creativecommons.org/licenses/by-nd/4.0/ 。数据遵循该许可，不因放入代码仓库而改用代码许可。
- 引用：Tangermann M. et al. (2012), *Review of the BCI Competition IV*, Frontiers in Neuroscience 6:55. https://doi.org/10.3389/fnins.2012.00055 。使用时还应保留官方页面中的相关引用。
- 本镜像不修改原始MAT字节，不代表原作者认可本项目。预处理NPZ、训练归一化、大计划和模型权重不在本次数据提交范围；处理后数据的对外分发须另行核对ND许可。

本轮官方页面快照、18文件URL/字节大小/SHA256和核验结果在 `recovery/2026-11-11/`。科学来源及历史审计仍保留在 `reports/sources/`、`reports/phase3/data/`。

## 2. 推荐：Git LFS下载

先安装Git与Git LFS。Git for Windows通常包含LFS，可用 `git lfs version` 检查；若找不到，安装 https://git-lfs.com/ 提供的客户端。

新获取项目：

```bat
git lfs install
git clone --branch work https://github.com/chuan-zai/About-EEG.git
cd About-EEG
git lfs pull origin work
```

已有E盘项目，在自己的项目目录执行：

```bat
cd /d "E:\About-EEG"
git lfs install
git pull --ff-only origin work
git lfs pull origin work
git lfs ls-files
```

`git lfs install`只需初始化一次。LFS下载受GitHub网络及仓库LFS可用额度影响。请预留至少约3GB磁盘空间用于LFS缓存、工作文件、预处理和计划，Python依赖另算。若本地已有未跟踪同名MAT导致更新拒绝，先保留本地文件并核验，勿强制覆盖或删除研究数据。

**不要把约130字节的LFS指针误认为MAT。** 指针文本开头为 `version https://git-lfs.github.com/spec/v1`。GitHub的源码ZIP是否包含LFS实物取决于仓库归档设置，本项目没有假定设置已开启；下载ZIP后若发现指针，请使用LFS或官方来源补齐。

## 3. 验证18个文件

在项目根目录运行（Windows无需先激活环境）：

```bat
.venv\Scripts\python.exe scripts\verify_dataset.py
```

Linux/WSL2：

```bash
.venv/bin/python scripts/verify_dataset.py
```

验证器仅使用Python标准库，逐一检查文件大小与SHA256，缺失、指针或不匹配会以非零退出码结束。验证成功后应显示 `18/18 files verified`。此检查不训练、不改文件。

## 4. 原始MAT还不能直接用于正式训练

1. 启动操作台，进入“数据与审计”。只拿到MAT时，“原始数据”已存在而“派生NPZ”仍缺失，这是正常情况。
2. 点击S01“下载并接入”。程序复用已有完整MAT，核验SHA，再创建规范 `data/processed/S01.npz` 和训练归一化，不重新下载已有文件。
3. 查看审计、运行测试及S01四步冒烟。
4. 逐个完成S02–S09；九人数据齐备后才生成新的配对计划。

不要直接在新机器使用历史清单里的大计划路径；这些gzip计划并未随LFS数据提交。网页“生成新配对计划”会创建新的清单和27份计划包，但不启动正式训练。

## 5. LFS网络受限时

从官方来源获取即可，不需要先解决GitHub LFS：操作台“下载并接入”使用官方URL，或使用 `scripts/download_bnci001.py` 的文档化参数。脚本默认目录不是操作台的raw目录，需显式指定：

```bat
.venv\Scripts\python.exe scripts\download_bnci001.py --output data\raw
.venv\Scripts\python.exe scripts\verify_dataset.py
```

来源限制、SHA不匹配、失败恢复见 [操作台完整指南](CONSOLE.md) 第6、14节。不要关闭TLS验证，也不要把不明镜像文件直接当作同一数据集。

## 6. 本次范围

原始记录包含25列（22EEG+3EOG）。模型仅用22EEG，250Hz，uV转换为V，cue为试次起点+2秒，截取[2,6)秒。先按trial划分T训练/验证、E最终测试，训练集独立拟合归一化。原始文件保留EOG和官方元数据以保持原样，加载时明确剔除EOG。

本次数据入库不代表正式实验已经运行，没有新的负迁移结论，也不启动270项正式训练。
