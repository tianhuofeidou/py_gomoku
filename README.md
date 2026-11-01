# 棋逢小鲸

棋逢小鲸是一个本地运行的 15×15 五子棋程序。黑方先行，不设禁手，连续五子及以上获胜。正式引擎使用纯 Python、纯算法实现，正常对弈不需要联网、Torch、NumPy 或神经网络；桌面界面使用 Python 标准库 tkinter。

本 README 面向使用者和集成者，说明入口、运行方式和能力边界。候选生成、温度搜索、叶子评分、五态判定和缓存续推的完整设计见 [算法设计文档](算法设计文档.md)。协作规则和长期不变量见 [AGENTS.md](AGENTS.md)。

## 项目入口

| 入口 | 用途 | 说明 |
|---|---|---|
| `play.py` | 桌面图形界面 | 人机、人人、逐步机机和棋谱复盘 |
| `play_cli.py` | 命令行对弈 | 玩家执黑、AI 执白的交互模式 |
| `gomoku.cli.vs_mode` | 自动机机对弈 | 保存棋谱，适合本地观察和批量运行 |
| `gomoku.protocol` | Gomocup/Piskvork 管道协议 | 由管理器驱动，与外部引擎对战 |
| `gomoku.tools.match` | 自建对战台 | 驱动两个协议引擎并记录结果 |
| `android/` | Android 客户端 | 通过 Chaquopy 运行同一套纯算法引擎 |
| `gomoku.tools.loss_book` | 败局回归库 | 收录、定位和复测旧败局 |

源码版本和已发布 EXE、APK、macOS 包是两个边界。源码修改不会自动进入已有下载资产；正式发布资产以 GitHub Release 及其 `SHA256SUMS.txt` 为准。

## 快速上手

本机开发和测试使用 Python 3.14。Windows 可直接使用项目约定的解释器：

```powershell
D:\python3.14\python.exe play.py
```

也可以使用当前环境中已经指向 Python 3.14 的 `python` 命令：

```bash
python play.py
python play_cli.py
python -m gomoku.cli.vs_mode --max-moves 60 --out game.json
```

图形界面默认进入人机模式，玩家执黑。命令行输入 `H8` 或 `8,8` 落子，输入 `u` 悔棋，输入 `q` 退出。AI 思考在后台线程进行，界面仍可刷新；思考期间不接受新的棋盘操作。

每次 AI 落子后，界面和命令行可以显示决策来源、耗时、叶子评估状态和候选结果。候选五态只描述本次有限搜索树，不是全盘穷举证明。

## 对弈功能

### 人机和人人

人机模式支持玩家执黑或执白。人人模式由玩家轮流落子，也可以请求 AI 辅助分析；辅助分析只展示候选，不自动落子，不改变棋盘或后续决策。

### 机机

桌面界面可以逐步让当前一方 AI 落子；需要连续完成对局时使用：

```bash
python -m gomoku.cli.vs_mode --max-moves 60 --out game.json
```

### 棋谱和记忆

棋谱复盘可以读取记录目录或 JSON，逐手回放并建立独立试走分支。对局记录、胜负记忆和败局库默认保存在 `~/.gomoku/`，可用 `GOMOKU_HOME` 修改目录。悔棋会同步撤销棋盘、搜索状态和已记账的终局。

## 算法概览

正式决策链为：

```text
棋盘
  -> 棋型识别
  -> 战术候选或普通候选
  -> 温度驱动的有限深推
  -> 叶子评分与五态汇总
  -> 状态优先、同档按分排序
```

算法有四个核心层：

| 层 | 作用 |
|---|---|
| 棋型层 | 从横、竖和两条斜线识别成五、活四、冲四、活三和潜力结构 |
| 搜索层 | 维护黑白状态表，优先生成直接战术、防守点和普通候选 |
| 深推层 | 用温度、候选权重和路径剪枝展开后续变化 |
| 决策层 | 将叶子分和五态分开汇总，优先选择五态更高的候选 |

算法的关键边界如下：

- `reason` 表示候选生成是否命中战术直接取点分支；它不是五态，也不是全盘原始 0-4 扫描；
- 实际成五优先结束；温度耗尽时，只允许下一方候选生成命中战术直接取点分支继续；普通打分候选不能借权重 100 续跑；
- 温度允许在正温度下扣到负数，但不补温、不清欠温；黑白两方使用相同规则；
- `+1/-1` 是本次展开中的硬必胜/硬必败，`+2/-2` 是直接子孙层的软证据，`0` 是本次搜索未确定；
- 叶子分是局面评价，不是胜率；五态排序优先于同档候选分；
- 真实落子后可以沿旧树换根复用局面结果和欠温叶子，但参数、权重、棋型规则或换根不匹配时必须重建。

完整公式、状态转换、候选宽度、缓存身份和回滚边界见 [算法设计文档](算法设计文档.md)。

## 开局库

开局库按完整棋盘和行棋方匹配人类整理的可靠推荐，使用八种旋转和镜像处理对称局面。前三次本方落子命中可靠推荐时可以接管；直接成五和必须堵五始终优先。提前出谱后立即回到正式候选和深推。

本方第四次落子起，命中的谱点只作为一个普通根候选加入，不改变递归宽度、原始权重或五态排序。研究线不等于必胜证明，谱着也不自动获得五态标签。

正式数据位于 `gomoku/data/classical_opening_recommendations.json` 和 `gomoku/core/opening_book_data.py`。原始图谱、外部引擎和模型不进入正式运行路径。

## 协议与对战台

启动协议入口：

```bash
python -m gomoku.protocol
```

协议坐标使用零基 `(X,Y)`，对应核心坐标 `(r,c)=(Y,X)`。协议入口只做命令解析、坐标换算和输出，选点委托核心服务；诊断信息不能写入标准输出。

自建对战台示例：

```bash
python -m gomoku.tools.match \
  --black "python -m gomoku.protocol --t0 80" \
  --white "path/to/pbrain-carbon.exe" \
  --openings 1 --parallel 1 --json comparison.json
```

对战比较应固定对手版本、双方配置、均衡开局集、黑白交换和计时设置，并保存逐局记录。胜率、置信区间和 Elo 差只描述本次对手与样本，不代表全局棋力或官方排名。搜索温度不是硬性单手时限，也不是全局节点上限；复杂局面可能超过管理器的比赛限时。

## 败局回归

败局库用于保留旧败局、记录来源和复测历史，不会自动修改正式算法参数。基本流程是：

```text
收录或导入败局
  -> 规则检查或人工指定关键手
  -> 在新版本恢复多个起点复测
  -> 全部起点完成且通过后归档
```

常用命令：

```bash
python -m gomoku.tools.loss_book list
python -m gomoku.tools.loss_book import --json games.json --ai-side a
python -m gomoku.tools.loss_book locate --id 1a2b3c4d
python -m gomoku.tools.loss_book retest --id 1a2b3c4d
python -m gomoku.tools.loss_book show --id 1a2b3c4d
python -m gomoku.tools.loss_book archive
```

规则定位只报告可直接核验的错过成五、漏堵唯一成五等局部事实；可选外部教师只用于离线比较。有限教师变化、单条主变化和“未发现败手”都不能当作全盘证明。预算不足、缓存缺项、超时和旧棋谱用尽必须保留为未完成或失败。

## 开发与验证

主要目录：

```text
gomoku/core/        棋型、候选、深推和决策链
gomoku/tests/       Python 回归测试
gomoku/tools/       对战、败局和离线实验工具
android/            Android 客户端
scripts/            引擎同步和构建辅助脚本
play.py             图形界面入口
play_cli.py         命令行入口
```

完整 Python 回归：

```bash
D:\python3.14\python.exe -B -m unittest discover -s gomoku/tests -v
```

深推、叶子和候选排序的定向回归：

```bash
D:\python3.14\python.exe -B -m unittest \
  gomoku.tests.test_leaf_value \
  gomoku.tests.test_deep_sync \
  gomoku.tests.test_rank_candidates -v
```

缓存和温度续推相关变更还应覆盖 `test_search_cache.py`、`test_search_cache_reuse.py`、`test_zero_cost_continuation.py` 和 `test_tactical_temperature.py`。测试通过只说明对应路径满足当前断言，不说明所有棋局和所有深度都已穷举。

修改根目录引擎后，Android 内嵌引擎应由同步脚本重新生成并检查：

```bash
D:\python3.14\python.exe scripts/sync_android_engine.py --check
```

## 发布与依赖边界

正式运行路径保持纯 Python 和标准库能力。Torch、NumPy、模型、教师程序、训练数据和实验目录属于本地实验或预留能力，不会因为文件存在就自动启用。

正式发布资产由 GitHub Actions 构建并上传到 Release：Windows EXE、Android debug APK 和 macOS zip。下载资产后必须用同一 Release 的 `SHA256SUMS.txt` 校验；源码更新、本地构建成功或单个平台构建成功，都不等于完整 Release 已更新。

离线调参和教师实验只写入独立实验目录，不自动替换正式参数。实验报告应记录输入局面、源码身份、解释器、搜索设置、开关、预算和停止原因；有限搜索结果不能写成全局证明。
