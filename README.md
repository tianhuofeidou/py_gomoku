# Gomoku · 五子棋独立版（python 包已从插件抽出）

把原 `dsh-gomoku` 插件里的算法包（原 `python/`）抽出来，重命名为 **`gomoku`**，
并补上独立的游玩界面与对局状态机/记忆，作为 **纯 Python 独立项目** 使用。
（不再依赖 DeepSeek Harness / Node 插件层。）

---

## 快速开始

```bash
# 图形界面（tkinter，零第三方依赖，推荐）
python play.py

# 命令行对战（终端也能玩）
python play_cli.py

# 机机对战（两个 AI 自对弈，可导棋谱）
python -m gomoku.cli.vs_mode --max-moves 60 --out game.json
```

核心引擎**纯算法零依赖**（仅标准库）。`tkinter` 为 Python 自带。

### 可选强化棋力（装 torch/numpy）

```bash
pip install numpy torch
```

未安装时引擎自动回退纯算法，不会崩（`deep_search._nn_score` 与决策网络均有 try/except 回退）。

---

## 功能（对照插件尽量还原）

| 功能 | 说明 |
|---|---|
| **三种模式** | 人机 `ai` / 人人 `pvp` / 机机 `vs` |
| **换边** | 人机可人类执黑先手 或 白后手 |
| **引擎落子** | 纯算法 + 可选决策网络；落子后展示推荐全候选（ranked） |
| **悔棋 / 认输 / 求和** | `Game.exec/undo`，与插件语义一致 |
| **新对局** | `Game.reset`，可带模式/换边/难度 |
| **对局状态机** | 落子校验、胜负（成五）、满盘和棋、轮次、人/机方 |
| **持久化** | 对局存档到 `~/.gomoku/games.json`（环境变量 `GOMOKU_HOME` 可覆盖） |
| **全局记忆** | `totals`（胜负平）/ `lossByType`（活四/冲四/跳四/双活三/其他）/ `dualMem` / `badLines` / `goodLines`，含 8 变换对称归一化 |
| **记忆提示** | GUI「查看记忆」/ CLI 对局结束后打印 memoryHint |
| **大模型辅助** | `aiMode='llm'` 预留：本独立版未接 LLM，界面用 `engine` 模式 |

> 注：插件专属的「LLM 决策」（`lib/server/decision.js`）与 React 前端（`lib/ui`）不在此独立版内；
> 大模型下棋如需接入，可在 `service.ai_move` 处扩展为大模型推荐 + 引擎校验。

---

## 目录结构（独立版）

```
gomoku/
├── paths.py       存储路径（~/.gomoku，可 GOMOKU_HOME 覆盖）
├── game.py        对局状态机（复刻插件 state.js）
├── memory.py      全局记忆（复刻插件 memory.js，含对称归一化）
├── service.py     引擎服务 + 规范化 AI 落子（含开局处理）
├── session.py     对局会话（GUI/CLI 共用：状态机 + 同步引擎）
├── core/          算法核心（engine / search / deep_search / pattern / evaluate / utils）
├── nn/            神经网络（decision_net / model / features / 训练脚本；权重 .pt 不入库）
├── adapter/       原 stdio 引擎接口（server / ai_move）
├── cli/           命令行入口（vs_mode 机机对战 / decide_cli）
├── tools/         训练与调参工具（ga_tune / katago_convert 等）
├── data/          GA 最优参数 + 训练样本 json
└── tests/         pytest
play.py            tkinter 图形界面（根目录）
play_cli.py        命令行对战（根目录）
requirements.txt   依赖说明
legacy_pseudocode.py  旧根目录扫描规则草稿（已改名保留）
```

---

## 说明

- 原 `python/` → `gomoku/`，所有 `from python.*` / `import python.*` / `python/nn/*.pt` 路径已同步改为 `gomoku.*`。
- 引擎核心在 `gomoku/core`，`service.ai_move` 与插件 `adapter/server.py` 的开局逻辑一致（黑第一手天元、白第二手取开局候选）。
- 训练/数据（`katago_data`、`nn_samples*.json` 等）仍在，独立版运行不需要，可按需清理。
- **模型权重 `.pt` 不入库**：克隆后引擎默认走纯算法（缺权重自动回退）。需要神经网络增强时，把本地训练好的 `value.pt` / `decision.pt` 放回 `gomoku/nn/` 即可（或重新训练）。
