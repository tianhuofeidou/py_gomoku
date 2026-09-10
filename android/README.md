# 棋逢小鲸 Android（主仓库 android/）

棋逢小鲸（五子棋）Android 客户端：Kotlin + Jetpack Compose 负责界面，Python 引擎通过
Chaquopy 17 在 App 内运行。引擎复用桌面独立版的纯算法核心：

本工程位于桌面主仓库的 `android/` 子目录；仓库根目录的 `gomoku/` 是引擎唯一事实来源，
`app/src/main/python/gomoku/` 由同步脚本生成，不要手工修改。

```
pattern -> search -> deep_search -> engine
```

移动端不启用神经网络叶子评估/决策网络，也不加载任何第三方 Python 包；
`gomoku/tools/ga_tune.py` 是仅供 `service.load_ga_params` 导入的空实现，
因为 GA 调参链路只在桌面使用（且需要显式设置 `DSH_GOMOKU_GA=1`）。

## 界面

- 竖屏锁定，棋盘按屏幕宽度自适应。
- 顶部：标题、棋力、记忆、新局；状态卡显示手数、回合、AI 思考计时、你执哪方和当前 AI 棋力。
- 底部：悔棋、认输、求和、新局、记忆。
- 记忆页：总战绩、败因分布、胜局/败局棋谱、最近对局、经验提示。
- 不显示候选分析日志、分支分和五态强制链标签；五态判定仍由引擎内部参与选点。

## 环境

| 组件 | 位置 |
| --- | --- |
| Android Studio | `D:\Android\Studio` |
| JDK 17 | `D:\Android\jdk17` |
| Android SDK | `D:\Android\Sdk` |
| Gradle 缓存 | `D:\Android\gradle` |
| buildPython | `D:\python3.14\python.exe`（CI 用 `CHAQUOPY_BUILD_PYTHON` 覆盖） |
| 模拟器/AVD | `D:\Android\avd` |

项目已配置：

- `gradle/wrapper/gradle-wrapper.properties` 使用腾讯 Gradle 镜像（Gradle 9.1.0）。
- `settings.gradle.kts` 优先使用阿里云 Google / Maven Central / Gradle Plugin 镜像。
- `local.properties` 指向 `D:\Android\Sdk`（该文件不进入版本库；CI 使用 `ANDROID_HOME`）。
- `gradle.properties` 关闭 configuration cache：Chaquopy 17 与 Gradle 配置缓存不兼容。

## 引擎同步

`app/src/main/python/gomoku/` 由仓库根目录的脚本从桌面 `gomoku/` 同步生成，
不要手工修改，否则会与桌面引擎漂移。

```bat
cd ..
python scripts\sync_android_engine.py
python scripts\sync_android_engine.py --check
```

`gomoku/tools/ga_tune.py` 会被写成移动端占位实现；`app/src/main/python/android_api.py`
是 Android 专用文件，不在同步范围内。

## 构建

在 `android/` 目录执行：

```bat
set JAVA_HOME=D:\Android\jdk17
set ANDROID_HOME=D:\Android\Sdk
set ANDROID_SDK_ROOT=D:\Android\Sdk
set ANDROID_USER_HOME=D:\Android\android-home
set GRADLE_USER_HOME=D:\Android\gradle
set TEMP=D:\Android\temp
set TMP=D:\Android\temp
set CHAQUOPY_BUILD_PYTHON=D:\python3.14\python.exe
gradlew.bat :app:assembleDebug --no-configuration-cache
```

CI 或特殊版本可用 `-PversionName` / `-PversionCode` 覆盖，例如：
`gradlew.bat :app:assembleDebug -PversionName=1.3.3 -PversionCode=10303 --no-configuration-cache`。

产物：

```
app\build\outputs\apk\debug\app-debug.apk
```

安装到已连接设备：

```bat
D:\Android\Sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

## 数据目录

App 私有目录（运行时注入 Python 的 `GOMOKU_HOME`）保存：

- `games.json`：当前/历史对局；
- `global-memory.json`：全局战绩、败因分布、胜败棋谱。

首次启动若没有存档，会以“人类执黑”的空棋盘开始；点“新局”可以改选执白。
对局中的每一步都会落盘，杀死进程后重新打开会恢复上次局面。

## 已知限制

- 当前只产出 debug APK；release 需要配置签名后才能分发。
- 神经网络推理未在 Android 端启用，移动端只跑纯算法引擎。
- AI 计算在 Python 线程执行，界面不会卡死；手机性能远低于桌面，思考时间会更长。

## 验证记录

- v1.2.0：已在 Android 15（API 35）x86_64 模拟器上完成运行验证：
  - 应用启动、Chaquopy 初始化正常；
  - 人类执黑、执白两种开局都能落子并触发 AI 应手；
  - 中盘首次深推在模拟器上约 5 秒完成，思考期间界面保持响应；
  - 认输结算后记忆页正确写入败局，胜负统计按玩家视角显示；
  - 重装/重启后会恢复上一局棋局与结果。
- v1.2.1：使用 D 盘工具链重新打包 debug APK：
  - `versionCode=10201`、`versionName=1.2.1`，包含 arm64-v8a 与 x86_64；
  - `apksigner verify --verbose` 通过（APK Signature Scheme v2）；
  - `assets/chaquopy/app.imy` 已核对包含 `gomoku/core/search.pyc` 等同步后的引擎文件。
- v1.2.1 冒烟：已在 Android 15（API 35）x86_64 模拟器安装并启动成功；
  `topResumedActivity=com.example.gomoku/.MainActivity`，进程存活，未发现 FATAL/Chaquopy 启动错误。
- v1.2.1 尚未做人工点击走子、悔棋、认输等交互验证。
- v1.3.0：默认版本升级后重新构建 debug APK：
  - `versionCode=10300`、`versionName=1.3.0`，`aapt2` 确认包含 arm64-v8a 与 x86_64；
  - `apksigner verify --verbose` 通过（APK Signature Scheme v2）；
  - 模拟器安装并启动成功，`topResumedActivity=com.example.gomoku/.MainActivity`，未发现 FATAL 错误；
  - 本地构建 APK SHA-256：`4FA37A10C8F0AA3CBA457AC086C955A4EF029315201F8D78CDB88DB48C90CAFF`（正式发布以 GitHub Actions 构建产物为准）。
- v1.3.0 尚未做人工点击走子、悔棋、认输等交互验证。
- v1.3.3：GitHub Actions 构建并发布 debug APK（`versionCode=10303`、`versionName=1.3.3`，arm64-v8a + x86_64），Release 同时包含 Windows EXE、macOS zip 和 `SHA256SUMS.txt`。
- v1.3.3 仍未做人工点击走子、悔棋、认输等交互验证；APK 为 debug 包，EXE 未签名。