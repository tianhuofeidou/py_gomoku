package com.example.gomoku

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

class GomokuViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(GameUiState())
    val state: StateFlow<GameUiState> = _state.asStateFlow()

    private val _memory = MutableStateFlow(MemoryUiState())
    val memory: StateFlow<MemoryUiState> = _memory.asStateFlow()

    private var aiJob: Job? = null
    private var timerJob: Job? = null
    // 与桌面 play.py 的 TEMP_LEVELS 保持一致；Android 端直接内置，避免每次启动多一次 Python 解析。
    private val gearLevels = listOf(
        GearLevel("轻快", 80, "约 0.3s/手"),
        GearLevel("标准", 160, "约 3s/手（复杂局面可达 17s）"),
        GearLevel("认真", 240, "约 15s/手（复杂局面可达 97s）"),
    )

    private val prefs = application.getSharedPreferences("gomoku", Context.MODE_PRIVATE)

    // Chaquopy 调用串行化：取消 Kotlin 协程无法中断已进入 Python 的计算，
    // 用单线程队列保证“AI 思考中改开新局/悔棋”不会让两个 Python 调用交错。
    private val pythonExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "gomoku-python") }
    private val pythonDispatcher = pythonExecutor.asCoroutineDispatcher()

    init {
        viewModelScope.launch {
            try {
                val home = getApplication<Application>().filesDir.absolutePath
                val savedGear = prefs.getInt("gear_t0", 160)
                val initState = withContext(pythonDispatcher) {
                    PythonBridge.init(home)
                    PythonBridge.setGear(savedGear)
                    PythonBridge.state().toGameUiState().withGearLevels()
                }
                _state.value = initState
                refreshMemoryInternal()
                maybeRunAi()
            } catch (e: Exception) {
                fail(e, "初始化失败")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        timerJob?.cancel()
        pythonExecutor.shutdown()
    }

    fun clearError() {
        _state.value = _state.value.copy(error = null)
    }

    fun setGear(t0: Int) {
        viewModelScope.launch {
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.setGear(t0) }
                if (obj.optBoolean("ok", false)) {
                    prefs.edit().putInt("gear_t0", t0).apply()
                    _state.value = (obj.optJSONObject("state")?.toGameUiState() ?: _state.value)
                        .withGearLevels()
                } else {
                    _state.value = _state.value.copy(error = obj.optString("error", "切换棋力失败"))
                }
            } catch (e: Exception) {
                fail(e, "切换棋力失败")
            }
        }
    }

    fun newGame(humanBlack: Boolean) {
        aiJob?.cancel()
        aiJob = null
        timerJob?.cancel()
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true, lastAiSeconds = null, thinkingSeconds = 0.0)
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.newGame(humanBlack) }
                _state.value = (obj.optJSONObject("state")?.toGameUiState() ?: _state.value.copy(busy = false))
                    .withGearLevels()
                refreshMemoryInternal()
                maybeRunAi()
            } catch (e: Exception) {
                fail(e, "开新局失败")
            }
        }
    }

    fun humanMove(r: Int, c: Int) {
        val s = _state.value
        if (s.over || s.thinking || s.busy || s.turn != "human") return
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true)
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.humanMove(r, c) }
                if (obj.optBoolean("ok", false)) {
                    _state.value = (obj.optJSONObject("state")?.toGameUiState() ?: _state.value.copy(busy = false))
                        .withGearLevels()
                    maybeRunAi()
                } else {
                    _state.value = _state.value.copy(
                        busy = false,
                        error = obj.optString("error", "落子失败"),
                    )
                }
            } catch (e: Exception) {
                fail(e, "落子失败")
            }
        }
    }

    fun undo() {
        aiJob?.cancel()
        aiJob = null
        timerJob?.cancel()
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true, lastAiSeconds = null)
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.undo() }
                _state.value = (obj.optJSONObject("state")?.toGameUiState() ?: _state.value.copy(busy = false))
                    .withGearLevels()
                refreshMemoryInternal()
            } catch (e: Exception) {
                fail(e, "悔棋失败")
            }
        }
    }

    fun resign() {
        aiJob?.cancel()
        aiJob = null
        timerJob?.cancel()
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true, lastAiSeconds = null)
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.resign() }
                _state.value = (obj.optJSONObject("state")?.toGameUiState() ?: _state.value.copy(busy = false))
                    .withGearLevels()
                refreshMemoryInternal()
            } catch (e: Exception) {
                fail(e, "认输失败")
            }
        }
    }

    fun draw() {
        aiJob?.cancel()
        aiJob = null
        timerJob?.cancel()
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true, lastAiSeconds = null)
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.draw() }
                _state.value = (obj.optJSONObject("state")?.toGameUiState() ?: _state.value.copy(busy = false))
                    .withGearLevels()
                refreshMemoryInternal()
            } catch (e: Exception) {
                fail(e, "求和失败")
            }
        }
    }

    fun refreshMemory() {
        viewModelScope.launch { refreshMemoryInternal() }
    }

    private suspend fun refreshMemoryInternal() {
        try {
            val obj = withContext(pythonDispatcher) { PythonBridge.memory() }
            _memory.value = obj.toMemoryUiState()
        } catch (e: Exception) {
            // 记忆页失败不阻塞下棋，保留旧数据。
        }
    }

    private fun maybeRunAi() {
        val s = _state.value
        if (s.over || s.turn != "ai" || s.thinking) return
        aiJob?.cancel()
        aiJob = viewModelScope.launch {
            val start = System.currentTimeMillis()
            _state.value = _state.value.copy(thinking = true, thinkingSeconds = 0.0)
            timerJob?.cancel()
            timerJob = launch {
                while (isActive) {
                    delay(100)
                    _state.value = _state.value.copy(
                        thinkingSeconds = (System.currentTimeMillis() - start) / 1000.0,
                    )
                }
            }
            try {
                val obj = withContext(pythonDispatcher) { PythonBridge.aiMove() }
                val elapsed = (System.currentTimeMillis() - start) / 1000.0
                val newState = obj.optJSONObject("state")?.toGameUiState()
                _state.value = (newState ?: _state.value).copy(
                    thinking = false,
                    thinkingSeconds = elapsed,
                    lastAiSeconds = elapsed,
                ).withGearLevels()
                if (!obj.optBoolean("ok", false) && obj.has("error") && !obj.isNull("error")) {
                    _state.value = _state.value.copy(error = obj.optString("error"))
                }
                refreshMemoryInternal()
            } catch (e: Exception) {
                fail(e, "AI 思考失败")
            } finally {
                timerJob?.cancel()
                timerJob = null
            }
        }
    }

    private fun GameUiState.withGearLevels(): GameUiState =
        copy(gearLevels = this@GomokuViewModel.gearLevels)

    private fun fail(e: Exception, prefix: String) {
        timerJob?.cancel()
        timerJob = null
        _state.value = _state.value.copy(
            thinking = false,
            busy = false,
            error = "$prefix：${e.message ?: "未知错误"}",
        )
    }
}
