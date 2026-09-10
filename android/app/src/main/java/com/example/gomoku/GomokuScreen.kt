package com.example.gomoku

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlin.math.min

@Composable
fun GomokuApp(viewModel: GomokuViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val memory by viewModel.memory.collectAsStateWithLifecycle()
    var screen by rememberSaveable { mutableStateOf("board") }

    if (screen == "memory") {
        MemoryScreen(
            memory = memory,
            onBack = { screen = "board" },
            onRefresh = viewModel::refreshMemory,
        )
    } else {
        BoardScreen(
            state = state,
            onTap = viewModel::humanMove,
            onUndo = viewModel::undo,
            onResign = viewModel::resign,
            onDraw = viewModel::draw,
            onNewGame = viewModel::newGame,
            onSetGear = viewModel::setGear,
            onMemory = {
                screen = "memory"
                viewModel.refreshMemory()
            },
            onClearError = viewModel::clearError,
        )
    }
}

@Composable
private fun BoardScreen(
    state: GameUiState,
    onTap: (Int, Int) -> Unit,
    onUndo: () -> Unit,
    onResign: () -> Unit,
    onDraw: () -> Unit,
    onNewGame: (Boolean) -> Unit,
    onSetGear: (Int) -> Unit,
    onMemory: () -> Unit,
    onClearError: () -> Unit,
) {
    var showNewGame by remember { mutableStateOf(false) }
    var showGear by remember { mutableStateOf(false) }
    var showResign by remember { mutableStateOf(false) }
    var showDraw by remember { mutableStateOf(false) }
    var resultDismissed by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(state.error) {
        val err = state.error
        if (!err.isNullOrBlank()) {
            snackbarHostState.showSnackbar(err)
            onClearError()
        }
    }
    LaunchedEffect(state.over) {
        if (!state.over) resultDismissed = false
    }

    Scaffold(snackbarHost = { SnackbarHost(snackbarHostState) }) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            HeaderRow(
                enabled = !state.thinking && !state.busy,
                onGear = { showGear = true },
                onMemory = onMemory,
                onNewGame = { showNewGame = true },
            )
            Spacer(Modifier.height(6.dp))
            StatusCard(state)
            Spacer(Modifier.height(8.dp))
            BoxWithConstraints(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center,
            ) {
                val side = min(maxWidth.value, maxHeight.value)
                BoardCanvas(
                    board = state.board,
                    last = state.last,
                    enabled = !state.over && !state.thinking && !state.busy && state.turn == "human",
                    onTap = onTap,
                    modifier = Modifier.size(side.dp),
                )
            }
            Spacer(Modifier.height(6.dp))
            Text(
                text = state.summary,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            BottomButtons(
                state = state,
                onUndo = onUndo,
                onResign = { showResign = true },
                onDraw = { showDraw = true },
                onNewGame = { showNewGame = true },
                onMemory = onMemory,
            )
        }
    }

    if (showNewGame) {
        NewGameDialog(
            onDismiss = { showNewGame = false },
            onStart = { humanBlack ->
                showNewGame = false
                onNewGame(humanBlack)
            },
        )
    }
    if (showGear) {
        GearDialog(
            state = state,
            onSelect = { t0 ->
                showGear = false
                onSetGear(t0)
            },
            onDismiss = { showGear = false },
        )
    }
    if (showResign) {
        ConfirmDialog(
            title = "确认认输？",
            text = "认输后本局判负，并记入对手的胜局棋谱。",
            confirmText = "认输",
            onConfirm = {
                showResign = false
                onResign()
            },
            onDismiss = { showResign = false },
        )
    }
    if (showDraw) {
        ConfirmDialog(
            title = "确认求和？",
            text = "求和后本局记为和棋，双方都不计入胜负。",
            confirmText = "求和",
            onConfirm = {
                showDraw = false
                onDraw()
            },
            onDismiss = { showDraw = false },
        )
    }
    if (state.over && !resultDismissed) {
        ResultDialog(
            result = state.result,
            onNewGame = {
                resultDismissed = true
                showNewGame = true
            },
            onDismiss = { resultDismissed = true },
        )
    }
}

@Composable
private fun HeaderRow(
    enabled: Boolean,
    onGear: () -> Unit,
    onMemory: () -> Unit,
    onNewGame: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "棋逢小鲸",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
            modifier = Modifier.weight(1f),
        )
        TextButton(onClick = onGear, enabled = enabled) { Text("棋力") }
        TextButton(onClick = onMemory) { Text("记忆") }
        TextButton(onClick = onNewGame, enabled = enabled) { Text("新局") }
    }
}

@Composable
private fun StatusCard(state: GameUiState) {
    Card(Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("第 ${state.moveCount} 手", style = MaterialTheme.typography.titleMedium)
                Text(
                    text = when {
                        state.over -> resultText(state.result)
                        state.busy -> "处理中…"
                        state.thinking -> "AI 思考中… %.1fs".format(state.thinkingSeconds)
                        state.turn == "human" -> "轮到你落子"
                        else -> "等待 AI"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
                if (!state.thinking && state.lastAiSeconds != null) {
                    Text(
                        text = "AI 用时 %.1fs".format(state.lastAiSeconds),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Column(horizontalAlignment = Alignment.End) {
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = MaterialTheme.colorScheme.secondaryContainer,
                    contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
                ) {
                    Text(
                        text = "你执${if (state.human == 1) "黑" else "白"}",
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        style = MaterialTheme.typography.labelMedium,
                    )
                }
                Spacer(Modifier.height(4.dp))
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = MaterialTheme.colorScheme.tertiaryContainer,
                    contentColor = MaterialTheme.colorScheme.onTertiaryContainer,
                ) {
                    Text(
                        text = "AI ${gearName(state)}",
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        style = MaterialTheme.typography.labelMedium,
                    )
                }
            }
        }
    }
}

@Composable
private fun BottomButtons(
    state: GameUiState,
    onUndo: () -> Unit,
    onResign: () -> Unit,
    onDraw: () -> Unit,
    onNewGame: () -> Unit,
    onMemory: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
    ) {
        TextButton(onClick = onUndo, enabled = state.canUndo && !state.thinking && !state.busy) { Text("悔棋") }
        TextButton(onClick = onResign, enabled = !state.over && !state.thinking && !state.busy) { Text("认输") }
        TextButton(onClick = onDraw, enabled = !state.over && !state.thinking && !state.busy) { Text("求和") }
        TextButton(onClick = onNewGame, enabled = !state.thinking && !state.busy) { Text("新局") }
        TextButton(onClick = onMemory) { Text("记忆") }
    }
}

@Composable
private fun NewGameDialog(onDismiss: () -> Unit, onStart: (Boolean) -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("开始新对局") },
        text = { Text("选择你执哪一方。黑方先行，白方后行；选择白方时 AI 会先落子。") },
        confirmButton = { TextButton(onClick = { onStart(true) }) { Text("执黑先行") } },
        dismissButton = { TextButton(onClick = { onStart(false) }) { Text("执白后行") } },
    )
}

@Composable
private fun ConfirmDialog(
    title: String,
    text: String,
    confirmText: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(text) },
        confirmButton = { TextButton(onClick = onConfirm) { Text(confirmText) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

@Composable
private fun ResultDialog(result: String?, onNewGame: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(resultText(result)) },
        text = { Text("本局已结束。可以再来一局，或先看看棋盘。") },
        confirmButton = { TextButton(onClick = onNewGame) { Text("再来一局") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("看看棋盘") } },
    )
}

@Composable
private fun GearDialog(
    state: GameUiState,
    onSelect: (Int) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("AI 棋力") },
        text = {
            Column {
                state.gearLevels.forEach { level ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onSelect(level.t0) }
                            .padding(vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(
                            selected = state.gear == level.t0,
                            onClick = { onSelect(level.t0) },
                        )
                        Column(Modifier.padding(start = 4.dp)) {
                            Text(level.name, style = MaterialTheme.typography.bodyLarge)
                            Text(
                                text = level.note,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
                Text(
                    text = "切换后从 AI 的下一手开始生效。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "棋逢小鲸 v${BuildConfig.VERSION_NAME}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } },
    )
}

private fun gearName(state: GameUiState): String =
    state.gearLevels.firstOrNull { it.t0 == state.gear }?.name ?: "标准"

private fun resultText(result: String?): String = when (result) {
    "human_win" -> "你赢了"
    "ai_win" -> "你输了"
    "draw" -> "和棋"
    else -> "对局结束"
}
