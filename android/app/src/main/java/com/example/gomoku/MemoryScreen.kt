package com.example.gomoku

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun MemoryScreen(
    memory: MemoryUiState,
    onBack: () -> Unit,
    onRefresh: () -> Unit,
) {
    Scaffold { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TextButton(onClick = onBack) { Text("返回") }
                Text(
                    text = "对局记忆",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onRefresh) { Text("刷新") }
            }
            TotalsCard(memory)
            Spacer(Modifier.height(12.dp))
            HintCard(memory.hint)
            Spacer(Modifier.height(12.dp))
            LossTypeCard(memory.lossByType, memory.aiLossTotal)
            Spacer(Modifier.height(12.dp))
            LinesCard(memory)
            Spacer(Modifier.height(12.dp))
            RecentCard(memory.recent)
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun TotalsCard(memory: MemoryUiState) {
    val total = memory.wins + memory.losses + memory.draws
    val rate = if (total > 0) memory.wins * 100 / total else 0
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("你的战绩", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                StatColumn("胜", memory.wins, Color(0xFF2E7D32))
                StatColumn("负", memory.losses, Color(0xFFC62828))
                StatColumn("和", memory.draws, Color(0xFF757575))
            }
            Spacer(Modifier.height(12.dp))
            Text(
                text = "共 $total 局 · 胜率 $rate%",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun StatColumn(label: String, value: Int, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value.toString(),
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = color,
        )
        Text(label, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun HintCard(hint: String) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("经验提示", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            Text(
                text = hint.ifBlank { "还没有足够的对局经验。多下几局后，这里会根据历史败因和棋谱给出提示。" },
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun LossTypeCard(lossByType: List<Pair<String, Int>>, aiLossTotal: Int) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("AI 败因分布（你取胜的杀法）", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            if (aiLossTotal <= 0) {
                Text("还没有你取胜的记录。", style = MaterialTheme.typography.bodyMedium)
            } else {
                lossByType.filter { it.second > 0 }.forEach { (type, count) ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(type, modifier = Modifier.width(56.dp), style = MaterialTheme.typography.bodyMedium)
                        LinearProgressIndicator(
                            progress = { count.toFloat() / aiLossTotal.toFloat() },
                            modifier = Modifier
                                .weight(1f)
                                .padding(horizontal = 8.dp),
                        )
                        Text("$count", style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}

@Composable
private fun LinesCard(memory: MemoryUiState) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("棋谱经验", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            Text(
                text = "AI 胜局棋谱 ${memory.goodLines.size} 局 · AI 败局棋谱 ${memory.badLines.size} 局",
                style = MaterialTheme.typography.bodyMedium,
            )
            if (memory.goodLines.isNotEmpty()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "AI 最近胜局：" + memory.goodLines.take(3).joinToString("、") { "${it.killType}(${it.moves}手)" },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (memory.badLines.isNotEmpty()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "AI 最近败局：" + memory.badLines.take(3).joinToString("、") { "${it.killType}(${it.moves}手)" },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (memory.goodLines.isEmpty() && memory.badLines.isEmpty()) {
                Spacer(Modifier.height(4.dp))
                Text("暂无棋谱记录。", style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

@Composable
private fun RecentCard(recent: List<RecentGame>) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("最近对局", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            if (recent.isEmpty()) {
                Text("还没有已完成的对局。", style = MaterialTheme.typography.bodyMedium)
            } else {
                recent.forEachIndexed { index, game ->
                    if (index > 0) {
                        HorizontalDivider(Modifier.padding(vertical = 6.dp))
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = resultLabel(game.result),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = resultColor(game.result),
                            modifier = Modifier.width(36.dp),
                        )
                        Text(
                            text = "你执${if (game.humanPlayer == 1) "黑" else "白"} · ${game.moves} 手",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            text = formatTime(game.at),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

private fun resultLabel(result: String?): String = when (result) {
    "human_win" -> "胜"
    "ai_win" -> "负"
    "draw" -> "和"
    else -> "—"
}

private fun resultColor(result: String?): Color = when (result) {
    "human_win" -> Color(0xFF2E7D32)
    "ai_win" -> Color(0xFFC62828)
    else -> Color(0xFF757575)
}

private fun formatTime(ms: Long): String {
    if (ms <= 0L) return "--"
    return SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(ms))
}
