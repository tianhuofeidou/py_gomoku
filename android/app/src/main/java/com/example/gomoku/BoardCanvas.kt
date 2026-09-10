package com.example.gomoku

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

private val WoodColor = Color(0xFFE0B878)
private val LineColor = Color(0xFF6D4C2A)
private val MarkerColor = Color(0xFFFF3D3D)

/**
 * 15×15 棋盘。坐标是零基 (row, col)，点击按最近交叉点取整。
 * 关掉候选日志后，这里只负责棋盘、棋子和最后一手标记。
 */
@Composable
fun BoardCanvas(
    board: List<Int>,
    last: Move?,
    enabled: Boolean,
    onTap: (Int, Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .aspectRatio(1f)
            .clip(RoundedCornerShape(12.dp))
            .background(WoodColor)
            .then(
                if (enabled) {
                    Modifier.pointerInput(Unit) {
                        detectTapGestures { offset ->
                            val cell = size.width / 15f
                            if (cell > 0f) {
                                val pad = cell / 2f
                                val c = ((offset.x - pad) / cell).roundToInt().coerceIn(0, 14)
                                val r = ((offset.y - pad) / cell).roundToInt().coerceIn(0, 14)
                                onTap(r, c)
                            }
                        }
                    }
                } else {
                    Modifier
                }
            )
    ) {
        Canvas(Modifier.fillMaxSize()) {
            val cell = size.width / 15f
            val pad = cell / 2f
            val right = size.width - pad
            val bottom = size.height - pad

            for (i in 0 until 15) {
                val p = pad + i * cell
                val stroke = if (i == 0 || i == 14) 2.5f else 1.2f
                drawLine(LineColor, Offset(pad, p), Offset(right, p), strokeWidth = stroke)
                drawLine(LineColor, Offset(p, pad), Offset(p, bottom), strokeWidth = stroke)
            }

            val stars = listOf(3 to 3, 3 to 11, 11 to 3, 11 to 11, 7 to 7)
            for ((r, c) in stars) {
                drawCircle(
                    color = LineColor,
                    radius = cell * 0.09f,
                    center = Offset(pad + c * cell, pad + r * cell),
                )
            }

            for (index in board.indices) {
                val value = board[index]
                if (value == 0) continue
                val r = index / 15
                val c = index % 15
                val center = Offset(pad + c * cell, pad + r * cell)
                val radius = cell * 0.43f
                if (value == 1) {
                    val brush = Brush.radialGradient(
                        colors = listOf(Color(0xFF808080), Color(0xFF161616), Color(0xFF000000)),
                        center = Offset(center.x - radius * 0.35f, center.y - radius * 0.35f),
                        radius = radius * 1.6f,
                    )
                    drawCircle(brush, radius, center)
                } else {
                    val brush = Brush.radialGradient(
                        colors = listOf(Color.White, Color(0xFFF2F2F2), Color(0xFFBDBDBD)),
                        center = Offset(center.x - radius * 0.35f, center.y - radius * 0.35f),
                        radius = radius * 1.6f,
                    )
                    drawCircle(brush, radius, center)
                    drawCircle(Color(0x33000000), radius, center, style = Stroke(width = 1.5f))
                }
            }

            if (last != null) {
                val center = Offset(pad + last.c * cell, pad + last.r * cell)
                drawCircle(MarkerColor, cell * 0.11f, center)
            }
        }
    }
}
