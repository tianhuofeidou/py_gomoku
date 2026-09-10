package com.example.gomoku

import org.json.JSONArray
import org.json.JSONObject

data class Move(val r: Int, val c: Int, val player: Int)

data class GearLevel(val name: String, val t0: Int, val note: String)

data class GameUiState(
    val board: List<Int> = List(225) { 0 },
    val moves: List<Move> = emptyList(),
    val moveCount: Int = 0,
    val current: Int = 1,
    val human: Int = 1,
    val winner: Int? = null,
    val over: Boolean = false,
    val last: Move? = null,
    val canUndo: Boolean = false,
    val turn: String = "human",
    val result: String? = null,
    val summary: String = "暂无历史战绩",
    val thinking: Boolean = false,
    val thinkingSeconds: Double = 0.0,
    val lastAiSeconds: Double? = null,
    val gear: Int = 160,
    val gearLevels: List<GearLevel> = emptyList(),
    val busy: Boolean = false,
    val error: String? = null,
)

data class RecentGame(val result: String?, val moves: Int, val at: Long, val humanPlayer: Int)

data class LineInfo(val killType: String, val moves: Int, val at: Long)

data class MemoryUiState(
    val wins: Int = 0,
    val losses: Int = 0,
    val draws: Int = 0,
    val lossByType: List<Pair<String, Int>> = emptyList(),
    val aiLossTotal: Int = 0,
    val recent: List<RecentGame> = emptyList(),
    val badLines: List<LineInfo> = emptyList(),
    val goodLines: List<LineInfo> = emptyList(),
    val hint: String = "",
)

fun JSONObject.toGameUiState(): GameUiState {
    val boardArray = optJSONArray("board")
    val board = ArrayList<Int>(225)
    if (boardArray != null) {
        for (i in 0 until boardArray.length()) {
            board.add(boardArray.optInt(i, 0))
        }
    }
    while (board.size < 225) {
        board.add(0)
    }

    val movesArray = optJSONArray("moves")
    val moves = ArrayList<Move>()
    if (movesArray != null) {
        for (i in 0 until movesArray.length()) {
            val m = movesArray.optJSONObject(i) ?: continue
            moves.add(Move(m.optInt("r"), m.optInt("c"), m.optInt("player")))
        }
    }

    val lastObj = optJSONObject("last")
    val last = if (lastObj != null) {
        Move(lastObj.optInt("r"), lastObj.optInt("c"), lastObj.optInt("player"))
    } else {
        null
    }

    return GameUiState(
        board = board,
        moves = moves,
        moveCount = optInt("moveCount", moves.size),
        current = optInt("current", 1),
        human = optInt("human", 1),
        winner = if (isNull("winner")) null else optInt("winner"),
        over = optBoolean("over", false),
        last = last,
        canUndo = optBoolean("canUndo", false),
        turn = optString("turn", "human"),
        result = if (isNull("result")) null else optString("result"),
        summary = optString("summary", ""),
        gear = optInt("gear", 160),
    )
}

fun JSONArray.toGearLevels(): List<GearLevel> {
    val out = ArrayList<GearLevel>()
    for (i in 0 until length()) {
        val item = optJSONObject(i) ?: continue
        out.add(GearLevel(item.optString("name"), item.optInt("t0", 160), item.optString("note")))
    }
    return out
}

fun JSONObject.toMemoryUiState(): MemoryUiState {
    val totals = optJSONObject("totals") ?: JSONObject()
    val lossObj = optJSONObject("lossByType") ?: JSONObject()
    val lossList = ArrayList<Pair<String, Int>>()
    val keys = lossObj.keys()
    while (keys.hasNext()) {
        val key = keys.next()
        lossList.add(key to lossObj.optInt(key, 0))
    }
    lossList.sortByDescending { it.second }

    val recentArray = optJSONArray("recent")
    val recent = ArrayList<RecentGame>()
    if (recentArray != null) {
        for (i in 0 until recentArray.length()) {
            val item = recentArray.optJSONObject(i) ?: continue
            recent.add(
                RecentGame(
                    result = if (item.isNull("result")) null else item.optString("result"),
                    moves = item.optInt("moves", 0),
                    at = item.optLong("at", 0L),
                    humanPlayer = item.optInt("humanPlayer", 1),
                )
            )
        }
    }

    fun parseLines(name: String): List<LineInfo> {
        val array = optJSONArray(name) ?: return emptyList()
        val out = ArrayList<LineInfo>()
        for (i in 0 until array.length()) {
            val item = array.optJSONObject(i) ?: continue
            out.add(
                LineInfo(
                    killType = item.optString("killType", "其他"),
                    moves = item.optInt("moves", 0),
                    at = item.optLong("at", 0L),
                )
            )
        }
        return out
    }

    return MemoryUiState(
        wins = totals.optInt("wins", 0),
        losses = totals.optInt("losses", 0),
        draws = totals.optInt("draws", 0),
        lossByType = lossList,
        aiLossTotal = optInt("aiLossTotal", 0),
        recent = recent,
        badLines = parseLines("badLines"),
        goodLines = parseLines("goodLines"),
        hint = optString("hint", ""),
    )
}
