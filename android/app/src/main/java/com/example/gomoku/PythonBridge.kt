package com.example.gomoku

import com.chaquo.python.PyObject
import com.chaquo.python.Python
import org.json.JSONArray
import org.json.JSONObject

/**
 * Chaquopy 调用入口。所有 Python 函数都返回 JSON 字符串，
 * 这里统一转成 org.json.JSONObject 供 UI 层使用。
 */
object PythonBridge {
    private val module: PyObject by lazy { Python.getInstance().getModule("android_api") }

    fun init(home: String): JSONObject = JSONObject(module.callAttr("init_app", home).toString())

    fun newGame(humanBlack: Boolean): JSONObject =
        JSONObject(module.callAttr("new_game", humanBlack).toString())

    fun state(): JSONObject = JSONObject(module.callAttr("state").toString())

    fun humanMove(r: Int, c: Int): JSONObject =
        JSONObject(module.callAttr("human_move", r, c).toString())

    fun aiMove(): JSONObject = JSONObject(module.callAttr("ai_move").toString())

    fun undo(): JSONObject = JSONObject(module.callAttr("undo").toString())

    fun resign(): JSONObject = JSONObject(module.callAttr("resign").toString())

    fun draw(): JSONObject = JSONObject(module.callAttr("draw").toString())

    fun memory(): JSONObject = JSONObject(module.callAttr("memory").toString())

    fun gearLevels(): JSONArray = JSONArray(module.callAttr("gear_levels").toString())

    fun setGear(t0: Int): JSONObject = JSONObject(module.callAttr("set_gear", t0).toString())
}
