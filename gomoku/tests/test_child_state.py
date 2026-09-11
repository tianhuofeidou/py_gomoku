# -*- coding: utf-8 -*-
"""5 态离散判定回归测试。

本文件只锁定状态判定规则，不改动 deep_search 的决策排序：
- _merge_child_states：新 5 态聚合
- _flip_state：黑白视角翻转
- _state_forced / _state_child_state / _state_name：对外字段导出
"""
import unittest

from gomoku.core.deep_search import DeepSearch


class MergeChildStatesTest(unittest.TestCase):
    """只有硬状态 ±1 向上传播；±2 是局部软证据，父层按 0 处理。"""

    def test_empty_children_is_none(self):
        self.assertEqual(DeepSearch._merge_child_states([]), DeepSearch.STATE_NONE)

    def test_hard_win_requires_all_children_hard_win(self):
        self.assertEqual(DeepSearch._merge_child_states([1, 1]), 1)
        self.assertEqual(DeepSearch._merge_child_states([1, 1, 1]), 1)
        # 有硬胜但没全胜 → 存在必胜
        self.assertEqual(DeepSearch._merge_child_states([1, 0]), 2)
        self.assertEqual(DeepSearch._merge_child_states([1, 2]), 2)
        # 软败不参与传播：+1 加 -2 仍按“有硬胜”处理
        self.assertEqual(DeepSearch._merge_child_states([1, -2]), 2)
        # 硬败优先于硬胜
        self.assertEqual(DeepSearch._merge_child_states([1, -1]), -2)

    def test_hard_lose_only_when_all_children_hard_lose(self):
        self.assertEqual(DeepSearch._merge_child_states([-1, -1]), -1)
        self.assertEqual(DeepSearch._merge_child_states([-1, -1, -1]), -1)
        # 有硬败但没全败 → 存在必败
        self.assertEqual(DeepSearch._merge_child_states([-1, -2]), -2)
        self.assertEqual(DeepSearch._merge_child_states([-2, -2, -1]), -2)
        # 全是软败：不传播，父层视为无
        self.assertEqual(DeepSearch._merge_child_states([-2, -2]), 0)

    def test_only_hard_evidence_propagates(self):
        # 软败单独存在：不传播
        self.assertEqual(DeepSearch._merge_child_states([-2, 0]), 0)
        self.assertEqual(DeepSearch._merge_child_states([-2, 0, 0]), 0)
        # 软胜单独存在：不传播
        self.assertEqual(DeepSearch._merge_child_states([2, 2]), 0)
        self.assertEqual(DeepSearch._merge_child_states([2, 0]), 0)
        # 有硬胜时，软败被忽略
        self.assertEqual(DeepSearch._merge_child_states([1, -2]), 2)
        self.assertEqual(DeepSearch._merge_child_states([-2, 1]), 2)
        # 有硬败时，硬败优先，软胜被忽略
        self.assertEqual(DeepSearch._merge_child_states([-1, 2]), -2)
        self.assertEqual(DeepSearch._merge_child_states([2, -1]), -2)

    def test_all_none_is_none(self):
        self.assertEqual(DeepSearch._merge_child_states([0, 0]), 0)
        self.assertEqual(DeepSearch._merge_child_states([0]), 0)


class StateFlipTest(unittest.TestCase):
    def test_flip(self):
        self.assertEqual(DeepSearch._flip_state(1), -1)
        self.assertEqual(DeepSearch._flip_state(-1), 1)
        self.assertEqual(DeepSearch._flip_state(2), -2)
        self.assertEqual(DeepSearch._flip_state(-2), 2)
        self.assertEqual(DeepSearch._flip_state(0), 0)


class StateFieldExportTest(unittest.TestCase):
    CASES = [
        (1, 1, 'none', '必胜'),
        (2, 0, 'exists_win', '存在必胜'),
        (0, 0, 'none', '无'),
        (-2, 0, 'exists_lose', '存在必败'),
        (-1, -1, 'none', '必败'),
    ]

    def test_export_fields(self):
        for state, forced, child_state, name in self.CASES:
            with self.subTest(state=state):
                self.assertEqual(DeepSearch._state_forced(state), forced)
                self.assertEqual(DeepSearch._state_child_state(state), child_state)
                self.assertEqual(DeepSearch._state_name(state), name)


class DecisionKeyTest(unittest.TestCase):
    def test_state_tier_order(self):
        # 必胜 > 存在必胜 > 无 > 存在必败 > 必败，档位优先于候选分。
        self.assertGreater(DeepSearch._pick_key_state(1, 0.0),
                           DeepSearch._pick_key_state(2, 999.0))
        self.assertGreater(DeepSearch._pick_key_state(2, 0.0),
                           DeepSearch._pick_key_state(0, 999.0))
        self.assertGreater(DeepSearch._pick_key_state(0, 0.0),
                           DeepSearch._pick_key_state(-2, 999.0))
        self.assertGreater(DeepSearch._pick_key_state(-2, 0.0),
                           DeepSearch._pick_key_state(-1, 999.0))

    def test_same_state_uses_score(self):
        self.assertGreater(DeepSearch._pick_key_state(0, 10.0),
                           DeepSearch._pick_key_state(0, 5.0))
        self.assertGreater(DeepSearch._pick_key_state(-2, 10.0),
                           DeepSearch._pick_key_state(-2, 5.0))


if __name__ == '__main__':
    unittest.main()
