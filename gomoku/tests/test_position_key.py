"""完整盘面键编码等价：行列次序、逐格区分、输入不变及合法行表示。"""
import copy
import random
import unittest

from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import SIZE,empty_board


class PositionKeyTest(unittest.TestCase):
    def test_empty_full_and_known_row_major_coordinates(self):
        self.assertEqual(DeepSearch._position_key(empty_board()),b'\x00'*225)
        for color in (1,2):
            self.assertEqual(DeepSearch._position_key([[color]*SIZE for _ in range(SIZE)]),bytes([color])*225)
        board=empty_board()
        board[0][1]=1; board[1][0]=2; board[14][14]=1
        expected=bytearray(225)
        expected[1]=1; expected[15]=2; expected[224]=1
        self.assertEqual(DeepSearch._position_key(board),bytes(expected))

    def test_every_cell_and_color_remains_distinguishable(self):
        keys={b'\x00'*225}
        for r in range(SIZE):
            for c in range(SIZE):
                for color in (1,2):
                    board=empty_board(); board[r][c]=color
                    key=DeepSearch._position_key(board)
                    expected=bytearray(225); expected[15*r+c]=color
                    self.assertEqual(key,bytes(expected))
                    self.assertNotIn(key,keys)
                    keys.add(key)
        self.assertEqual(len(keys),451)

    def test_seeded_boards_match_previous_encoder_without_mutation(self):
        rng=random.Random(20261104)
        for _ in range(128):
            board=[[rng.randrange(3) for _ in range(SIZE)] for _ in range(SIZE)]
            before=copy.deepcopy(board)
            reference=bytes(cell for row in board for cell in row)
            self.assertEqual(DeepSearch._position_key(board),reference)
            self.assertEqual(len(reference),225)
            self.assertEqual(board,before)

    def test_list_tuple_and_byte_rows_have_identical_keys(self):
        board=[[(r+2*c)%3 for c in range(SIZE)] for r in range(SIZE)]
        expected=DeepSearch._position_key(board)
        for rows in (tuple(tuple(row) for row in board),[bytes(row) for row in board],
                     [bytearray(row) for row in board]):
            self.assertEqual(DeepSearch._position_key(rows),expected)
            self.assertEqual(bytes(cell for row in rows for cell in row),expected)


if __name__=='__main__':
    unittest.main()
