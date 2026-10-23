"""人类棋谱导入的格式、分支视角、筛选与可复现性。"""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import import_classical_openings as importer
from scripts.import_classical_openings import (read_library, history_board,
                                               opening_record, study_candidates)
from gomoku.core.opening_book import OpeningBook
from gomoku.core.opening_book_data import RECORDS, ALTERNATIVES, PROVENANCE

HEADER = b'\xffRenLib\xff\x03\0' + b'\xff' * 10


class ClassicalImportTest(unittest.TestCase):
    def test_manifest_lf_and_crlf_generate_identical_artifacts(self):
        manifest = {'sources': {'human': {'format': 'text'}}, 'recommendations': [
            {'source': 'human', 'basis': 'human recommendation', 'history': [], 'move': [7, 7]}]}
        raw = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
        with tempfile.TemporaryDirectory(prefix='gomoku-manifest-newlines-') as tmp:
            root = Path(tmp)
            edited = raw.replace(b'human recommendation', b'changed recommendation')
            for name, data in [('lf', raw), ('crlf', raw.replace(b'\n', b'\r\n')), ('edited', edited)]:
                path = root / (name + '.json')
                path.write_bytes(data)
                with contextlib.redirect_stdout(io.StringIO()):
                    importer.main(['--manifest', str(path), '--library-dir', tmp,
                                   '--output-dir', str(root / name)])
            for filename in ('opening_book_data.py', 'classical_audit.json'):
                self.assertEqual((root / 'lf' / filename).read_bytes(),
                                 (root / 'crlf' / filename).read_bytes(), filename)
                self.assertNotIn(b'\r\n', (root / 'lf' / filename).read_bytes())
            original_audit = json.loads((root / 'lf' / 'classical_audit.json').read_bytes())
            edited_audit = json.loads((root / 'edited' / 'classical_audit.json').read_bytes())
            self.assertEqual(edited_audit['provenance']['manifest_sha256'], hashlib.sha256(edited).hexdigest())
            self.assertNotEqual(original_audit['provenance']['manifest_sha256'],
                                edited_audit['provenance']['manifest_sha256'])

    def test_fork_returns_to_parent_and_sibling_does_not_change_turn(self):
        nodes = read_library(HEADER + bytes((0x78, 0, 0x68, 128, 0x69, 64, 0x89, 0)))
        self.assertEqual([n['point'] for n in nodes], [(7, 7), (6, 7), (6, 8), (8, 8)])
        self.assertEqual([n['parent'] for n in nodes], [None, 0, 1, 0])

    def test_new_comment_consumes_both_bytes_including_padding(self):
        nodes = read_library(HEADER + b'\x78\x08ab\0\0\x68\0')
        self.assertEqual(nodes[0]['comment'], b'ab')
        self.assertEqual(nodes[1]['point'], (6, 7))

    def test_old_comment_and_extended_board_text_preserve_alignment(self):
        nodes = read_library(HEADER + b'\x78\x20a\xe2\x68\x01\0\x01Z\0')
        self.assertEqual(nodes[0]['comment'], b'ab')
        self.assertEqual(nodes[1]['label'], b'Z')
        self.assertEqual(nodes[1]['info'] & 256, 256)

    def test_malformed_input_is_rejected_instead_of_generating_wrong_positions(self):
        for data in (b'', HEADER + b'\x78', HEADER + b'\x78\x08ab',
                     HEADER + b'\x78\x80', HEADER + b'\x78\x40\x68\0',
                     HEADER + b'\x70\0'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_library(data)

    def test_pass_and_invalid_terminal_histories_are_not_silently_shortened(self):
        for history in (((7, 7), None), ((7, 7), (7, 7)), ((15, 0),),
                        tuple(p for c in range(5) for p in ((7, c), (0, c * 2)))[:-1]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                history_board(history)

    def test_only_study_continuation_is_not_an_early_takeover_recommendation(self):
        # 前六子是无成五的合法盘面，第七子为唯一未评级研究着。
        stream = HEADER + bytes((0x78, 0, 0x68, 0, 0x59, 0, 0x87, 0,
                                 0x99, 0, 0x46, 0, 0x58, 0))
        nodes = read_library(stream)
        candidates = list(study_candidates(nodes))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][1]['basis'], 'single-study-continuation')
        self.assertEqual(list(study_candidates(nodes, min_stones=7)), [])

    def test_bad_or_foul_dependent_continuations_are_excluded(self):
        prefix = HEADER + bytes((0x78, 0, 0x68, 0, 0x59, 0, 0x87, 0, 0x99, 0, 0x46, 0))
        for text in (b'?', b'Not good', b'forbidden'):
            comment = text + b'\0'
            if len(comment) % 2:
                comment += b'\0'
            self.assertEqual(list(study_candidates(read_library(prefix + b'\x58\x08' + comment))), [])

    def test_multiple_unrated_variants_do_not_choose_first_stored_move(self):
        prefix = HEADER + bytes((0x78, 0, 0x68, 0, 0x59, 0, 0x87, 0, 0x99, 0, 0x46, 0))
        nodes = read_library(prefix + bytes((0x58, 192, 0x75, 0)))
        self.assertEqual(list(study_candidates(nodes, max_stones=6)), [])

    def test_explicit_positive_sibling_is_selected_over_unrated_first_move(self):
        prefix = HEADER + bytes((0x78, 0, 0x68, 0, 0x59, 0, 0x87, 0, 0x99, 0, 0x46, 0))
        nodes = read_library(prefix + bytes((0x58, 192, 0x75, 8)) + b'Best\0\0')
        selected = list(study_candidates(nodes, max_stones=6))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1]['node'], 7)
        self.assertEqual(selected[0][1]['basis'], 'explicit-comment')

    def test_packaged_recommendations_are_exactly_reproducible_from_manifest(self):
        path = Path(__file__).resolve().parents[1] / 'data/classical_opening_recommendations.json'
        raw = path.read_bytes()
        manifest = json.loads(raw)
        # 文本清单以 LF 计算摘要；换行不改变内容，其他任何字节变化仍须被发现。
        lf = raw.replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(lf).hexdigest(), PROVENANCE['manifest_sha256'])
        rows, alternatives = set(), set()
        book = OpeningBook()
        for entry in manifest['recommendations']:
            self.assertIn(entry['source'], manifest['sources'])
            self.assertTrue(entry['basis'])
            row = opening_record(entry['history'], entry['move'])
            (alternatives if entry.get('alternative') else rows).add(row)
            board = history_board(entry['history'])
            self.assertTrue(book._lookup_choices(board, len(entry['history']) % 2 + 1))
        self.assertEqual(rows, set(RECORDS))
        self.assertEqual(alternatives, set(ALTERNATIVES))


if __name__ == '__main__':
    unittest.main()
