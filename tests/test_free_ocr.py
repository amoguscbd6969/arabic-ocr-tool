import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import free_ocr


class NuqtaLoopGuardTests(unittest.TestCase):
    def test_keeps_normal_output(self):
        text = "السطر الأول\nالسطر الثاني"
        self.assertEqual(free_ocr._truncate_repetition_loop(text), (text, False))

    def test_discards_contaminated_character_loop_line(self):
        text = "السطر الأول\nالسطر الثاني\nالمبلغ ٦٥٠" + "٠" * 20
        self.assertEqual(
            free_ocr._truncate_repetition_loop(text),
            ("السطر الأول\nالسطر الثاني", True),
        )

    def test_discards_repeated_phrase_line(self):
        text = "نص صالح\n" + "كلمة " * 12
        self.assertEqual(free_ocr._truncate_repetition_loop(text), ("نص صالح", True))

    def test_discards_repeated_complete_lines_after_first_copy(self):
        text = "مقدمة\nسطر مكرر\nسطر مكرر\nسطر مكرر\nسطر مكرر"
        self.assertEqual(
            free_ocr._truncate_repetition_loop(text),
            ("مقدمة\nسطر مكرر", True),
        )

    @patch("free_ocr.subprocess.run")
    @patch(
        "free_ocr._find_nuqta_executable",
        return_value=Path("llama-mtmd-cli.exe"),
    )
    def test_recognize_returns_recovered_prefix_as_warning(self, _, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="السطر الأول\nالسطر الثاني\nالمبلغ ٦٥٠" + "٠" * 20,
            stderr="",
        )
        result = free_ocr.recognize_nuqta(Image.new("RGB", (20, 20)))
        self.assertEqual(result.lines, ["السطر الأول", "السطر الثاني"])
        self.assertTrue(result.loop_detected)
        self.assertIsNone(result.error)
        self.assertEqual(
            result.warning,
            "Nuqta loop suffix removed; complete prefix retained",
        )

if __name__ == "__main__":
    unittest.main()
