import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("sizes", Path(__file__).resolve().parents[1] / "sizes.py")
sizes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sizes)


def plan(width, height, minimum=512):
    return sizes.FastH3AccelSizes().sizes(width, height, minimum)["result"]


class SizesTest(unittest.TestCase):
    def test_validated_sizes_are_exact(self):
        self.assertEqual(plan(1024, 1792), (512, 896, 256, 448, 0.5, 1024, 1792))
        self.assertEqual(plan(1792, 1024), (896, 512, 448, 256, 0.5, 1792, 1024))

    def test_small_output_is_lifted_to_the_minimum(self):
        width, height, low_w, low_h, scale_by, crop_w, crop_h = plan(720, 1280)
        self.assertEqual((width, height, low_w, low_h), (512, 896, 256, 448))
        self.assertEqual((crop_w, crop_h), (720, 1280))
        self.assertGreaterEqual(round(height * 4 * scale_by), crop_h)
        self.assertGreaterEqual(round(width * 4 * scale_by), crop_w)

    def test_grid_and_two_x_relation(self):
        for size in [(1024, 1792), (1088, 1920), (1280, 720), (1024, 1024), (2048, 1152)]:
            width, height, low_w, low_h, scale_by, crop_w, crop_h = plan(*size)
            self.assertEqual((width, height), (low_w * 2, low_h * 2))
            self.assertEqual((low_w % 32, low_h % 32), (0, 0))
            self.assertGreaterEqual(min(width, height), 512)
            # resized frames cover the output, so the crop never stretches
            self.assertGreaterEqual(round(width * 4 * scale_by), crop_w)
            self.assertGreaterEqual(round(height * 4 * scale_by), crop_h)


if __name__ == "__main__":
    unittest.main()
