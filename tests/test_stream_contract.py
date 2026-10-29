import ast
import unittest
from pathlib import Path


source = ast.parse((Path(__file__).resolve().parents[1] / "stream.py").read_text(encoding="utf-8"))
sink = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "UpscaleSink")
getitem = next(n for n in sink.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
scope = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[getitem], type_ignores=[])), "stream.py", "exec"), scope)


class Sink:
    __getitem__ = scope["__getitem__"]

    def __init__(self, written=0):
        self.written = written
        self.pending = None


class StreamContractTest(unittest.TestCase):
    def test_ordinary_h3_five_axis_write(self):
        sink = Sink()
        self.assertIs(sink[:, :, 0:17, :, :], sink)
        self.assertEqual(sink.pending, 17)

    def test_sol_implicit_spatial_axes_and_tail(self):
        sink = Sink()
        self.assertIs(sink[0:1, :, 0:17], sink)
        self.assertEqual(sink.pending, 17)
        sink.written = 119
        self.assertIs(sink[0:1, :, 119:124], sink)
        self.assertEqual(sink.pending, 5)

    def test_rejects_batch_channel_or_spatial_subset(self):
        for key in [(slice(1, 2), slice(None), slice(0, 17)),
                    (slice(None), slice(0, 1), slice(0, 17)),
                    (slice(None), slice(None), slice(0, 17), slice(0, 1), slice(None))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                Sink()[key]

    def test_rejects_nonsequential_and_strided_time(self):
        with self.assertRaises(ValueError):
            Sink(written=17)[0:1, :, 0:17]
        with self.assertRaises(ValueError):
            Sink()[0:1, :, 0:17:2]


if __name__ == "__main__":
    unittest.main()
