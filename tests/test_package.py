import ast
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBH_SHA256 = "9fb4e2ec679f7ba9cc8e10685053d2080ff2891cc0f8913ec9a2660614441879"
NATIVE = {"BasicGuider", "BasicScheduler", "BlockSparseAttention", "CLIPLoader", "DisableNoise", "EmptyMiniMaxH3LatentAV",
          "KSamplerSelect", "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LoadImage", "MiniMaxH3ReferenceToVideo",
          "MiniMaxH3SigmaShift", "ModelAttentionBackend", "RandomNoise", "SamplerCustomAdvanced", "SplitSigmas", "UNETLoader",
          "UpscaleModelLoader", "VAEDecodeAudio", "VAELoader"}


def package_nodes():
    names = set()
    for path in ROOT.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NODE_CLASS_MAPPINGS" for t in node.targets):
                if isinstance(node.value, ast.Dict):
                    names.update(key.value for key in node.value.keys)
    return names


class PackageTest(unittest.TestCase):
    def test_vendored_lbh_is_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT / "vendor/lbh_3d.py").read_bytes()).hexdigest(), LBH_SHA256)
        self.assertTrue((ROOT / "vendor/LBH_LICENSE.txt").read_text(encoding="utf-8").startswith("MIT License"))

    def test_workflows_use_only_native_and_package_nodes(self):
        known = NATIVE | package_nodes()
        api = json.loads((ROOT / "workflows/fasth3_omniref_accel.api.json").read_text(encoding="utf-8"))
        self.assertEqual({n["class_type"] for n in api.values()} - known, set())
        gui = json.loads((ROOT / "workflows/fasth3_omniref_accel.json").read_text(encoding="utf-8"))
        self.assertEqual({n["type"] for n in gui["nodes"]} - known, set())

    def test_api_links_resolve(self):
        api = json.loads((ROOT / "workflows/fasth3_omniref_accel.api.json").read_text(encoding="utf-8"))
        for node in api.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], api)


if __name__ == "__main__":
    unittest.main()
