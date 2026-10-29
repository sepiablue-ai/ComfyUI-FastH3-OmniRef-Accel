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
        for stem in ("fasth3_omniref_accel", "fasth3_omniref_accel_sol"):
            with self.subTest(workflow=stem):
                api = json.loads((ROOT / "workflows" / (stem + ".api.json")).read_text(encoding="utf-8"))
                self.assertEqual({n["class_type"] for n in api.values()} - known, set())
                gui = json.loads((ROOT / "workflows" / (stem + ".json")).read_text(encoding="utf-8"))
                self.assertEqual({n["type"] for n in gui["nodes"]} - known, set())

    def test_api_links_resolve(self):
        for path in (ROOT / "workflows").glob("*.api.json"):
            api = json.loads(path.read_text(encoding="utf-8"))
            for node in api.values():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                        self.assertIn(value[0], api)

    def test_sol_only_changes_final_video_decode(self):
        base = json.loads((ROOT / "workflows/fasth3_omniref_accel.api.json").read_text(encoding="utf-8"))
        sol = json.loads((ROOT / "workflows/fasth3_omniref_accel_sol.api.json").read_text(encoding="utf-8"))
        self.assertEqual(set(sol) - set(base), {"30"})
        self.assertEqual(sol["30"]["inputs"]["vae_name"], "minimax_h3_sol_vae_int8_convrot.safetensors")
        self.assertEqual(sol["9"]["inputs"]["vae"], ["4", 0])
        self.assertEqual(sol["29"]["inputs"]["vae"], ["30", 0])
        for key, node in base.items():
            expected = dict(node["inputs"])
            if key == "29":
                expected["vae"] = ["30", 0]
                expected["filename_prefix"] = "video/FastH3_OmniRef_Accel_Sol"
            self.assertEqual(sol[key]["inputs"], expected)

    def test_sol_gui_vae_links_match_api(self):
        gui = json.loads((ROOT / "workflows/fasth3_omniref_accel_sol.json").read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in gui["nodes"]}
        self.assertEqual(nodes[30]["widgets_values"], ["minimax_h3_sol_vae_int8_convrot.safetensors"])
        self.assertTrue(any(l[1] == 4 and l[3] == 9 and l[5] == "VAE" for l in gui["links"]))
        self.assertTrue(any(l[1] == 30 and l[3] == 29 and l[5] == "VAE" for l in gui["links"]))
        for lid, src, slot, dst, index, kind in gui["links"]:
            self.assertEqual(nodes[dst]["inputs"][index]["link"], lid)
            self.assertIn(lid, nodes[src]["outputs"][slot]["links"])


if __name__ == "__main__":
    unittest.main()
