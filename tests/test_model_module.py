import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import model as model_module


class ModelModuleTests(unittest.TestCase):
    def test_model_files_include_cassava_and_maize(self):
        self.assertIn("cassava", model_module.MODEL_FILES)
        self.assertIn("maize", model_module.MODEL_FILES)

    def test_model_files_exist_on_disk(self):
        for crop, filename in model_module.MODEL_FILES.items():
            path = Path(filename)
            if not path.is_absolute():
                path = ROOT / filename
            self.assertTrue(path.exists(), f"{crop} model not found at {path}")


if __name__ == "__main__":
    unittest.main()
