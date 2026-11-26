import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import config


class DemoAppTests(unittest.TestCase):
    def test_demo_supports_all_feature_sets_and_predicts(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        app = AppTest.from_file(str(app_path), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.selectbox[0].options), len(config.FEATURE_SETS))
        self.assertEqual(app.selectbox[0].value, "C")
        self.assertEqual(len(app.selectbox[1].options), 5)
        self.assertEqual(app.selectbox[1].value, "gb")

        for feature_set in ("A", "B", "C"):
            with self.subTest(feature_set=feature_set):
                app.selectbox[0].set_value(feature_set)
                app.run()
                app.form_submit_button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.metric), 1)
                self.assertEqual(len(app.error), 0)


if __name__ == "__main__":
    unittest.main()
