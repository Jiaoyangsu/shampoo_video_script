import sys
import types
import unittest


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType("gradio")
        sys.modules.setdefault("gradio", stub)

    def test_run_app_imports_without_starting_gradio(self):
        # The old run_app imported `demo` from a module where it only existed
        # under __main__; this import is the regression test.
        import scripts.run_app as run_app

        self.assertTrue(callable(run_app.build_app))

    def test_form_ui_wrapper_imports(self):
        import src.app.form_ui as form_ui

        self.assertTrue(callable(form_ui.create_form_demo))


if __name__ == "__main__":
    unittest.main()
