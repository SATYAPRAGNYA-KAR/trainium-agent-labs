#!/usr/bin/env python3
"""Offline tests for tile_lint.py and the v7 controller changes. No Neuron SDK, no model."""
import argparse
import glob
import os
import sys
import types
import unittest

import tile_lint

HERE = os.path.dirname(os.path.abspath(__file__))

LEVEL4_BAD = '''
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def k(lhsT, rhs):
    K, M = lhsT.shape
    result = nl.ndarray((M, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
    acc = nl.ndarray((128, 512), nl.float32, buffer=nl.psum)
    for m in nl.affine_range(M // 128):
        acc[m, n] += a[p, k] * a[p, n]
    nisa.tensor_copy(dst=result, src=acc)
    nisa.nc_matmul(dst=a, stationary=a, moving=acc)
    return result
'''


def fake_nki():
    """nki.language has `divide`; nki.isa does not. isa.tensor_tensor has a docstring."""
    nki = types.ModuleType("nki")
    nl = types.ModuleType("nki.language")
    isa = types.ModuleType("nki.isa")
    nl.divide = lambda a, b: None
    nl.ndarray = lambda *a, **k: None

    def tensor_tensor(dst, data1, data2, op):
        """Elementwise op between two tiles.

        dst and both inputs must have the same shape."""
    isa.tensor_tensor = tensor_tensor
    nki.language, nki.isa = nl, isa
    return {"nki": nki, "nki.language": nl, "nki.isa": isa}


class TestLanguageLint(unittest.TestCase):
    def test_reference_kernels_are_clean(self):
        # The false-positive test that matters: the shipped, passing kernels must never be flagged.
        for path in sorted(glob.glob(os.path.join(HERE, "reference_level*.py"))):
            with open(path) as f:
                self.assertEqual(tile_lint.lint_tile_model(f.read()), [], path)

    def test_scalar_parameter_arithmetic_is_not_a_tile_operator(self):
        code = ("import nki\nimport nki.language as nl\n@nki.jit\ndef k(x, pool_size):\n"
                "    h, w = x.shape\n    s = 1.0 / (pool_size * pool_size)\n    return x\n")
        self.assertEqual(tile_lint.lint_tile_operators(code), [])

    def test_level4_failure_is_reported_all_at_once(self):
        found = tile_lint.lint_tile_model(LEVEL4_BAD)
        text = " ".join(i["msg"] for i in found)
        self.assertIn("`n`, `p`", text)                       # undefined names, one finding
        self.assertIn("`+=` between two tiles", text)
        self.assertIn("`dst` lives in shared_hbm", text)           # tensor_copy into the output
        self.assertIn("`dst` lives in sbuf", text)                 # nc_matmul dst must be psum
        self.assertIn("`moving` lives in psum", text)
        self.assertEqual([i["line"] for i in found], sorted(i["line"] for i in found))

    def test_tile_times_scalar_is_not_flagged(self):
        code = ("import nki\nimport nki.language as nl\n@nki.jit\ndef k(x):\n"
                "    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)\n    y = 2 * 3\n    return x\n")
        self.assertEqual(tile_lint.lint_tile_operators(code), [])

    def test_syntax_error_returns_nothing(self):
        self.assertEqual(tile_lint.lint_tile_model("def ("), [])

    def test_format_is_bounded(self):
        found = tile_lint.lint_tile_model(LEVEL4_BAD)
        self.assertTrue(tile_lint.format_findings(found, limit=2).count("[line") == 2)
        self.assertEqual(tile_lint.format_findings([]), "")


class TestSdkBacked(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("nki", "nki.language", "nki.isa")}
        sys.modules.update(fake_nki())

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_missing_name_points_at_the_other_module(self):
        hint = tile_lint.sibling_hint("nki.isa.divide")
        self.assertIn("`nl.divide`", hint)

    def test_ceil_div_gets_the_plain_python_note(self):
        self.assertIn("plain Python", tile_lint.sibling_hint("nki.language.ceil_div"))

    def test_call_docs_returns_the_real_docstring_for_the_failing_line(self):
        code = ("import nki.isa as nisa\nimport nki.language as nl\n"
                "def k(a, b, c):\n    nisa.tensor_tensor(dst=a, data1=b, data2=c, op=None)\n")
        docs = tile_lint.call_docs(code, 4)
        self.assertIn("tensor_tensor(dst, data1, data2, op)", docs)
        self.assertIn("Elementwise op between two tiles", docs)

    def test_call_docs_is_silent_without_an_nki_call_or_line(self):
        self.assertEqual(tile_lint.call_docs("x = 1\n", 1), "")
        self.assertEqual(tile_lint.call_docs("x = 1\n", 0), "")

    def test_tile_operator_message_quotes_the_real_signature(self):
        found = tile_lint.lint_tile_operators(LEVEL4_BAD)
        self.assertIn("tensor_tensor", found[0]["msg"])


class TestControllerV7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rl_trace_agent as rl
        cls.rl = rl

    def args(self, **kw):
        base = dict(mode="reflect", hints="shape", terse=0, trace=False, plain_restart=True,
                    model_card=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_fresh_prompt_omits_the_failing_code_and_keeps_the_analysis(self):
        code = "def secret_kernel_marker(): pass"
        repair = self.rl.build_prompt(self.args(), 3, "reflect", code, "boom", "boom", diagnosis="ANALYSIS-X",
                                      diag_src="grounded")
        fresh = self.rl.build_prompt(self.args(), 3, "reflect", code, "boom", "boom", diagnosis="ANALYSIS-X",
                                     diag_src="grounded", fresh=True)
        self.assertIn("secret_kernel_marker", repair)
        self.assertNotIn("secret_kernel_marker", fresh)
        self.assertIn("ANALYSIS-X", fresh)
        self.assertNotIn("Several replies in a row failed", fresh)   # that sentence belongs to a restart

    def test_model_card_is_opt_in(self):
        off = self.rl.build_prompt(self.args(), 1, "direct")
        on = self.rl.build_prompt(self.args(model_card=True), 1, "direct")
        self.assertNotIn("TILE MODEL", off)
        self.assertIn("TILE MODEL", on)

    def test_model_card_is_level_agnostic(self):
        for word in ("pool", "transpose", "attention", "softmax"):
            self.assertNotIn(word, self.rl.TILE_MODEL_CARD.lower())

    def test_lint_api_names_the_sibling_module(self):
        saved = {k: sys.modules.get(k) for k in ("nki", "nki.language", "nki.isa")}
        sys.modules.update(fake_nki())
        self.rl._lookup.cache_clear()
        try:
            issues = self.rl.lint_api("import nki.isa as nisa\nx = nisa.divide\n")
        finally:
            for k, v in saved.items():
                sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
            self.rl._lookup.cache_clear()
        self.assertTrue(issues and "`nl.divide`" in issues[0]["msg"])


if __name__ == "__main__":
    unittest.main(verbosity=1)