#!/usr/bin/env python3
"""
tile_lint.py -- level-agnostic static checks for an NKI kernel, and grounded docs for a failing call.

Why this exists. The level 1/3/4 traces (reflect/shape, Qwen3-8B) do not fail because the model lacks
an algorithm. They fail because it does not know the NKI programming model, and the simulator
reveals ONE violation per run, so the model fixes one and meets the next:

    level 4:  tile_psum[m, n] += tile_sbuf[p, k] * tile_sbuf[p, n]      NameError 'n' ... then
                                                                        TypeError: tile * tile ...
    level 3:  nisa.tensor_copy(dst=<shared_hbm output>, ...)            wrong buffer, found one round later
    level 1:  nisa.divide                                               `nl.divide` exists; the "closest
                                                                        names" offered were unrelated

Every check below is about the LANGUAGE, not about any level's operation, so none of it changes with
the level. Each finding has the same shape as `agent.lint_api`'s ({line, src, msg, name, sig}) so the
controller can attach it to feedback the same way. All of it is advisory and is attached only to
kernels that already failed: the simulator stays the judge.

The buffer rules are copied from messages the checker itself printed in the traces
(`moving must be in ['sbuf']`, `tensor_copy dst must be in ['sbuf', 'psum']`). They are not guesses.

No Neuron SDK needed for the checks (pure `ast`). `call_docs` and `sibling_hint` use the SDK when it
is importable and return "" when it is not.
"""

import ast
import builtins
import difflib
import importlib
import inspect

BUFFER_KINDS = {"sbuf", "psum", "shared_hbm", "private_hbm", "hbm"}

# (positional order, {argument: allowed buffers}). Source: the checker's own assertion messages.
BUFFER_RULES = {
    "nki.isa.tensor_copy": (("dst", "src"), {"dst": {"sbuf", "psum"}, "src": {"sbuf", "psum"}}),
    "nki.isa.nc_matmul": (("dst", "stationary", "moving"),
                          {"dst": {"psum"}, "stationary": {"sbuf"}, "moving": {"sbuf"}}),
}

ALLOC_LEAVES = {"ndarray", "zeros", "ones", "full", "empty"}
BINOP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
                ast.MatMult: "@"}

# Names that look like an arithmetic helper on integer sizes. There is no such helper in nl.
SIZE_HELPERS = ("ceil_div", "cdiv", "div_ceil", "ceildiv", "ceil_divide")


# ---------------------------------------------------------------------------- shared helpers

def _aliases(tree):
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                alias[a.asname or a.name] = f"{node.module}.{a.name}"
    return alias


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve(dotted, alias):
    if not dotted:
        return None
    head, _, rest = dotted.partition(".")
    if head not in alias:
        return None
    full = alias[head] + ("." + rest if rest else "")
    return full if full.split(".")[0] == "nki" else None


def _issue(lines, node, msg, name="", sig=""):
    n = getattr(node, "lineno", 0)
    src = lines[n - 1].strip()[:100] if 0 < n <= len(lines) else ""
    return {"line": n, "src": src, "msg": msg, "name": name, "sig": sig}


def _parse(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _kernel_functions(tree):
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


# ---------------------------------------------------------------------------- 1. undefined names

def lint_undefined_names(code):
    """A name that is read but never bound anywhere in the file. Flow-insensitive on purpose: a
    name bound anywhere counts as bound, so this has no false positives from branches or loops.
    Catches `NameError: name 'n' is not defined` with its line, before the simulator."""
    tree = _parse(code)
    if tree is None:
        return []
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None:
                        bound.add(arg.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                if arg is not None:
                    bound.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    lines = code.splitlines()
    by_line = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            by_line.setdefault(node.lineno, {}).setdefault(node.id, node)
    out = []
    for n, names in sorted(by_line.items()):
        listed = ", ".join(f"`{k}`" for k in names)
        first = next(iter(names.values()))
        out.append(_issue(
            lines, first,
            f"{listed} {'is' if len(names) == 1 else 'are'} used but never defined anywhere in the file. "
            f"Define {'it' if len(names) == 1 else 'each'} first: a loop variable from "
            f"`for i in nl.affine_range(...)`, or an assignment.", name=",".join(names)))
    return out


# ---------------------------------------------------------------------------- 2. tile operators

def _tile_names(fn, alias):
    """Names holding a tile or tensor: the kernel's parameters and anything bound to an allocation
    (`nl.ndarray`, `nl.zeros`, ...). Conservative: results of other calls are not tracked."""
    params = {a.arg for a in fn.args.args + fn.args.posonlyargs}
    tiles = set()
    for node in ast.walk(fn):               # a parameter is a tensor only if the code treats it as one
        if isinstance(node, ast.Attribute) and node.attr in ("shape", "dtype") \
                and isinstance(node.value, ast.Name) and node.value.id in params:
            tiles.add(node.value.id)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in params:
            tiles.add(node.value.id)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            full = _resolve(_dotted(node.value.func), alias)
            if full and full.rsplit(".", 1)[-1] in ALLOC_LEAVES:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        tiles.add(t.id)
    return tiles


def _is_tile(node, tiles):
    if isinstance(node, ast.Name):
        return node.id in tiles
    if isinstance(node, ast.Subscript):
        return _is_tile(node.value, tiles)
    return False


def _mentions_tile(node, tiles):
    """True if any operand anywhere inside the expression is a tile."""
    return any(_is_tile(n, tiles) for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Subscript)))


def _sig_text(full):
    try:
        mod_name, _, attr = full.rpartition(".")
        return f"{full}{inspect.signature(getattr(importlib.import_module(mod_name), attr))}"
    except Exception:
        return ""


def lint_tile_operators(code):
    """`a[i] * b[j]` or `acc[i, j] += a[i] * b[j]` where the operands are tiles. The checker raises
    `TypeError: unsupported operand type(s) for *: 'NkiTensor' and 'NkiTensor'`; this says where, and
    that the fix is an `nisa` op rather than a rewritten expression."""
    tree = _parse(code)
    if tree is None:
        return []
    alias, lines, out = _aliases(tree), code.splitlines(), []
    for fn in _kernel_functions(tree):
        tiles = _tile_names(fn, alias)
        for node in ast.walk(fn):
            sym = None
            if isinstance(node, ast.AugAssign) and _is_tile(node.target, tiles) \
                    and _mentions_tile(node.value, tiles):
                sym = BINOP_SYMBOL.get(type(node.op), "?") + "="
            elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SYMBOL \
                    and _is_tile(node.left, tiles) and _is_tile(node.right, tiles):
                sym = BINOP_SYMBOL[type(node.op)]
            if sym:
                sig = _sig_text("nki.isa.tensor_tensor")
                out.append(_issue(
                    lines, node,
                    f"`{sym}` between two tiles is not supported (a tile is not a number). Combine "
                    f"tiles with an `nisa` op"
                    + (f", for example `{sig}`" if sig else "")
                    + "; for a running sum over a loop, let `nisa.nc_matmul` accumulate into one PSUM "
                      "tile instead of adding in Python."))
    # A statement can match as both AugAssign and the BinOp inside it: keep one per line.
    seen, uniq = set(), []
    for i in sorted(out, key=lambda i: i["line"]):
        if i["line"] not in seen:
            seen.add(i["line"])
            uniq.append(i)
    return uniq


# ---------------------------------------------------------------------------- 3. buffer kinds

def _buffer_kind(call, alias):
    for kw in call.keywords:
        if kw.arg == "buffer":
            d = _dotted(kw.value)
            leaf = d.rsplit(".", 1)[-1] if d else ""
            return leaf if leaf in BUFFER_KINDS else None
    return None


def lint_buffer_kinds(code):
    """Pass a tile allocated in one memory to an op that needs another, e.g. `tensor_copy` into the
    `shared_hbm` output, or `nc_matmul` with a PSUM tile as `moving`. Kinds follow names through
    slicing. The kernel's own parameters are HBM."""
    tree = _parse(code)
    if tree is None:
        return []
    alias, lines, out = _aliases(tree), code.splitlines(), []
    for fn in _kernel_functions(tree):
        kind = {a.arg: "hbm" for a in fn.args.args + fn.args.posonlyargs}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                full = _resolve(_dotted(node.value.func), alias)
                if full and full.rsplit(".", 1)[-1] in ALLOC_LEAVES:
                    k = _buffer_kind(node.value, alias)
                    if k:
                        for t in node.targets:
                            if isinstance(t, ast.Name):
                                kind[t.id] = k

        def kind_of(expr):
            while isinstance(expr, ast.Subscript):
                expr = expr.value
            return kind.get(expr.id) if isinstance(expr, ast.Name) else None

        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            full = _resolve(_dotted(node.func), alias)
            if full not in BUFFER_RULES:
                continue
            order, allowed = BUFFER_RULES[full]
            given = dict(zip(order, node.args))
            given.update({k.arg: k.value for k in node.keywords if k.arg})
            leaf = full.rsplit(".", 1)[-1]
            for arg, ok in allowed.items():
                k = kind_of(given[arg]) if arg in given else None
                if k and k not in ok:
                    want = " or ".join(sorted(ok))
                    if k in ("shared_hbm", "private_hbm", "hbm") and leaf == "tensor_copy":
                        fix = (" `tensor_copy` only moves data between on-chip buffers. Reach HBM with "
                               "`nisa.dma_copy` (sbuf -> hbm).")
                    else:
                        fix = (f" Allocate it with buffer=nl.{sorted(ok)[0]}, and move data between "
                               f"memories with `nisa.dma_copy` (hbm <-> sbuf) or "
                               f"`nisa.tensor_copy` (sbuf <-> psum).")
                    out.append(_issue(
                        lines, node,
                        f"`nisa.{leaf}` needs `{arg}` in {want}, but this `{arg}` lives in {k}." + fix,
                        name=f"nisa.{leaf}", sig=f"nisa.{leaf} {arg} in {want}"))
    out.sort(key=lambda i: i["line"])
    return out


# ---------------------------------------------------------------------------- 4. missing names

def _public(mod):
    return [n for n in dir(mod) if not n.startswith("_")]


def sibling_hint(full):
    """For a missing `nki.isa.divide`: say what exists. Looks for the same leaf in the other nki
    modules, then by spelling, and flags size-arithmetic helpers that do not exist anywhere."""
    mod_name, _, leaf = full.rpartition(".")
    notes = []
    if leaf.lower() in SIZE_HELPERS or ("ceil" in leaf.lower() and "div" in leaf.lower()):
        notes.append("There is no integer-size helper in nki; compute sizes with plain Python, "
                     "e.g. `(a + b - 1) // b`.")
    found = []
    for other in ("nki.language", "nki.isa"):
        if other == mod_name:
            continue
        try:
            m = importlib.import_module(other)
        except Exception:
            continue
        if hasattr(m, leaf):
            found.append(f"{other.replace('nki.language', 'nl').replace('nki.isa', 'nisa')}.{leaf}")
    if found:
        notes.insert(0, "It exists as " + " and ".join(f"`{f}`" for f in found)
                     + " (a different module), so use that name instead.")
    elif not notes:
        try:
            m = importlib.import_module(mod_name)
            close = difflib.get_close_matches(leaf, _public(m), n=3, cutoff=0.6)
        except Exception:
            close = []
        if close:
            notes.append("Closest spelling: " + ", ".join(f"`{c}`" for c in close) + ".")
    return " ".join(notes)


# ---------------------------------------------------------------------------- 5. grounded docs

def call_docs(code, lineno, max_lines=9, limit=2):
    """The real signature and docstring of the nki call(s) on the failing line. Retrieval, not
    advice: the model is shown what the SDK says the call does (reduction axes, buffer rules,
    shape rules) at the moment it misused it. Returns "" if the SDK is not importable or the line
    holds no nki call."""
    tree = _parse(code)
    if tree is None or not lineno:
        return ""
    alias, chunks, done = _aliases(tree), [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node, "lineno", 0) <= lineno <= getattr(node, "end_lineno", node.lineno):
            full = _resolve(_dotted(node.func), alias)
            if not full or full in done:
                continue
            done.add(full)
            mod_name, _, attr = full.rpartition(".")
            try:
                fn = getattr(importlib.import_module(mod_name), attr)
                doc = inspect.getdoc(fn) or ""
                sig = str(inspect.signature(fn))
            except Exception:
                continue
            body = [ln.rstrip() for ln in doc.splitlines() if ln.strip()][:max_lines]
            if not body:
                continue
            chunks.append(f"`{attr}{sig}`: " + " ".join(s.strip() for s in body))
            if len(chunks) >= limit:
                break
    return ("SDK REFERENCE for the call on that line: " + " || ".join(chunks)) if chunks else ""


# ---------------------------------------------------------------------------- everything at once

def lint_tile_model(code):
    """All language-level findings in one list, sorted by line. One call, so the model hears about
    every violation at once instead of one per round."""
    found = []
    for f in (lint_undefined_names, lint_tile_operators, lint_buffer_kinds):
        found.extend(f(code))
    found.sort(key=lambda i: i["line"])
    return found


def format_findings(issues, limit=4):
    if not issues:
        return ""
    return "LANGUAGE CHECK (all violations found by reading the code, not only the first one the " \
           "simulator hits): " + " ".join(f"[line {i['line']}: `{i['src']}`] {i['msg']}"
                                          for i in issues[:limit])


if __name__ == "__main__":
    import sys
    for i in lint_tile_model(open(sys.argv[1]).read()):
        print(f"line {i['line']}: {i['msg']}")