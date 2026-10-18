"""Records how a piece of Python/NumPy code runs so the browser can replay it.

The code is executed under ``sys.settrace``. After every executed line we
store:

* the variables of every active frame (array values are de-duplicated into
  a shared ``blobs`` table, steps only hold indices into it),
* which array / list cells the line read and which it wrote,
* the line with the current values filled in, e.g.
  ``c[1, 2] = 0.31 + 0.62 * 0.41  ->  0.5642``,
* how much had been printed so far.

Entry point: ``trace_json(code, stdin, max_steps)`` returns the trace as JSON.
"""

import ast
import builtins
import copy
import inspect
import io
import json
import math
import sys
import types

import numpy as np

FILENAME = "<exercise>"
MAX_CELLS = 2500   # arrays bigger than this are summarised, not sent cell by cell
MAX_SEQ = 400      # longest Python list that is still drawn as a grid
MAX_MERGED = 500   # line events folded into one step (multi-line statements, comprehensions)
COMPREHENSIONS = {"<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>", "<lambda>"}
META_ATTRS = {"shape", "size", "ndim", "dtype", "itemsize", "nbytes"}
GENERATOR_FLAGS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR

_FAIL = object()

# Expressions made only of these nodes can be evaluated a second time (to find
# out which cells `a[i, k]` touches, or what value it has) without side effects.
_SAFE_NODES = (
    ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Attribute, ast.Call,
    ast.keyword, ast.IfExp,
)
_SAFE_BUILTINS = (len, abs, min, max, round, int, float, complex, bool, range, sum, pow, divmod)
_SAFE_METHODS = {
    "sum", "max", "min", "mean", "dot", "transpose", "copy", "reshape", "flatten",
    "ravel", "item", "tolist", "trace", "diagonal", "argmax", "argmin", "prod",
    "std", "var", "round", "astype", "conj", "all", "any",
}
_UNSAFE_NUMPY = {
    "fill_diagonal", "copyto", "put", "place", "putmask", "put_along_axis",
    "save", "savez", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile", "tofile",
}


class StepLimit(BaseException):
    """Raised from the trace hook to stop programs that run too long."""


class _KeyProbe:
    """`probe[<index>]` returns the index itself, so slices can be evaluated."""

    def __getitem__(self, key):
        return key


_PROBE_NAME = "__mn_probe__"
_PROBE = _KeyProbe()


# ---------------------------------------------------------------- values

def _is_number(value):
    return isinstance(value, (bool, int, float, complex, np.number, np.bool_))


def _is_scalar(value):
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return _is_number(value) or (isinstance(value, str) and len(value) <= 40)


def _shown(name, value):
    if name.startswith("__") or name.startswith("."):
        return False
    return not (isinstance(value, types.ModuleType) or callable(value))


def _num(value):
    """A cell value as something JSON can hold (no NaN / inf / complex)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (complex, np.complexfloating)):
        return str(complex(value))
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _fmt_scalar(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (complex, np.complexfloating)):
        return str(complex(value))
    if isinstance(value, str):
        return repr(value)
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value != 0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
        return f"{value:.4g}"
    text = f"{value:.4f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _fmt_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 1:
        if value.size <= 6:
            return "[" + ", ".join(_fmt_scalar(v) for v in value) + "]"
        return f"array of {value.size}"
    if isinstance(value, np.ndarray) and value.ndim > 1:
        return "×".join(str(n) for n in value.shape) + " array"
    if _is_scalar(value):
        return _fmt_scalar(value)
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        if len(value) <= 6 and all(_is_scalar(v) for v in value):
            inner = ", ".join(_fmt_scalar(v) for v in value)
            return f"[{inner}]" if isinstance(value, list) else f"({inner})"
        return f"{type(value).__name__} of {len(value)}"
    return type(value).__name__


def _grid_shape(value):
    """Shape of `value` if it can be drawn as a 1-D / 2-D grid, else None."""
    if isinstance(value, np.ndarray):
        return value.shape if value.ndim in (1, 2) else None
    if isinstance(value, (list, tuple)) and 0 < len(value) <= MAX_SEQ:
        if all(_is_number(v) for v in value):
            return (len(value),)
        if all(isinstance(row, (list, tuple)) for row in value):
            width = len(value[0])
            if (width and len(value) * width <= MAX_CELLS
                    and all(len(row) == width and all(_is_number(v) for v in row) for row in value)):
                return (len(value), width)
    return None


def _cells(value, index_values):
    """Flat indices of the cells that `value[i0][i1]...` touches."""
    shape = _grid_shape(value)
    if shape is None:
        return None
    size = int(np.prod(shape))
    if size > MAX_CELLS:
        return None
    selection = np.arange(size).reshape(shape)
    try:
        for index in index_values:
            selection = selection[index]
    except Exception:
        return None
    return [int(i) for i in np.atleast_1d(selection).ravel()]


def _cell_list(array):
    kind = array.dtype.kind
    if kind == "c":
        return [_num(v) for v in array.ravel()]
    data = array.ravel().tolist()
    if kind == "f" and not np.isfinite(array).all():
        data = [v if math.isfinite(v) else str(v) for v in data]
    return data


def _blob(value):
    """A JSON-friendly description of a variable's value."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _blob(value.item())
        info = {"t": "arr", "kind": "ndarray", "shape": [int(n) for n in value.shape],
                "dtype": str(value.dtype)}
        if value.ndim > 2 or value.size > MAX_CELLS or value.dtype.kind not in "biufc":
            info["t"] = "big"
            info["text"] = np.array2string(value, threshold=30, edgeitems=3)
        else:
            info["d"] = _cell_list(value)
        return info
    if _is_number(value):
        return {"t": "num", "v": _num(value), "py": type(value).__name__}
    if isinstance(value, str):
        return {"t": "str", "v": value[:200]}
    if value is None:
        return {"t": "none"}
    shape = _grid_shape(value)
    if shape is not None:
        flat = list(value) if len(shape) == 1 else [v for row in value for v in row]
        return {"t": "arr", "kind": type(value).__name__, "shape": list(shape), "dtype": "",
                "d": [_num(v) for v in flat]}
    if isinstance(value, (list, tuple, dict, set)):
        text = repr(value)
        return {"t": "repr", "v": text if len(text) <= 200 else text[:197] + "...",
                "py": type(value).__name__}
    return {"t": "obj", "v": type(value).__name__}


def _frame_label(code):
    labels = {"<module>": "Global", "<genexpr>": "generator expression",
              "<listcomp>": "list comprehension", "<lambda>": "lambda"}
    return labels.get(code.co_name, code.co_name + "()")


# ---------------------------------------------------------------- source

def _neutralise_shell_lines(code):
    """Turn notebook-only lines (`!pip ...`, `%timeit`) into `pass`, keeping line numbers."""
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    return "\n".join(lines)


def _line_map(tree):
    """line number -> (statement that owns it, [first, last] lines to highlight),
    plus the lines holding a list/set/dict comprehension."""
    owners, comprehension_lines = {}, set()
    for node in ast.walk(tree):  # breadth first, so inner statements win
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            comprehension_lines.update(range(node.lineno, node.end_lineno + 1))
        if not isinstance(node, ast.stmt):
            continue
        first, last = node.lineno, node.end_lineno
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            last = max(first, body[0].lineno - 1)  # only the header of if/for/def
        for line in range(first, last + 1):
            owners[line] = (node, [first, last])
    return owners, comprehension_lines


def _unparse(node):
    """ast.unparse without the parentheses around a bare tuple (`x, y = 3, 2`)."""
    text = ast.unparse(node)
    if isinstance(node, ast.Tuple) and text.startswith("(") and text.endswith(")"):
        return text[1:-1]
    return text


def _is_literal(node):
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        return False


def _stmt_parts(stmt):
    """(expressions the statement reads, targets it writes)."""
    if isinstance(stmt, ast.Assign):
        return [stmt.value], list(stmt.targets)
    if isinstance(stmt, ast.AugAssign):
        return [stmt.value, stmt.target], [stmt.target]
    if isinstance(stmt, ast.AnnAssign):
        return ([stmt.value], [stmt.target]) if stmt.value else ([], [])
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter], [stmt.target]
    if isinstance(stmt, (ast.If, ast.While, ast.Assert)):
        return [stmt.test], []
    if isinstance(stmt, (ast.Expr, ast.Return)):
        return ([stmt.value] if stmt.value else []), []
    if isinstance(stmt, ast.Delete):
        return [], list(stmt.targets)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return ([item.context_expr for item in stmt.items],
                [item.optional_vars for item in stmt.items if item.optional_vars])
    return [], []


def _subscript_chain(node):
    """`a[i][j]` -> ("a", [i, j]); None when the base is not a plain name."""
    slices = []
    while isinstance(node, ast.Subscript):
        slices.append(node.slice)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, slices[::-1]
    return None


def _is_name_chain(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _as_load(node):
    node = copy.deepcopy(node)
    for sub in ast.walk(node):
        if hasattr(sub, "ctx"):
            sub.ctx = ast.Load()
    return node


def _lit(value):
    """A node that unparses as the formatted value."""
    text = _fmt_scalar(value)
    if text.startswith("-"):
        text = f"({text})"
    return ast.Name(id=text, ctx=ast.Load())


def _body_range(body):
    return body[0].lineno, max(node.end_lineno for node in body)


def _callable_ok(fn):
    try:
        if any(fn is b for b in _SAFE_BUILTINS):
            return True
        if isinstance(fn, np.ufunc):
            return True
        name = getattr(fn, "__name__", "")
        if name in _UNSAFE_NUMPY:
            return False
        module = getattr(fn, "__module__", None) or ""
        if module.startswith("numpy") and not module.startswith("numpy.random"):
            return True
        owner = getattr(fn, "__self__", None)
        return isinstance(owner, np.ndarray) and name in _SAFE_METHODS
    except Exception:
        return False


class _Substituter(ast.NodeTransformer):
    """Rewrites an expression with the current values of scalars and cells."""

    def __init__(self, tracer, frame):
        self.tracer = tracer
        self.frame = frame
        self.local_vars = frame.f_locals

    def expr(self, node):
        return self.visit(copy.deepcopy(node))

    def target(self, node):
        """`c[i, j]` -> `c[1, 2]` (keeps the name being assigned)."""
        node = copy.deepcopy(node)
        if isinstance(node, ast.Subscript):
            return self._indices_only(node)
        if isinstance(node, (ast.Tuple, ast.List)):
            node.elts = [self.target(e) for e in node.elts]
        return node

    def _indices_only(self, node):
        if isinstance(node, ast.Subscript):
            node.slice = self.visit(node.slice)
            node.value = self._indices_only(node.value)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.local_vars:
                value = self.local_vars[node.id]
            elif node.id in self.frame.f_globals:
                value = self.frame.f_globals[node.id]
            else:
                return node
            if _is_scalar(value):
                return _lit(value)
        return node

    def visit_Subscript(self, node):
        value = self.tracer._safe_eval(_as_load(node), self.frame)
        if value is not _FAIL and _is_scalar(value):
            return _lit(value)
        return self._indices_only(node)

    def visit_Call(self, node):
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        return node

    def _keep(self, node):
        return node

    # Names inside these are not the frame's variables (or are attribute names).
    visit_Attribute = visit_JoinedStr = visit_Lambda = _keep
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _keep


# ---------------------------------------------------------------- tracer

class Tracer:
    def __init__(self, code, stdin="", max_steps=3000):
        self.code = code
        self.stdin = [line for line in stdin.splitlines()]
        self.max_steps = max(1, int(max_steps))
        self.stdout = io.StringIO()
        self.steps = []
        self.blobs = []
        self.blob_ids = {}
        self.array_cache = {}
        self.uids = {}
        self.frame_refs = {}
        self.next_uid = 0
        self.module_uid = None
        self.pending = None
        self.exc_frame = None
        self.truncated = False
        self.inputs_used = []
        self.line_map = {}
        self.comprehension_lines = set()

    # -- running

    def run(self):
        source = _neutralise_shell_lines(self.code)
        try:
            tree = ast.parse(source, FILENAME)
            compiled = compile(tree, FILENAME, "exec")
        except SyntaxError as exc:
            return self._result({"type": "SyntaxError", "msg": exc.msg, "line": exc.lineno})
        self.line_map, self.comprehension_lines = _line_map(tree)

        env = {
            "__name__": "__main__",
            "__builtins__": dict(vars(builtins), input=self._input),
            "display": lambda *values, **_: print(*values),
        }
        error = None
        saved = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self.stdout
        try:
            sys.settrace(self._on_call)
            exec(compiled, env)
        except StepLimit:
            pass
        except SystemExit:
            pass
        except BaseException as exc:  # the exercise's own error: report it
            error = _describe(exc)
        finally:
            sys.settrace(None)
            sys.stdout, sys.stderr = saved
        if self.pending is not None and error is not None:
            self.pending["error"] = True
            try:
                self._finish(self.pending["frame"])
            except StepLimit:
                pass
        return self._result(error)

    def _result(self, error):
        if error is not None and self.steps:
            self.steps[-1]["error"] = True
        return {
            "steps": self.steps,
            "blobs": self.blobs,
            "stdout": self.stdout.getvalue(),
            "error": error,
            "truncated": self.truncated,
            "maxSteps": self.max_steps,
            "inputs": self.inputs_used,
        }

    def _input(self, prompt=""):
        prompt = str(prompt)
        self.stdout.write(prompt)
        if self.stdin:
            value = self.stdin.pop(0)
        else:
            value = _ask_browser(prompt)
            if value is None:
                raise EOFError("input() needs a value - add it to the Input box")
        self.inputs_used.append(value)
        self.stdout.write(value + "\n")
        return value

    # -- trace hooks

    def _on_call(self, frame, event, arg):
        if frame.f_code.co_filename != FILENAME:
            return None
        uid = self._uid(frame)
        if self.module_uid is None and frame.f_code.co_name == "<module>":
            self.module_uid = uid
        return self._on_event

    def _on_event(self, frame, event, arg):
        if event == "line":
            self.exc_frame = None
            if self._same_statement(frame):
                self.pending["merged"] += 1
                return self._on_event
            self._finish(frame)
            self._begin(frame)
        elif event == "exception":
            self.exc_frame = frame
        elif event == "return":
            unwinding = self.exc_frame is frame
            if unwinding and self.pending is not None:
                self.pending["error"] = True
            self._finish(frame, ret=(frame, arg))
            caller = frame.f_back
            if not frame.f_code.co_flags & GENERATOR_FLAGS:
                self._forget(frame)
                if (not unwinding and caller is not None
                        and caller.f_code.co_filename == FILENAME):
                    self._begin(caller, resumed=True)
        return self._on_event

    def _same_statement(self, frame):
        """True when this line event is still the pending statement: the next line
        of a multi-line statement, or another round of an inlined comprehension."""
        p = self.pending
        if p is None or p["frame"] is not frame or p["merged"] >= MAX_MERGED:
            return False
        stmt = self.line_map.get(frame.f_lineno, (None,))[0]
        if stmt is None or stmt is not p["stmt"] or isinstance(getattr(stmt, "body", None), list):
            return False
        return (frame.f_lineno != p["line"] or p["merged"] > 0
                or frame.f_lineno in self.comprehension_lines)

    def _uid(self, frame):
        key = id(frame)
        uid = self.uids.get(key)
        if uid is None:
            uid = self.uids[key] = self.next_uid
            self.next_uid += 1
            self.frame_refs[key] = frame
        return uid

    def _forget(self, frame):
        self.uids.pop(id(frame), None)
        self.frame_refs.pop(id(frame), None)

    # -- one step = one executed line

    def _begin(self, frame, resumed=False):
        line = frame.f_lineno
        stmt, span = self.line_map.get(line, (None, [line, line]))
        accesses, explain = [], None
        if frame.f_code.co_name in COMPREHENSIONS:
            accesses = None  # its loop variable only moves on after this event: read at yield
        elif stmt is not None:
            try:
                accesses = self._accesses(stmt, frame, writes_only=resumed)
            except Exception:
                accesses = []
        if stmt is not None:
            try:
                explain = self._explain_before(stmt, frame)
            except Exception:
                explain = None
        self.pending = {
            "frame": frame, "uid": self._uid(frame), "line": line, "span": span,
            "stmt": stmt, "accesses": accesses, "explain": explain, "resumed": resumed,
            "error": False, "merged": 0, "loop_done": False,
        }

    def _finish(self, event_frame, ret=None):
        p = self.pending
        if p is None:
            return
        self.pending = None
        frame = p["frame"]
        called = ret is None and event_frame is not frame and event_frame.f_back is frame
        try:
            explain = self._explain_after(p, event_frame, ret, called)
        except Exception:
            explain = ""
        if p["resumed"] and not explain:
            explain = "back from the call"

        accesses = p["accesses"]
        if accesses is None and ret is not None and ret[1] is None:
            accesses = []  # a generator expression that has run out
        if accesses is None:
            try:
                accesses = self._accesses(p["stmt"], frame) if p["stmt"] is not None else []
            except Exception:
                accesses = []
        if called:  # assignments of this line happen when the call returns
            accesses = [a for a in accesses if a[0] == "r"]
        if p["loop_done"]:  # the loop variable keeps its last value
            accesses = [a for a in accesses if a[0] == "r"]

        reads, writes = {}, {}
        for mode, uid, name, cells in accesses:
            bucket = reads if mode == "r" else writes
            key = (uid, name)
            if key in bucket and (bucket[key] is None or cells is None):
                bucket[key] = None
            elif key in bucket:
                bucket[key] = sorted(set(bucket[key]) | set(cells))
            else:
                bucket[key] = cells

        ret_uid, ret_value = (self._uid(ret[0]), ret[1]) if ret else (None, None)
        step = {
            "line": p["line"],
            "span": p["span"],
            "frame": p["uid"],
            "reads": [[uid, name, cells] for (uid, name), cells in reads.items()],
            "writes": [[uid, name, cells] for (uid, name), cells in writes.items()],
            "explain": explain,
            "out": self.stdout.tell(),
            "state": self._snapshot(event_frame, ret_uid, ret_value, p),
        }
        if p["resumed"]:
            step["resumed"] = True
        if p["error"]:
            step["error"] = True
        self.steps.append(step)
        if len(self.steps) >= self.max_steps:
            self.truncated = True
            raise StepLimit()

    def _snapshot(self, frame, ret_uid, ret_value, pending):
        frames = []
        while frame is not None:
            if frame.f_code.co_filename == FILENAME:
                frames.append(frame)
            frame = frame.f_back
        state = []
        for f in reversed(frames):
            uid = self._uid(f)
            scope = f.f_globals if f.f_code.co_name == "<module>" else f.f_locals
            variables = [[name, self._intern(uid, name, value)]
                         for name, value in list(scope.items()) if _shown(name, value)]
            if uid == ret_uid and not pending["error"] and f.f_code.co_name != "<module>":
                variables.append(["return", self._intern(uid, "return", ret_value)])
            state.append({"uid": uid, "name": _frame_label(f.f_code), "vars": variables})
        return state

    def _intern(self, uid, name, value):
        """Index of `value` in the blob table, reusing unchanged arrays."""
        if (isinstance(value, np.ndarray) and value.dtype.kind in "biufc"
                and 0 < value.ndim <= 2 and value.size <= MAX_CELLS):
            cached = self.array_cache.get((uid, name))
            if cached is not None:
                previous, index = cached
                if (previous.shape == value.shape and previous.dtype == value.dtype
                        and np.array_equal(previous, value, equal_nan=value.dtype.kind in "fc")):
                    return index
            index = self._store(_blob(value))
            self.array_cache[(uid, name)] = (value.copy(), index)
            return index
        return self._store(_blob(value))

    def _store(self, blob):
        key = json.dumps(blob, sort_keys=True)
        index = self.blob_ids.get(key)
        if index is None:
            index = self.blob_ids[key] = len(self.blobs)
            self.blobs.append(blob)
        return index

    # -- which cells a line touches

    def _accesses(self, stmt, frame, writes_only=False):
        out = []
        if frame.f_code.co_name in COMPREHENSIONS:
            # Inside a comprehension frame: look at the comprehension's own expressions.
            for node in ast.walk(stmt):
                if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                    parts = [node.elt]
                elif isinstance(node, ast.DictComp):
                    parts = [node.key, node.value]
                elif isinstance(node, ast.Lambda):
                    parts = [node.body]
                else:
                    continue
                for generator in getattr(node, "generators", []):
                    parts.extend(generator.ifs)
                for part in parts:
                    self._collect(part, "r", frame, out, inside_comprehension=True)
            return out
        reads, writes = _stmt_parts(stmt)
        if not writes_only:
            for node in reads:
                self._collect(node, "r", frame, out)
        for node in writes:
            self._collect(node, "w", frame, out)
        return out

    def _collect(self, node, mode, frame, out, inside_comprehension=False):
        if node is None:
            return
        if isinstance(node, ast.Subscript):
            chain = _subscript_chain(node)
            if chain is not None:
                name, slices = chain
                uid, value = self._lookup(frame, name)
                if uid is not None and _grid_shape(value) is not None:
                    indices = [self._eval_index(s, frame) for s in slices]
                    if all(i is not _FAIL for i in indices):
                        cells = _cells(value, indices)
                        if cells is not None:
                            out.append((mode, uid, name, cells))
                for s in slices:
                    self._collect(s, "r", frame, out, inside_comprehension)
                return
        elif isinstance(node, ast.Name):
            uid, value = self._lookup(frame, node.id)
            if mode == "w":
                out.append(("w", self._store_owner(frame, node.id), node.id, None))
            elif uid is not None and _shown(node.id, value):
                out.append(("r", uid, node.id, None))
            return
        elif isinstance(node, (ast.Tuple, ast.List)) and mode == "w":
            for element in node.elts:
                self._collect(element, "w", frame, out)
            return
        elif isinstance(node, ast.Starred):
            self._collect(node.value, mode, frame, out, inside_comprehension)
            return
        elif isinstance(node, ast.Attribute):
            if node.attr not in META_ATTRS:
                self._collect(node.value, "r", frame, out, inside_comprehension)
            return
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "len":
                return
        elif isinstance(node, (ast.Lambda, ast.GeneratorExp, ast.ListComp,
                               ast.SetComp, ast.DictComp)) and not inside_comprehension:
            # Loop variables of a comprehension don't exist yet: mark whole arrays.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    uid, value = self._lookup(frame, sub.id)
                    if uid is not None and _grid_shape(value) is not None:
                        out.append(("r", uid, sub.id, None))
            return
        for child in ast.iter_child_nodes(node):
            self._collect(child, "r", frame, out, inside_comprehension)

    def _lookup(self, frame, name):
        """(uid of the frame that owns `name`, its value) or (None, None)."""
        code = frame.f_code
        if code.co_name != "<module>":
            local_vars = frame.f_locals
            if name in local_vars:
                if name in code.co_freevars:  # closure: owned by an enclosing frame
                    outer = frame.f_back
                    while outer is not None:
                        if (outer.f_code.co_filename == FILENAME
                                and name in outer.f_code.co_cellvars):
                            return self._uid(outer), local_vars[name]
                        outer = outer.f_back
                return self._uid(frame), local_vars[name]
        if name in frame.f_globals and self.module_uid is not None:
            return self.module_uid, frame.f_globals[name]
        return None, None

    def _store_owner(self, frame, name):
        code = frame.f_code
        if code.co_name == "<module>" or name in code.co_varnames or name in code.co_cellvars:
            return self._uid(frame)
        if name in code.co_freevars:
            uid, _ = self._lookup(frame, name)
            if uid is not None:
                return uid
        return self.module_uid

    # -- evaluating small expressions again

    def _is_safe(self, node, frame):
        for sub in ast.walk(node):
            if not isinstance(sub, _SAFE_NODES):
                return False
            if isinstance(sub, ast.Attribute) and sub.attr.startswith("_"):
                return False
            if isinstance(sub, ast.Call):
                if any(k.arg == "out" for k in sub.keywords) or not _is_name_chain(sub.func):
                    return False
                fn = self._eval(sub.func, frame)
                if fn is _FAIL or not _callable_ok(fn):
                    return False
        return True

    def _eval(self, node, frame, extra=None):
        try:
            code = compile(ast.fix_missing_locations(ast.Expression(body=node)), "<probe>", "eval")
            local_vars = frame.f_locals
            if extra:
                local_vars = dict(local_vars, **extra)
            return eval(code, frame.f_globals, local_vars)
        except Exception:
            return _FAIL

    def _safe_eval(self, node, frame):
        if not self._is_safe(node, frame):
            return _FAIL
        return self._eval(node, frame)

    def _eval_index(self, slice_node, frame):
        """Value of the `[...]` part of a subscript, slices included."""
        probe = ast.Subscript(value=ast.Name(id=_PROBE_NAME, ctx=ast.Load()),
                              slice=_as_load(slice_node), ctx=ast.Load())
        ast.copy_location(probe, slice_node)
        if not self._is_safe(probe, frame):
            return _FAIL
        return self._eval(probe, frame, {_PROBE_NAME: _PROBE})

    # -- the "c[1, 2] = 0.31 + 0.62 * 0.41  ->  0.56" line

    def _explain_before(self, stmt, frame):
        if frame.f_code.co_name in COMPREHENSIONS:
            return {"kind": "comprehension"}
        sub = _Substituter(self, frame)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if getattr(stmt, "value", None) is None:
                return None
            return {"kind": "assign", "targets": targets, "literal": _is_literal(stmt.value),
                    "lhs": " = ".join(_unparse(sub.target(t)) for t in targets),
                    "rhs": _unparse(sub.expr(stmt.value))}
        if isinstance(stmt, ast.AugAssign):
            current = sub.expr(_as_load(stmt.target))
            rhs = ast.BinOp(left=current, op=stmt.op, right=sub.expr(stmt.value))
            return {"kind": "assign", "targets": [stmt.target], "literal": False,
                    "lhs": _unparse(sub.target(stmt.target)), "rhs": _unparse(rhs)}
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            return {"kind": "for", "target": stmt.target, "body": _body_range(stmt.body),
                    "iter": _unparse(sub.expr(stmt.iter))}
        if isinstance(stmt, (ast.If, ast.While)):
            return {"kind": "while" if isinstance(stmt, ast.While) else "if",
                    "body": _body_range(stmt.body), "test": ast.unparse(sub.expr(stmt.test))}
        if isinstance(stmt, ast.Return):
            value = _unparse(sub.expr(stmt.value)) if stmt.value else "None"
            return {"kind": "return", "value": value}
        if isinstance(stmt, ast.Expr):
            return {"kind": "expr", "text": ast.unparse(sub.expr(stmt.value)),
                    "plain": ast.unparse(stmt.value)}
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {"kind": "text", "text": f"define function {stmt.name}()"}
        if isinstance(stmt, ast.ClassDef):
            return {"kind": "text", "text": f"define class {stmt.name}"}
        return None

    def _explain_after(self, p, event_frame, ret, called):
        if called:
            return "→ entering " + _frame_label(event_frame.f_code)
        e = p["explain"]
        if e is None:
            return ""
        frame = p["frame"]
        next_line = event_frame.f_lineno if event_frame is frame and ret is None else None
        kind = e["kind"]
        if kind == "comprehension":
            if not ret:
                return ""
            return "generator finished" if ret[1] is None else "produces " + _fmt_value(ret[1])
        if kind == "text":
            return e["text"]
        if kind == "assign":
            text = f"{e['lhs']} = {e['rhs']}"
            if p["error"] or e["literal"]:
                return text
            value = self._safe_eval(_as_load(e["targets"][0]), frame)
            if value is not _FAIL:
                result = _fmt_value(value)
                if result != e["rhs"]:
                    text += "  →  " + result
            return text
        if kind == "for":
            target = _unparse(e["target"])
            header = f"for {target} in {e['iter']}"
            low, high = e["body"]
            if next_line is not None and low <= next_line <= high:
                value = self._safe_eval(_as_load(e["target"]), frame)
                shown = _fmt_value(value) if value is not _FAIL else "?"
                return f"{header}  →  {target} = {shown}"
            p["loop_done"] = True
            return f"{header}  →  loop finished"
        if kind in ("if", "while"):
            low, high = e["body"]
            taken = next_line is not None and low <= next_line <= high
            outcome = "True" if taken else "False"
            if kind == "while":
                outcome += ", repeat the loop" if taken else ", loop ends"
            return f"{kind} {e['test']}  →  {outcome}"
        if kind == "return":
            text = "return " + e["value"]
            if ret and not p["error"]:
                result = _fmt_value(ret[1])
                if result != e["value"]:
                    text += "  →  " + result
            return text
        if kind == "expr":
            return e["text"] if e["text"] != e["plain"] else ""
        return ""


def _describe(exc):
    line = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == FILENAME:
            line = tb.tb_lineno
        tb = tb.tb_next
    return {"type": type(exc).__name__, "msg": str(exc), "line": line}


def _ask_browser(prompt):
    """Ask with a browser prompt() when the Input box ran out of values."""
    try:
        import js  # only exists under Pyodide
        value = js.prompt(prompt or "input()")
    except Exception:
        return None
    return None if value is None else str(value)


def trace_json(code, stdin="", max_steps=3000):
    return json.dumps(Tracer(code, stdin, max_steps).run(), default=str)
