"""Read both projects' contract sources as data, with the standard library only.

The workspace parity tests (`test_*.py` beside this file) hold the console's copies of the
contract facts equal to the simulator's contract modules and to `robots_specs/`
(workspace spec §1). They must run with any Python >= 3.11 and no venv, so they can import
neither project: the console needs numpy, OpenCV and roslibpy, and the simulator's
modules pull in numpy and MuJoCo at module scope.

So a module is **parsed**, never imported. `Module(path)["NAME"]` evaluates one top-level
binding on demand from its syntax tree, following only the bindings that expression
needs:

* literals, containers, arithmetic, comprehensions and calls of safe builtins evaluate
  directly;
* a name bound by `import`/`from ... import` resolves to a whitelisted stdlib module, or to
  another project module under the same root, which is parsed the same way; any other
  import (numpy, mujoco, cv2, ...) is unavailable, and so is every binding that needs it;
* a top-level `def` or `class` is compiled on its own -- not the module around it -- in a
  namespace holding only the names its body mentions. That is how a pure rule, such as
  namespace composition, is compared by behaviour rather than by its spelling.

A binding that cannot be evaluated this way raises `Unavailable` with the reason, so a
test asking for it fails loudly rather than comparing against nothing.

`ros_file(path)` reads a `robots_specs/<id>/ros*.yml` and `robots_yml(path)` reads
`robots_specs/robots.yml`; both are small parsers for the regular subset of YAML those
files use, not general YAML.
"""

from __future__ import annotations

import __future__
import ast
import builtins
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONSOLE = ROOT / "robot_console"
CONSOLE_SRC = CONSOLE / "src"
SIMULATOR = ROOT / "simulator"
SIM_SHARED = SIMULATOR / "shared"
SPECS = ROOT / "robots_specs"

#: The stdlib modules a parsed binding may use. Everything here is pure.
STDLIB = frozenset({
    "math", "typing", "dataclasses", "collections", "collections.abc", "enum", "re",
    "itertools", "functools", "types", "__future__", "string", "operator",
})

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "tuple", "list", "dict", "set", "frozenset", "float", "int", "str", "bool", "bytes",
        "len", "sorted", "min", "max", "abs", "round", "zip", "enumerate", "range", "sum",
        "any", "all", "isinstance", "issubclass", "reversed", "map", "filter", "repr",
        "getattr", "hasattr", "iter", "next", "divmod", "pow", "hash", "object", "type",
        "property", "staticmethod", "classmethod", "super", "Exception", "ValueError",
        "KeyError", "TypeError", "IndexError", "RuntimeError", "NotImplementedError",
        "AttributeError", "ImportError", "__build_class__", "None", "True", "False",
    )
    if hasattr(builtins, name)
}

_FUTURE_ANNOTATIONS = __future__.annotations.compiler_flag


class Unavailable(LookupError):
    """A binding that cannot be evaluated without importing its module."""


class _Missing:
    """An import this reader will not follow (a third-party or unresolvable module)."""

    def __init__(self, what: str) -> None:
        self.what = what


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class Module:
    """One project module, parsed. Index it by a top-level name to evaluate that binding."""

    _cache: dict[Path, "Module"] = {}

    def __new__(cls, path: Path | str, root: Path | None = None):
        path = Path(path).resolve()
        if path not in cls._cache:
            self = super().__new__(cls)
            self._init(path, root)
            cls._cache[path] = self
        return cls._cache[path]

    def _init(self, path: Path, root: Path | None) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self.root = root or _root_of(path)
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.bindings: dict[str, tuple[str, Any]] = {}
        for node in self.tree.body:
            self._collect(node)
        self._values: dict[str, Any] = {}
        self._evaluating: set[str] = set()

    # ------------------------------------------------------------------ collecting

    def _collect(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                self._bind_target(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self._bind_target(node.target, node.value)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            self.bindings[node.name] = ("def", node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    self.bindings[alias.asname] = ("import", (alias.name, 0, None))
                else:
                    top = alias.name.split(".")[0]
                    self.bindings[top] = ("import", (top, 0, None))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                self.bindings[alias.asname or alias.name] = (
                    "import", (node.module or "", node.level, alias.name))

    def _bind_target(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bindings[target.id] = ("expr", value)
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts):
            for t, v in zip(target.elts, value.elts):
                self._bind_target(t, v)

    # ------------------------------------------------------------------ evaluating

    def __contains__(self, name: str) -> bool:
        return name in self.bindings

    def __getitem__(self, name: str) -> Any:
        if name in self._values:
            value = self._values[name]
            if isinstance(value, Unavailable):
                raise value
            return value
        if name not in self.bindings:
            raise Unavailable(f"{self.path.name} binds no top-level {name!r}")
        if name in self._evaluating:
            raise Unavailable(f"{self.path.name}: {name!r} depends on itself")
        self._evaluating.add(name)
        try:
            value = self._evaluate(name)
        except Unavailable as exc:
            self._values[name] = exc
            raise
        except Exception as exc:  # the binding needs something this reader cannot give
            err = Unavailable(f"{self.path.name}: {name!r} cannot be evaluated as data: "
                              f"{type(exc).__name__}: {exc}")
            self._values[name] = err
            raise err from exc
        finally:
            self._evaluating.discard(name)
        if isinstance(value, _Missing):
            err = Unavailable(f"{self.path.name}: {name!r} is {value.what}, not followed")
            self._values[name] = err
            raise err
        self._values[name] = value
        return value

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except Unavailable:
            return default

    def __getattr__(self, name: str) -> Any:
        # For `module.NAME` inside another module's expression (`from . import topics`).
        if name.startswith("_") and name not in self.__dict__.get("bindings", {}):
            raise AttributeError(name)
        try:
            return self[name]
        except Unavailable as exc:
            raise AttributeError(str(exc)) from exc

    def _namespace(self, node: ast.AST) -> dict[str, Any]:
        """Globals for compiling `node` alone: safe builtins and the names it mentions."""
        ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS,
                              # A class's __module__. `builtins` is always in sys.modules,
                              # which `dataclasses` looks the module up in.
                              "__name__": "builtins"}
        for name in _names_in(node):
            if name in self.bindings and name not in self._evaluating:
                try:
                    ns[name] = self[name]
                except Unavailable:
                    pass  # only an error if the code actually reaches it
        return ns

    def _evaluate(self, name: str) -> Any:
        kind, node = self.bindings[name]
        if kind == "import":
            return self._import(*node)
        if kind == "expr":
            code = compile(ast.Expression(node), str(self.path), "eval",
                           flags=_FUTURE_ANNOTATIONS, dont_inherit=True)
            return eval(code, self._namespace(node))  # noqa: S307 - parsed constants only
        # A def or a class, compiled on its own.
        module = ast.Module(body=[node], type_ignores=[])
        code = compile(module, str(self.path), "exec", flags=_FUTURE_ANNOTATIONS,
                       dont_inherit=True)
        ns = self._namespace(node)
        exec(code, ns)  # noqa: S102 - one definition, no module-level side effects
        return ns[node.name]

    def _import(self, module: str, level: int, attr: str | None) -> Any:
        if level == 0 and (module in STDLIB or module.split(".")[0] in STDLIB):
            imported = __import__(module, fromlist=["_"] if "." in module or attr else [])
            return getattr(imported, attr) if attr else imported
        target = self._resolve(module, level)
        if target is None:
            return _Missing(f"an import of {'.' * level}{module}")
        if attr is None:
            return target if isinstance(target, Module) else _Missing(module)
        if isinstance(target, Path):  # a package: `from pkg import submodule`
            sub = target / f"{attr}.py"
            if sub.exists():
                return Module(sub, self.root)
            init = target / "__init__.py"
            return Module(init, self.root)[attr] if init.exists() else _Missing(module)
        if attr in target:
            return target[attr]
        sub = target.path.with_name(f"{attr}.py")
        if target.path.name == "__init__.py" and sub.exists():
            return Module(sub, self.root)
        return target[attr]

    def _resolve(self, module: str, level: int) -> "Module | Path | None":
        if level:
            base = self.path.parent
            for _ in range(level - 1):
                base = base.parent
        else:
            base = self.root
        parts = [p for p in module.split(".") if p]
        where = base.joinpath(*parts) if parts else base
        if where.with_suffix(".py").exists() and parts:
            return Module(where.with_suffix(".py"), self.root)
        if where.is_dir():
            if (where / "__init__.py").exists() and not level:
                return Module(where / "__init__.py", self.root)
            return where
        return None


def _root_of(path: Path) -> Path:
    """The directory a module's absolute imports are resolved against."""
    if CONSOLE_SRC in path.parents:
        return CONSOLE_SRC
    if SIM_SHARED in path.parents:
        return SIM_SHARED
    return path.parent


def console(dotted: str) -> Module:
    """A console module by dotted name, e.g. `console("robot_console.arm.ros_settings")`."""
    return Module(CONSOLE_SRC.joinpath(*dotted.split(".")).with_suffix(".py"))


def simulator(relative: str) -> Module:
    """A simulator module by path under `simulator/shared/`, e.g. `"ros_surfaces/myagv.py"`."""
    return Module(SIM_SHARED / relative)


# ---------------------------------------------------------------------- YAML subsets


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if text[0] in "\"'" and text[-1] == text[0]:
        return text[1:-1]
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "~"):
        return None
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(p) for p in text[1:-1].split(",") if p.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _strip_comment(line: str) -> str:
    """Drop a trailing ` # comment` that is not inside quotes."""
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def ros_file(path: Path | str) -> dict[str, Any]:
    """A ROS interface file: its top-level scalars, and each list section's rows.

    Returns `{key: scalar}` for top-level `key: value` lines (a `>-` block becomes its
    folded text) and `{section: [row, ...]}` for top-level lists of `- name:` mappings,
    each row a `{field: scalar}` of its four-space fields. Nested mappings under a row
    are ignored; the contract facts are all scalars.
    """
    out: dict[str, Any] = {}
    section: str | None = None
    rows: list[dict[str, Any]] | None = None
    block: tuple[str, list[str]] | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if block is not None:
            if raw.startswith("  ") or not raw.strip():
                block[1].append(raw.strip())
                continue
            out[block[0]] = " ".join(p for p in block[1] if p)
            block = None
        line = _strip_comment(raw)
        if not line.strip():
            continue
        top = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if top:
            key, value = top.groups()
            section, rows = None, None
            if value in (">-", ">", "|", "|-"):
                block = (key, [])
            elif value == "":
                section, rows = key, []
                out[key] = rows
            else:
                out[key] = _scalar(value)
            continue
        if rows is None:
            continue
        item = re.match(r"^  - (\w+):\s*(.*)$", line)
        if item:
            rows.append({item.group(1): _scalar(item.group(2))})
            continue
        field = re.match(r"^    (\w+):\s*(.*)$", line)
        if field and rows:
            rows[-1].setdefault(field.group(1), _scalar(field.group(2)))
    if block is not None:
        out[block[0]] = " ".join(p for p in block[1] if p)
    return out


def robots_yml(path: Path | str = SPECS / "robots.yml") -> dict[str, dict[str, Any]]:
    """`{id: {field: scalar | [scalar]}}` for each entry of `robots:`; nested maps skipped."""
    entries: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    listing: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw)
        start = re.match(r"^  - id:\s*(\S+)", line)
        if start:
            current = entries.setdefault(start.group(1), {"id": start.group(1)})
            listing = None
            continue
        if current is None:
            continue
        field = re.match(r"^    (\w+):\s*(.*)$", line)
        if field:
            key, value = field.groups()
            if value == "":
                listing = key
                current[key] = []
            else:
                listing = None
                current[key] = _scalar(value)
            continue
        item = re.match(r"^      - (.+)$", line)
        if item and listing is not None and isinstance(current.get(listing), list):
            current[listing].append(_scalar(item.group(1)))
    return entries


def ros_files() -> dict[str, dict[str, Any]]:
    """Every simulated robot's ROS file, parsed, by robot id."""
    return {rid: ros_file(SPECS / entry["ros"]) for rid, entry in robots_yml().items()
            if entry.get("simulated") is True}
