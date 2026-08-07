#!/usr/bin/env python
"""Flatten ainex.urdf.xacro to plain URDF without needing a ROS/xacro toolchain.

The AiNex xacro is nearly a flat URDF already: 7 scalar <xacro:property> definitions,
one arithmetic substitution (${M_PI/2}), and two <xacro:include>s -- materials.xacro
(colour definitions, no macros) and transmissions.xacro (one macro instantiated 24
times). There are no conditionals, no loops and no parameterised link geometry, so the
expansion is deterministic and this reproduces what `xacro` would emit.

Usage: python flatten_xacro.py <ainex_description_dir> <out.urdf>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROPERTY_RE = re.compile(r'<xacro:property\s+name="([^"]+)"\s+value="([^"]*)"\s*/>')
INCLUDE_RE = re.compile(r'<xacro:include\s+filename="[^"]*/([^"/]+)"\s*/>')
MACRO_RE = re.compile(
    r'<xacro:macro\s+name="([^"]+)"\s+params="([^"]*)"\s*>(.*?)</xacro:macro>', re.S
)
CALL_RE = re.compile(r'<xacro:(\w+)\s+name="([^"]+)"\s*/>')
SUBST_RE = re.compile(r"\$\{([^}]+)\}")


def substitute(text: str, props: dict[str, str]) -> str:
    """Resolve ${...}. Only names and name/2 appear, so eval on a whitelist is enough."""

    def repl(match: re.Match) -> str:
        expr = match.group(1).strip()
        if expr in props:
            return props[expr]
        # The one arithmetic form in the file is ${M_PI/2}.
        numeric = {k: float(v) for k, v in props.items() if _is_number(v)}
        return repr(eval(expr, {"__builtins__": {}}, numeric))  # noqa: S307

    return SUBST_RE.sub(repl, text)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def body_of(xml: str) -> str:
    """Strip the <?xml?> declaration and the outer <robot> element."""
    xml = re.sub(r"<\?xml[^>]*\?>", "", xml)
    xml = re.sub(r"^\s*<robot[^>]*>", "", xml.strip())
    return re.sub(r"</robot>\s*$", "", xml.strip())


def main() -> int:
    src_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = (src_dir / "urdf" / "ainex.urdf.xacro").read_text()

    props = dict(PROPERTY_RE.findall(text))
    text = PROPERTY_RE.sub("", text)

    for name in INCLUDE_RE.findall(text):
        included = (src_dir / "urdf" / name).read_text()
        macros = {m[0]: (m[1].split(), m[2]) for m in MACRO_RE.findall(included)}
        included = MACRO_RE.sub("", included)

        def expand(match: re.Match) -> str:
            macro, arg = match.group(1), match.group(2)
            if macro not in macros:
                return match.group(0)
            params, body = macros[macro]
            return substitute(body, {params[0]: arg})

        included = CALL_RE.sub(expand, included)
        text = text.replace(
            next(m for m in re.findall(r"<xacro:include[^>]*/>", text) if name in m),
            body_of(included),
        )

    text = substitute(text, props)
    # xacro drops the namespace declaration once nothing in the document uses it.
    text = text.replace(' xmlns:xacro="http://ros.org/wiki/xacro"', "")
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    out.write_text(text)

    leftover = SUBST_RE.findall(text) + re.findall(r"<xacro:", text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    print(f"unresolved xacro constructs: {leftover or 'none'}")
    return 1 if leftover else 0


if __name__ == "__main__":
    raise SystemExit(main())
