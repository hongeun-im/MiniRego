"""Parser for MiniRego: Lark (LALR) grammar + parse-tree -> AST builder.

Pipeline:  source text --Lark--> parse tree --ASTBuilder--> rego_ast.Module

Conventions (same as the S-language skeleton):
* `rule: a b c`  production, `|` separates alternatives.
* `?rule`        inlined when it produces exactly one child (precedence chains).
* `alt -> name`  the ASTBuilder method `name` is called for that alternative.
* `_TERMINAL`    terminals starting with `_` are filtered out of the tree, so
                 the builder only receives the children that matter.
                 IDENT, NUMBER and STRING are the only terminals that are kept.
"""

import json
import sys
from dataclasses import dataclass
from typing import List, Optional

from lark import Lark, Transformer, v_args
from lark.exceptions import (UnexpectedCharacters, UnexpectedEOF,
                             UnexpectedInput, UnexpectedToken)

import rego_ast as ast

# ---------------------------------------------------------------------------
# 1. Grammar
# ---------------------------------------------------------------------------

with open("grammar2.lark") as f:
    GRAMMAR = f.read()

PARSER = Lark(GRAMMAR, parser="lalr", propagate_positions=True,
              maybe_placeholders=False)


# ---------------------------------------------------------------------------
# 2. Errors (same code/format as OPA: rego_parse_error (line L, col C): msg)
# ---------------------------------------------------------------------------

class RegoParseError(Exception):
    code = "rego_parse_error"

    def __init__(self, msg, line, col):
        super().__init__(f"{self.code} (line {line}, col {col}): {msg}")
        self.msg, self.line, self.col = msg, line, col


def _tok_name(t):
    return "EOF" if t == "$END" else t.lstrip("_")


def _convert(e: UnexpectedInput) -> RegoParseError:
    if isinstance(e, UnexpectedCharacters):
        if e.char == '"':
            return RegoParseError("non-terminated string", e.line, e.column)
        return RegoParseError(f"illegal character {e.char!r}", e.line, e.column)
    if isinstance(e, UnexpectedToken):
        t = e.token
        what = "EOF" if t.type == "$END" else f"{_tok_name(t.type)} {str(t)!r}"
        exp = sorted({_tok_name(x) for x in e.expected})
        hint = f" (expected one of: {', '.join(exp[:6])}{', ...' if len(exp) > 6 else ''})" if exp else ""
        return RegoParseError(f"unexpected {what}{hint}",
                              t.line or 0, t.column or 0)
    return RegoParseError("unexpected end of input", 0, 0)


# ---------------------------------------------------------------------------
# 3. Semantic actions: parse tree -> AST (see rego_ast.py)
# ---------------------------------------------------------------------------

# Small helper objects used only while building the AST.
class _Default: pass

@dataclass
class _HeadArgs: args: list

@dataclass
class _HeadKey: kind: str; term: object

@dataclass
class _HeadValue: term: object

@dataclass
class _SomeIn: domain: object

@dataclass
class _Head:
    name: str
    args: Optional[list] = None
    key_kind: Optional[str] = None
    key: object = None
    value: object = None


def _at(node, meta):
    if not meta.empty:
        node.loc = (meta.line, meta.column)
    return node


def _binop(op):
    def action(self, meta, children):
        return _at(ast.BinOp(op, children[0], children[1]), meta)
    return action


def _package_ref(node):
    """`package a.b` -> data.a.b  (every path element is a String)."""
    terms = node.terms if isinstance(node, ast.Ref) else [node]
    parts = [ast.String(t.name) if isinstance(t, ast.Var) else t for t in terms]
    return ast.Ref([ast.Var("data")] + parts)


@v_args(meta=True)
class ASTBuilder(Transformer):

    def start(self, meta, children):
        return children[0]

    # ---- module / imports
    def module(self, meta, children):
        pkg, *rest = children
        return _at(ast.Module(
            package=_at(_package_ref(pkg), meta),
            imports=[c for c in rest if isinstance(c, ast.Import)],
            rules=[c for c in rest if isinstance(c, ast.Rule)]), meta)

    def import_stmt(self, meta, children):
        path = children[0]
        if not isinstance(path, ast.Ref):
            path = ast.Ref([path])
        alias = children[1].name if len(children) > 1 else None
        return _at(ast.Import(path, alias), meta)

    # ---- rules
    def default_kw(self, meta, children):
        return _Default()

    def rule(self, meta, children):
        default, head, body, elses = False, None, None, []
        for c in children:
            if isinstance(c, _Default):
                default = True
            elif isinstance(c, _Head):
                head = c
            elif isinstance(c, list):
                body = c
            elif isinstance(c, ast.Else):
                elses.append(c)
        return _at(ast.Rule(name=head.name, default=default, args=head.args,
                            key_kind=head.key_kind, key=head.key,
                            value=head.value, body=body, elses=elses), meta)

    def rule_head(self, meta, children):
        var, *rest = children
        head = _Head(var.name)
        for c in rest:
            if isinstance(c, _HeadArgs):
                head.args = c.args
            elif isinstance(c, _HeadKey):
                head.key_kind, head.key = c.kind, c.term
            elif isinstance(c, _HeadValue):
                head.value = c.term
        return head

    def head_args(self, meta, children):
        return _HeadArgs(list(children))

    def head_contains(self, meta, children):
        return _HeadKey("contains", children[0])

    def head_index(self, meta, children):
        return _HeadKey("index", children[0])

    def head_value(self, meta, children):
        return _HeadValue(children[0])

    def else_clause(self, meta, children):
        value = children[0].term if isinstance(children[0], _HeadValue) else None
        return _at(ast.Else(value, children[-1]), meta)

    def body_single(self, meta, children):
        return [children[0]]

    def body_block(self, meta, children):
        return children[0]

    def query_block(self, meta, children):
        return children[0]

    def query(self, meta, children):
        return list(children)

    # ---- body lines
    def lit_some(self, meta, children):
        return _at(ast.Expr(False, children[0]), meta)

    def lit_pos(self, meta, children):
        return _at(ast.Expr(False, children[0]), meta)

    def lit_neg(self, meta, children):
        return _at(ast.Expr(True, children[0]), meta)

    def some_decl(self, meta, children):
        vars_ = [c for c in children if isinstance(c, ast.Var)]
        domain = next((c.domain for c in children if isinstance(c, _SomeIn)), None)
        return _at(ast.SomeDecl(vars_, domain), meta)

    def some_in(self, meta, children):
        return _SomeIn(children[0])

    # ---- expressions
    assign = _binop(":=")
    unify = _binop("=")
    bit_or = _binop("|")
    bit_and = _binop("&")
    eq = _binop("==")
    neq = _binop("!=")
    lt = _binop("<")
    lte = _binop("<=")
    gt = _binop(">")
    gte = _binop(">=")
    add = _binop("+")
    sub = _binop("-")
    mul = _binop("*")
    div = _binop("/")
    mod = _binop("%")

    def neg(self, meta, children):
        x = children[0]
        if isinstance(x, ast.Number):          # fold `-1` into Number(-1)
            return _at(ast.Number(-x.value), meta)
        return _at(ast.UnaryMinus(x), meta)

    # ---- terms
    def ref(self, meta, children):
        if len(children) == 1:                 # bare variable / call
            return children[0]
        return _at(ast.Ref(list(children)), meta)

    def arg_dot(self, meta, children):
        return _at(ast.String(children[0].name), meta)

    def arg_index(self, meta, children):
        return children[0]

    def call(self, meta, children):
        name, *args = children
        return _at(ast.Call(name.name, args), meta)

    def array(self, meta, children):
        return _at(ast.Array(list(children)), meta)

    def var(self, meta, children):
        tok = children[0]
        node = ast.Var(str(tok))
        node.loc = (tok.line, tok.column)
        return node

    def number(self, meta, children):
        t = str(children[0])
        v = float(t) if any(c in t for c in ".eE") else int(t)
        return _at(ast.Number(v), meta)

    def string(self, meta, children):
        raw = str(children[0])
        try:
            v = json.loads(raw)
        except ValueError:
            v = raw[1:-1]
        return _at(ast.String(v), meta)

    def true_(self, meta, children):
        return _at(ast.Boolean(True), meta)

    def false_(self, meta, children):
        return _at(ast.Boolean(False), meta)

    def null(self, meta, children):
        return _at(ast.Null(), meta)


# ---------------------------------------------------------------------------
# 4. Public API + dumps
# ---------------------------------------------------------------------------

def parse(source: str) -> ast.Module:
    try:
        tree = PARSER.parse(source)
    except UnexpectedInput as e:
        raise _convert(e) from None
    return ASTBuilder().transform(tree)


_SHOW_VALUE = {"IDENT", "NUMBER", "STRING", "ASSIGN", "UNIFY", "EQ", "NEQ",
               "LT", "LTE", "GT", "GTE", "PLUS", "MINUS", "STAR", "SLASH",
               "PERCENT", "PIPE", "AMP"}


def dump_tokens(source: str) -> str:
    """Token list, one source line per output line (newlines are skipped)."""
    lines, cur, cur_line = [], [], None
    try:
        for tok in PARSER.lex(source):
            if tok.type in ("_NL", "COMMENT"):
                continue
            name = tok.type.lstrip("_")
            text = f"{name}({tok.value})" if name in _SHOW_VALUE else name
            if cur_line is not None and tok.line != cur_line:
                lines.append(" ".join(cur)); cur = []
            cur_line = tok.line
            cur.append(f"{text}({tok.line}:{tok.column})")
    except UnexpectedInput as e:
        raise _convert(e) from None
    if cur:
        lines.append(" ".join(cur))
    lines.append("EOF")
    return "\n".join(lines)


def dump_ast(m: ast.Module) -> str:
    out = [f"Module(package={m.package})"]
    items = [("import", i) for i in m.imports] + [("rule", r) for r in m.rules]
    for idx, (kind, it) in enumerate(items):
        last = idx == len(items) - 1
        out.append(("└─ " if last else "├─ ") + _head_str(kind, it))
        pad = "   " if last else "│  "
        children = []
        if kind == "rule":
            if it.body is not None:
                children += [str(e) for e in it.body]
            for el in it.elses:
                children.append("else" + (f" := {el.value}" if el.value is not None else "")
                                + " { " + "; ".join(str(e) for e in el.body) + " }")
        for j, c in enumerate(children):
            out.append(pad + ("└─ " if j == len(children) - 1 else "├─ ") + "Expr " + c
                       if not c.startswith("else") else
                       pad + ("└─ " if j == len(children) - 1 else "├─ ") + c)
    return "\n".join(out)


def _head_str(kind, n):
    if kind == "import":
        return f"Import({n.path}" + (f" as {n.alias}" if n.alias else "") + ")"
    parts = ["default"] if n.default else []
    parts.append(f"{n.kind}")
    head = n.name
    if n.args is not None:
        head += "(" + ", ".join(map(str, n.args)) + ")"
    if n.key_kind == "contains":
        head += f" contains {n.key}"
    elif n.key_kind == "index":
        head += f"[{n.key}]"
    if n.value is not None:
        head += f" := {n.value}"
    return f"Rule({', '.join(parts)}: {head})"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python minirego_parser.py policy.rego [--tokens]")
    src = open(sys.argv[1], encoding="utf-8").read()
    try:
        if "--tokens" in sys.argv:
            print(dump_tokens(src))
        else:
            print(dump_ast(parse(src)))
    except RegoParseError as e:
        sys.exit(str(e))
