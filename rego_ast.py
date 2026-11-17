"""Abstract syntax for MiniRego (node names follow the OPA AST).

Every node may carry `loc = (line, col)`, set by the ASTBuilder. It is a plain
attribute (not a dataclass field), so it does not take part in `==`.

`str(node)` gives a compact OPA-like rendering (e.g. `data.admins[role]`),
which is what `dump_ast` in parser.py prints.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


class Node:
    loc = None  # (line, col)


# ---------------------------------------------------------------- terms
@dataclass
class Var(Node):
    name: str

    def __str__(self):
        return self.name


@dataclass
class String(Node):
    value: str

    def __str__(self):
        return json.dumps(self.value)


@dataclass
class Number(Node):
    value: object  # int or float

    def __str__(self):
        return str(self.value)


@dataclass
class Boolean(Node):
    value: bool

    def __str__(self):
        return "true" if self.value else "false"


@dataclass
class Null(Node):
    def __str__(self):
        return "null"


@dataclass
class Array(Node):
    items: List[Node]

    def __str__(self):
        return "[" + ", ".join(map(str, self.items)) + "]"


@dataclass
class Ref(Node):
    """terms[0] is the head (Var or Call). `.name` becomes String("name"),
    `[t]` stays t -- the same shape OPA produces."""
    terms: List[Node]

    def __str__(self):
        out = str(self.terms[0])
        for t in self.terms[1:]:
            if isinstance(t, String) and _IDENT.match(t.value):
                out += "." + t.value
            else:
                out += "[" + str(t) + "]"
        return out


@dataclass
class Call(Node):
    name: str
    args: List[Node]

    def __str__(self):
        return f"{self.name}(" + ", ".join(map(str, self.args)) + ")"


# ---------------------------------------------------------- expressions
@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node

    def __str__(self):
        def w(n):
            return f"({n})" if isinstance(n, BinOp) else str(n)
        return f"{w(self.left)} {self.op} {w(self.right)}"


@dataclass
class UnaryMinus(Node):
    operand: Node

    def __str__(self):
        return f"-{self.operand}"


@dataclass
class SomeDecl(Node):
    vars: List[Var]
    domain: Optional[Node] = None

    def __str__(self):
        s = "some " + ", ".join(v.name for v in self.vars)
        return s + (f" in {self.domain}" if self.domain is not None else "")


@dataclass
class Expr(Node):
    """One line of a rule body."""
    negated: bool
    value: Node  # BinOp | Call | term | SomeDecl

    def __str__(self):
        return ("not " if self.negated else "") + str(self.value)


# ---------------------------------------------------------------- rules
@dataclass
class Else(Node):
    value: Optional[Node]
    body: List[Expr]


@dataclass
class Rule(Node):
    name: str
    default: bool = False
    args: Optional[List[Node]] = None      # f(x, y) -> [x, y]
    key_kind: Optional[str] = None         # "contains" | "index" | None
    key: Optional[Node] = None             # contains <key>  /  name[<key>]
    value: Optional[Node] = None           # := <value>
    body: Optional[List[Expr]] = None
    elses: List[Else] = field(default_factory=list)

    @property
    def kind(self):
        if self.args is not None:
            return "function"
        if self.key_kind == "contains":
            return "partial_set"
        if self.key_kind == "index":
            return "partial_object"
        return "complete"


@dataclass
class Import(Node):
    path: Ref
    alias: Optional[str] = None


@dataclass
class Module(Node):
    package: Ref            # always rooted at `data`: package a.b -> data.a.b
    imports: List[Import]
    rules: List[Rule]
