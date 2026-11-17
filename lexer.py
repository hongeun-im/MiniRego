from lark import Lark
with open("policy.rego") as f:
    source = f.read()

with open("grammar2.lark") as f:
    grammar = f.read()

parser = Lark(grammar, parser="lalr", lexer="basic")  # or omit lexer for contextual

for tok in parser.lex(source):
    if tok.type == "_NL":
        continue
    print(f"{tok.type}({tok.value!r})({tok.line}:{tok.column})")