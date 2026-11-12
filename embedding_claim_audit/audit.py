"""Offline documentation and Python syntax inspection."""
import ast
import os
import re
from pathlib import Path
from . import phrases as p


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def paragraphs(text):
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append('')
        elif match:
            fence = match[1]
            lines.append('')
        else:
            lines.append(line)
    return [normalize(x) for x in re.split(r'\n\s*\n', '\n'.join(lines)) if x.strip()]


def pattern(phrases):
    return re.compile(r'\b(?:' + '|'.join(re.escape(normalize(x)) for x in sorted(phrases, key=lambda x: (-len(x), x))) + r')\b')


POSITIVE = pattern(p.EMBEDDING)
NEGATIVE = pattern(p.NOEMBEDDING)


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    shadow = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    aliases = {}
    gates = set()
    bindings = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
                gates.add(a.name.split('.')[0].split('_')[0] if a.name.startswith('langchain') else a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            gates.add('langchain' if module.startswith('langchain') else module.split('.')[0])
            for a in node.names:
                aliases[a.asname or a.name] = module + '.' + a.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    bindings[item.optional_vars.id] = item.context_expr

    def name(expr):
        if isinstance(expr, ast.Name):
            return '' if expr.id in shadow else aliases.get(expr.id, expr.id)
        if isinstance(expr, ast.Attribute):
            if expr.attr in shadow:
                return ''
            base = name(expr.value)
            return base + '.' + expr.attr if base else ''
        return ''

    def construction(expr, names, seen=None):
        seen = set() if seen is None else seen
        if isinstance(expr, ast.Name) and expr.id in bindings and expr.id not in seen:
            return construction(bindings[expr.id], names, seen | {expr.id})
        return isinstance(expr, ast.Call) and name(expr.func).split('.')[-1] in names

    def pipeline(expr, seen=None):
        seen = set() if seen is None else seen
        if isinstance(expr, ast.Name) and expr.id in bindings and expr.id not in seen:
            return pipeline(bindings[expr.id], seen | {expr.id})
        if not construction(expr, p.PIPELINE_CONSTRUCTIONS):
            return False
        tasks = ([expr.args[0]] if expr.args else []) + [k.value for k in expr.keywords if k.arg == 'task']
        return any(isinstance(x, ast.Constant) and x.value == p.PIPELINE_TASK[0] for x in tasks)

    routes = [('sentence_transformers', p.SENTENCE_CONSTRUCTIONS, p.SENTENCE_OPERATIONS),
              ('cohere', p.COHERE_CONSTRUCTIONS, p.COHERE_OPERATIONS),
              ('langchain', p.LANGCHAIN_CONSTRUCTIONS, p.LANGCHAIN_OPERATIONS),
              ('llama_index', p.LLAMA_CONSTRUCTIONS, p.LLAMA_OPERATIONS)]
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if 'transformers' in gates and pipeline(func):
            return True
        if isinstance(func, ast.Attribute):
            if func.attr in shadow:
                continue
            for gate, constructors, operations in routes:
                if gate in gates and func.attr in operations and construction(func.value, constructors):
                    return True
            if 'openai' in gates and func.attr == 'create':
                base = func.value
                if isinstance(base, ast.Attribute) and base.attr == 'embeddings' and construction(base.value, p.OPENAI_CONSTRUCTIONS):
                    return True
                if name(base) == 'openai.Embedding':
                    return True
            # Recognised paired API methods require their construction half.
            reserved = (func.attr == 'embed' and 'cohere' in gates) or (func.attr == 'embed_documents' and 'langchain' in gates)
            if func.attr in p.GENERIC_NAMES and not reserved:
                return True
        elif isinstance(func, ast.Name) and func.id not in shadow and func.id in aliases:
            if aliases[func.id].split('.')[-1] in p.GENERIC_NAMES:
                return True
    return False


def audit(repo='.', ignore=()):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('not a directory: ' + str(root))
    positive, negative, production, tests = [], [], [], []
    for directory, dirs, files in os.walk(root, onerror=lambda error: (_ for _ in ()).throw(error), followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for filename in sorted(files):
            path = Path(directory) / filename
            rel = path.relative_to(root)
            if path.is_symlink() or any(rel.as_posix().startswith(prefix) for prefix in ignore):
                continue
            if rel.as_posix() == 'README.md' or (path.suffix == '.md' and 'docs' in rel.parts[:-1]):
                for para in paragraphs(path.read_text(encoding='utf-8')):
                    if NEGATIVE.search(para):
                        negative.append(rel.as_posix())
                    if POSITIVE.search(NEGATIVE.sub(' ', para)):
                        positive.append(rel.as_posix())
            if path.suffix == '.py' and not set(rel.parts[:-1]) & {'fixtures', 'docs', 'examples'}:
                if evidence(path.read_text(encoding='utf-8')):
                    is_test = 'tests' in rel.parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
                    (tests if is_test else production).append(rel.as_posix())
    findings = []
    if positive and not production:
        findings.append(dict(id='documented-embedding-claim-false', severity='MEDIUM' if tests else 'HIGH', message='Documented capability has only test evidence.' if tests else 'Documented capability has no production evidence.', paths=sorted(set(positive))))
    if negative and production:
        findings.append(dict(id='documented-no-embedding-claim-false', severity='HIGH', message='Documented denial contradicts production evidence.', paths=sorted(set(negative))))
    return dict(verdict='FAIL' if findings else 'PASS', findings=findings, evidence=production, test_evidence=tests)
