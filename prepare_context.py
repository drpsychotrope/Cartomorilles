#!/usr/bin/env python3
"""
prepare_context.py — Générateur de contexte IA + Gestionnaire de sessions.

Analyse le projet Cartomorilles, extrait les métadonnées de chaque module,
et génère un AI_CONTEXT.md avec balises Claude-natives (<role>, <rules>,
<forbidden>, <style_reference>, <checkpoint>).

Intègre un gestionnaire de sessions parallèles avec verrouillage de fichiers,
branches git, et détection de conflits pour le travail multi-instances.

Usage (contexte) :
    python prepare_context.py
    python prepare_context.py --focus grid_builder.py
    python prepare_context.py --focus "Implémenter score_twi()"
    python prepare_context.py --compact
    python prepare_context.py --exclude scoring.py visualize.py
    python prepare_context.py --dry-run --verbose

Usage (sessions) :
    python prepare_context.py session create "twi" --files grid_builder.py
    python prepare_context.py session list
    python prepare_context.py session status twi
    python prepare_context.py session context twi
    python prepare_context.py session apply twi --file grid_builder.py
    python prepare_context.py session merge twi
    python prepare_context.py session merge-all
    python prepare_context.py session abort twi
    python prepare_context.py session history
"""

from __future__ import annotations

import ast
import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_OUTPUT = PROJECT_ROOT / "AI_CONTEXT.md"
DECISIONS_FILE = PROJECT_ROOT / "DECISIONS.md"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Sessions parallèles
SESSIONS_DIR = PROJECT_ROOT / ".sessions"
LOCKS_FILE = SESSIONS_DIR / "locks.json"
HISTORY_FILE = SESSIONS_DIR / "history.json"
BACKUP_DIR = SESSIONS_DIR / "backups"

# Fichiers Python du projet (ordre d'affichage souhaité)
MODULE_ORDER = [
    "config.py",
    "data_loader.py",
    "grid_builder.py",
    "scoring.py",
    "visualize.py",
    "landcover_detector.py",
    "species_enricher.py",
    "main.py",
    "prepare_context.py",
]

# Modules internes (pour détecter les imports inter-modules)
INTERNAL_MODULES = {
    "config", "data_loader", "grid_builder", "scoring",
    "visualize", "landcover_detector", "species_enricher",
    "main",
}

# Fichiers auto-générés (jamais verrouillés par les sessions)
AUTO_GENERATED = {
    "AI_CONTEXT.md", "AI_CONTEXT.json",
    "prepare_context.py",
}

# Fichiers qui ne doivent JAMAIS être verrouillés exclusivement
SHARED_READ_FILES = {"config.py", "config.yaml"}

# Patterns de code représentatifs à chercher pour <style_reference>
STYLE_PATTERNS = {
    "np_asarray": re.compile(
        r"np\.asarray\(.+?\)", re.DOTALL
    ),
    "np_full_like": re.compile(
        r"np\.full_like\(.+?\)", re.DOTALL
    ),
    "np_clip": re.compile(
        r"np\.clip\(.+?\)"
    ),
    "np_interp": re.compile(
        r"np\.interp\(.+?\)", re.DOTALL
    ),
    "logger_info": re.compile(
        r'logger\.\w+\(.+?\)', re.DOTALL
    ),
    "type_hint_signature": re.compile(
        r"def \w+\(self.*?\)\s*->\s*\w+.*?:"
    ),
}

# Tags TODO reconnus, par ordre de sévérité
TODO_TAGS = {
    "FIXME": ("🔴", "Haute"),
    "HACK":  ("🟠", "Moyenne"),
    "TODO":  ("🟡", "Normale"),
    "XXX":   ("⚠️", "Attention"),
    "NOTE":  ("ℹ️", "Info"),
}

logger = logging.getLogger("prepare_context")


# ─────────────────────────────────────────────
# Data classes — Contexte
# ─────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Métadonnées d'une fonction."""
    name: str
    args: list[str]
    returns: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    is_private: bool = False
    line_number: int = 0
    line_count: int = 0
    source: str = ""

    @property
    def signature_short(self) -> str:
        args_str = ", ".join(self.args[:5])
        if len(self.args) > 5:
            args_str += ", ..."
        ret = f" → {self.returns}" if self.returns else ""
        return f"{self.name}({args_str}){ret}"


@dataclass
class ClassInfo:
    """Métadonnées d'une classe."""
    name: str
    bases: list[str]
    docstring: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)
    class_attrs: list[str] = field(default_factory=list)
    line_number: int = 0

    @property
    def public_methods(self) -> list[FunctionInfo]:
        return [m for m in self.methods if not m.name.startswith("_")]

    @property
    def private_methods(self) -> list[FunctionInfo]:
        return [
            m for m in self.methods
            if m.name.startswith("_") and not m.name.startswith("__")
        ]


@dataclass
class ConstantInfo:
    """Métadonnées d'une constante module-level."""
    name: str
    value_repr: str
    type_hint: str | None = None
    line_number: int = 0


@dataclass
class TodoItem:
    """Un commentaire TODO/FIXME/HACK."""
    tag: str
    text: str
    file: str
    line_number: int


@dataclass
class StyleSample:
    """Un extrait de code représentatif du style."""
    source: str
    filename: str
    function_name: str
    line_number: int
    score: float = 0.0


@dataclass
class ModuleInfo:
    """Métadonnées complètes d'un module Python."""
    filepath: Path
    filename: str
    docstring: str | None = None
    version: str | None = None
    imports_internal: list[str] = field(default_factory=list)
    imports_external: list[str] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    constants: list[ConstantInfo] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    file_hash: str = ""

    @property
    def code_lines(self) -> int:
        return self.total_lines - self.blank_lines - self.comment_lines

    @property
    def public_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions if not f.is_private]

    @property
    def private_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions if f.is_private]


@dataclass
class DataFileInfo:
    """Métadonnées d'un fichier de données."""
    filepath: Path
    size_mb: float
    extension: str


@dataclass
class DecisionInfo:
    """Une décision d'architecture verrouillée."""
    id: str
    decision: str
    justification: str


@dataclass
class RejectedApproach:
    """Une approche rejetée."""
    proposition: str
    reason: str


@dataclass
class ProjectInfo:
    """Métadonnées complètes du projet."""
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    data_files: list[DataFileInfo] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    todos: list[TodoItem] = field(default_factory=list)
    style_samples: list[StyleSample] = field(default_factory=list)
    decisions: list[DecisionInfo] = field(default_factory=list)
    rejected: list[RejectedApproach] = field(default_factory=list)
    timestamp: str = ""
    project_version: str = "unknown"


# ─────────────────────────────────────────────
# Data classes — Sessions parallèles
# ─────────────────────────────────────────────

class SessionState(str, Enum):
    ACTIVE = "active"
    MERGING = "merging"
    MERGED = "merged"
    ABORTED = "aborted"
    CONFLICT = "conflict"


class LockType(str, Enum):
    EXCLUSIVE = "exclusive"
    READ_ONLY = "read_only"
    SHARED = "shared"


@dataclass
class FileLock:
    """Verrou sur un fichier."""
    filename: str
    session_name: str
    lock_type: LockType
    locked_at: str
    file_hash_at_lock: str


@dataclass
class FileChange:
    """Un changement appliqué à un fichier pendant une session."""
    filename: str
    hash_before: str
    hash_after: str
    lines_added: int
    lines_removed: int
    timestamp: str
    description: str = ""


@dataclass
class Session:
    """Une session de travail parallèle."""
    name: str
    state: SessionState = SessionState.ACTIVE
    created_at: str = ""
    focus_files: list[str] = field(default_factory=list)
    exclude_files: list[str] = field(default_factory=list)
    read_only_files: list[str] = field(default_factory=list)    
    changes: list[FileChange] = field(default_factory=list)
    description: str = ""
    git_branch: str = ""
    context_file: str = ""
    merged_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["changes"] = [asdict(c) for c in self.changes]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        data["state"] = SessionState(data["state"])
        data["changes"] = [
            FileChange(**c) for c in data.get("changes", [])
        ]
        return cls(**data)


@dataclass
class MergeConflict:
    """Un conflit détecté lors de la fusion."""
    filename: str
    session_a: str
    session_b: str
    conflict_type: str
    details: str = ""


@dataclass
class MergeResult:
    """Résultat d'une fusion."""
    success: bool
    session_name: str
    files_merged: list[str] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Parseur AST
# ─────────────────────────────────────────────

class ModuleParser:
    """Parse un fichier Python via AST et extrait les métadonnées."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.source = filepath.read_text(encoding="utf-8", errors="replace")
        self.lines = self.source.splitlines()

    def parse(self) -> ModuleInfo:
        """Parse complet du module."""
        info = ModuleInfo(
            filepath=self.filepath,
            filename=self.filepath.name,
        )

        info.total_lines = len(self.lines)
        info.blank_lines = sum(1 for line in self.lines if not line.strip())
        info.comment_lines = sum(
            1 for line in self.lines
            if line.strip().startswith("#")
        )
        info.file_hash = hashlib.md5(
            self.source.encode("utf-8")
        ).hexdigest()[:8]

        try:
            tree = ast.parse(self.source, filename=str(self.filepath))
        except SyntaxError as e:
            logger.warning(f"SyntaxError dans {self.filepath.name}: {e}")
            info.docstring = f"⚠️ ERREUR SYNTAXE: {e}"
            return info

        info.docstring = ast.get_docstring(tree)
        info.version = self._extract_version(tree)
        info.imports_internal, info.imports_external = (
            self._extract_imports(tree)
        )
        info.functions = self._extract_functions(tree)
        info.classes = self._extract_classes(tree)
        info.constants = self._extract_constants(tree)
        info.todos = self._extract_todos()

        return info

    def _extract_version(self, tree: ast.Module) -> str | None:
        """Cherche VERSION = '...' ou __version__ = '...'."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                        "VERSION", "__version__", "version"
                    ):
                        if isinstance(node.value, ast.Constant):
                            return str(node.value.value)
        return None

    def _extract_imports(
        self, tree: ast.Module
    ) -> tuple[list[str], list[str]]:
        """Sépare imports internes (projet) vs externes."""
        internal: set[str] = set()
        external: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in INTERNAL_MODULES:
                        internal.add(root)
                    else:
                        external.add(root)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in INTERNAL_MODULES:
                        internal.add(root)
                    else:
                        external.add(root)

        return sorted(internal), sorted(external)

    def _extract_functions(self, tree: ast.Module) -> list[FunctionInfo]:
        """Extrait les fonctions top-level avec leur source."""
        functions: list[FunctionInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._parse_function(node)
                functions.append(func)

        return functions

    def _parse_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> FunctionInfo:
        """Parse une seule définition de fonction."""
        args: list[str] = []
        for arg in node.args.args:
            name = arg.arg
            if arg.annotation:
                name += f": {self._annotation_str(arg.annotation)}"
            args.append(name)

        returns = None
        if node.returns:
            returns = self._annotation_str(node.returns)

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._annotation_str(dec))

        end_line = node.end_lineno or node.lineno
        line_count = end_line - node.lineno + 1

        source = ""
        if node.lineno and end_line:
            source_lines = self.lines[node.lineno - 1:end_line]
            source = "\n".join(source_lines)

        return FunctionInfo(
            name=node.name,
            args=args,
            returns=returns,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            is_private=node.name.startswith("_"),
            line_number=node.lineno,
            line_count=line_count,
            source=source,
        )

    def _extract_classes(self, tree: ast.Module) -> list[ClassInfo]:
        """Extrait les classes et leurs méthodes."""
        classes: list[ClassInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = [self._annotation_str(b) for b in node.bases]
                methods = []
                class_attrs = []

                for item in node.body:
                    if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        methods.append(self._parse_function(item))
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_attrs.append(target.id)

                classes.append(ClassInfo(
                    name=node.name,
                    bases=bases,
                    docstring=ast.get_docstring(node),
                    methods=methods,
                    class_attrs=class_attrs,
                    line_number=node.lineno,
                ))

        return classes

    def _extract_constants(self, tree: ast.Module) -> list[ConstantInfo]:
        """Extrait les constantes module-level (UPPER_CASE)."""
        constants: list[ConstantInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        target.id.isupper()
                        or (
                            target.id.startswith("_")
                            and target.id[1:].isupper()
                        )
                    ):
                        value_repr = self._safe_value_repr(node.value)
                        constants.append(ConstantInfo(
                            name=target.id,
                            value_repr=value_repr,
                            line_number=node.lineno,
                        ))
            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and node.value is not None
                ):
                    name = node.target.id
                    if name.isupper() or (
                        name.startswith("_") and name[1:].isupper()
                    ):
                        value_repr = self._safe_value_repr(node.value)
                        type_hint = self._annotation_str(node.annotation)
                        constants.append(ConstantInfo(
                            name=name,
                            value_repr=value_repr,
                            type_hint=type_hint,
                            line_number=node.lineno,
                        ))

        return constants

    def _extract_todos(self) -> list[TodoItem]:
        """Extrait les TODO/FIXME/HACK/XXX/NOTE des commentaires."""
        todos: list[TodoItem] = []
        pattern = re.compile(
            r"#\s*(TODO|FIXME|HACK|XXX|NOTE)\s*:?\s*(.*)",
            re.IGNORECASE,
        )

        for i, line in enumerate(self.lines, start=1):
            match = pattern.search(line)
            if match:
                todos.append(TodoItem(
                    tag=match.group(1).upper(),
                    text=match.group(2).strip(),
                    file=self.filepath.name,
                    line_number=i,
                ))

        return todos

    def _annotation_str(self, node: ast.expr) -> str:
        """Convertit un noeud AST d'annotation en string lisible."""
        try:
            return ast.unparse(node)
        except Exception:
            return "?"

    def _safe_value_repr(self, node: ast.expr, max_len: int = 80) -> str:
        """Représentation sûre d'une valeur constante."""
        try:
            unparsed = ast.unparse(node)
            if len(unparsed) > max_len:
                return unparsed[:max_len] + "..."
            return unparsed
        except Exception:
            return "<complex>"


# ─────────────────────────────────────────────
# Extracteur de style
# ─────────────────────────────────────────────

class StyleExtractor:
    """Identifie les fonctions les plus représentatives du style du projet."""

    STYLE_INDICATORS = [
        (r"np\.asarray\(", 2.0),
        (r"np\.full_like\(", 2.0),
        (r"np\.clip\(", 1.5),
        (r"np\.interp\(", 2.0),
        (r"~np\.isnan\(", 1.5),
        (r"logger\.\w+\(", 1.0),
        (r"def \w+\(self.*\)\s*->\s*", 1.0),
        (r"scores\[", 1.0),
        (r"valid\s*=\s*~", 1.5),
        (r"np\.where\(", 1.0),
        (r"self\.scores\[", 1.5),
        (r"dtype=np\.float64", 1.0),
        (r"fill_value=", 1.0),
    ]

    def __init__(self, modules: dict[str, ModuleInfo]):
        self.modules = modules

    def extract_best_samples(
        self, max_samples: int = 3, max_lines: int = 25
    ) -> list[StyleSample]:
        """Trouve les N fonctions les plus représentatives du style."""
        candidates: list[StyleSample] = []

        priority_files = [
            "grid_builder.py", "scoring.py", "data_loader.py",
        ]

        for filename in priority_files:
            if filename not in self.modules:
                continue
            mod = self.modules[filename]

            all_funcs = list(mod.functions)
            for cls in mod.classes:
                all_funcs.extend(cls.methods)

            for func in all_funcs:
                if not func.source or func.line_count < 5:
                    continue
                if func.line_count > max_lines:
                    continue

                score = self._score_function(func)
                if score > 3.0:
                    candidates.append(StyleSample(
                        source=func.source,
                        filename=filename,
                        function_name=func.name,
                        line_number=func.line_number,
                        score=score,
                    ))

        candidates.sort(key=lambda s: s.score, reverse=True)
        return candidates[:max_samples]

    def _score_function(self, func: FunctionInfo) -> float:
        """Score de représentativité stylistique d'une fonction."""
        score = 0.0
        source = func.source

        for pattern_str, weight in self.STYLE_INDICATORS:
            pattern = re.compile(pattern_str)
            matches = pattern.findall(source)
            score += len(matches) * weight

        if func.name.startswith("score_"):
            score += 3.0
        if func.returns:
            score += 1.0
        if func.line_count < 8:
            score *= 0.7
        elif func.line_count > 20:
            score *= 0.8

        return score


# ─────────────────────────────────────────────
# Parseur DECISIONS.md
# ─────────────────────────────────────────────

class DecisionsParser:
    """Parse DECISIONS.md pour extraire décisions verrouillées
    et approches rejetées."""

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def parse(self) -> tuple[list[DecisionInfo], list[RejectedApproach]]:
        """Parse le fichier et retourne décisions + rejets."""
        if not self.filepath.exists():
            return [], []

        content = self.filepath.read_text(encoding="utf-8")
        decisions = self._parse_decisions(content)
        rejected = self._parse_rejected(content)
        return decisions, rejected

    def _parse_decisions(self, content: str) -> list[DecisionInfo]:
        """Extrait les décisions du tableau principal."""
        decisions: list[DecisionInfo] = []
        pattern = re.compile(
            r"\|\s*(D\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|"
        )
        for match in pattern.finditer(content):
            decisions.append(DecisionInfo(
                id=match.group(1).strip(),
                decision=match.group(2).strip(),
                justification=match.group(3).strip(),
            ))
        return decisions

    def _parse_rejected(self, content: str) -> list[RejectedApproach]:
        """Extrait les approches rejetées."""
        rejected: list[RejectedApproach] = []

        rejected_section = re.search(
            r"(?:rejet|REJET).*?\n((?:\|.*\n)+)",
            content,
            re.IGNORECASE,
        )
        if not rejected_section:
            return rejected

        pattern = re.compile(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|")
        for match in pattern.finditer(rejected_section.group(1)):
            prop = match.group(1).strip()
            reason = match.group(2).strip()
            if prop.startswith("---") or prop.lower() == "proposition":
                continue
            rejected.append(RejectedApproach(
                proposition=prop,
                reason=reason,
            ))
        return rejected


# ─────────────────────────────────────────────
# Analyseur de projet
# ─────────────────────────────────────────────

class ProjectAnalyzer:
    """Analyse l'ensemble du projet Cartomorilles."""

    def __init__(self, root: Path):
        self.root = root

    def analyze(self) -> ProjectInfo:
        """Analyse complète du projet."""
        project = ProjectInfo(
            timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        )

        logger.info("Analyse des modules Python...")
        for filename in MODULE_ORDER:
            filepath = self.root / filename
            if filepath.exists():
                logger.info(f"  Parsing {filename}...")
                parser = ModuleParser(filepath)
                module_info = parser.parse()
                project.modules[filename] = module_info
            else:
                logger.warning(f"  {filename} non trouvé — ignoré")

        for py_file in sorted(self.root.glob("*.py")):
            if py_file.name not in project.modules:
                logger.info(
                    f"  Parsing {py_file.name} (non listé)..."
                )
                parser = ModuleParser(py_file)
                module_info = parser.parse()
                project.modules[py_file.name] = module_info

        if "main.py" in project.modules:
            v = project.modules["main.py"].version
            if v:
                project.project_version = v

        logger.info("Scan des fichiers de données...")
        project.data_files = self._scan_data_files()

        logger.info("Construction du graphe de dépendances...")
        project.dependency_graph = self._build_dependency_graph(
            project.modules
        )

        for mod in project.modules.values():
            project.todos.extend(mod.todos)

        logger.info("Extraction du style de code...")
        extractor = StyleExtractor(project.modules)
        project.style_samples = extractor.extract_best_samples()
        logger.info(
            f"  {len(project.style_samples)} sample(s) extrait(s)"
        )

        logger.info("Chargement des décisions d'architecture...")
        dec_parser = DecisionsParser(DECISIONS_FILE)
        project.decisions, project.rejected = dec_parser.parse()
        logger.info(
            f"  {len(project.decisions)} décision(s), "
            f"{len(project.rejected)} rejet(s)"
        )

        return project

    def _scan_data_files(self) -> list[DataFileInfo]:
        """Scanne les fichiers de données du projet."""
        data_files: list[DataFileInfo] = []
        data_extensions = {
            ".shp", ".shx", ".dbf", ".prj", ".cpg",
            ".tif", ".tiff", ".json", ".csv", ".geojson",
        }

        data_dir = self.root / "data"
        if data_dir.exists():
            for f in sorted(data_dir.rglob("*")):
                if f.is_file() and f.suffix in data_extensions:
                    data_files.append(DataFileInfo(
                        filepath=f.relative_to(self.root),
                        size_mb=f.stat().st_size / (1024 * 1024),
                        extension=f.suffix,
                    ))

        for f in sorted(self.root.glob("*.tif")):
            data_files.append(DataFileInfo(
                filepath=f.relative_to(self.root),
                size_mb=f.stat().st_size / (1024 * 1024),
                extension=f.suffix,
            ))

        return data_files

    def _build_dependency_graph(
        self, modules: dict[str, ModuleInfo]
    ) -> dict[str, list[str]]:
        """Construit le graphe d'imports inter-modules."""
        graph: dict[str, list[str]] = {}
        for filename, mod in modules.items():
            module_name = filename.replace(".py", "")
            deps = [
                dep for dep in mod.imports_internal
                if dep != module_name
            ]
            if deps:
                graph[module_name] = deps
        return graph


# ─────────────────────────────────────────────
# Générateur de contexte — Version Claude-native
# ─────────────────────────────────────────────

class ClaudeContextGenerator:
    """Génère AI_CONTEXT.md avec balises Claude-natives."""

    def __init__(
        self,
        project: ProjectInfo,
        compact: bool = False,
        focus: str | None = None,
        exclude: list[str] | None = None,
        session: Session | None = None,
    ):
        self.project = project
        self.compact = compact
        self.focus = focus
        self.exclude = set(exclude or [])
        self.session = session
        self.lines: list[str] = []

    def generate(self) -> str:
        """Génère le contenu complet."""
        self._header()
        if self.session:
            self._section_session_header()
        self._section_role()
        self._section_rules()
        self._section_forbidden()
        self._section_style_reference()
        self._section_project_identity()
        self._section_architecture()
        self._section_data_flow()
        self._section_modules()
        self._section_constants_dense()
        self._section_dependencies()
        self._section_conventions()
        self._section_decisions()
        self._section_bugs_todos()
        self._section_stats()
        self._section_focus()
        self._section_checkpoint()
        self._footer()

        return "\n".join(self.lines)

    # ── Header / Footer ───────────────────────

    def _header(self) -> None:
        v = self.project.project_version
        ts = self.project.timestamp
        mode = "compact" if self.compact else "full"
        session_tag = ""
        if self.session:
            session_tag = f" | session:{self.session.name}"
        self._add(
            f"# 🍄 CARTOMORILLES — AI Context v{v} | "
            f"{ts} | {mode}{session_tag}"
        )
        self._add("")
        self._add(
            "> Auto-généré par `prepare_context.py` — "
            "optimisé Claude Opus"
        )

        all_hashes = "".join(
            m.file_hash for m in self.project.modules.values()
        )
        global_hash = hashlib.md5(
            all_hashes.encode()
        ).hexdigest()[:10]
        self._add(f"> Project hash: `{global_hash}`")
        self._add("")

    def _footer(self) -> None:
        total_lines = len(self.lines)
        budget_remaining = 1750 - total_lines
        self._add("")
        self._add("---")
        self._add(
            f"_Généré le {self.project.timestamp} | "
            f"{total_lines + 3} lignes | "
            f"~{budget_remaining} lignes restantes pour prompt + code_"
        )

    # ── Section session parallèle ─────────────

    def _section_session_header(self) -> None:
        if not self.session:
            return

        self._add(
            f"> 🔀 SESSION PARALLÈLE : **{self.session.name}**"
        )
        self._add(
            f"> Branche git : `{self.session.git_branch}`"
        )
        self._add(
            f"> Créée le : {self.session.created_at}"
        )
        self._add(">")

        self._add("> **Fichiers modifiables (verrouillés) :**")
        for f in self.session.focus_files:
            if f in SHARED_READ_FILES:
                self._add(
                    f"> - 🔓 `{f}` (partagé — "
                    f"attention aux conflits)"
                )
            else:
                self._add(f"> - 🔒 `{f}` (exclusif)")

        if self.session.read_only_files:
            self._add("> **Lecture seule :**")
            for f in self.session.read_only_files:
                self._add(f"> - 👁️ `{f}`")

        self._add(">")
        self._add(
            "> ⚠️ Ne modifie QUE les fichiers verrouillés. "
            "Pour tout autre fichier → `# TODO:`"
        )
        self._add("")

    # ── Sections Claude-natives ───────────────

    def _section_role(self) -> None:
        self._add("<role>")
        self._add(
            "Tu es un expert Python géospatial et mycologue "
            "computationnel."
        )
        self._add(
            "Tu as écrit ce code lors de sessions précédentes."
        )
        self._add(
            "Maintiens la cohérence avec tes choix antérieurs "
            "décrits ci-dessous."
        )
        self._add("</role>")
        self._add("")

    def _section_rules(self) -> None:
        self._add("<rules>")
        self._add("- Code complet uniquement — jamais de fragments")
        self._add(
            "- 0 explication non sollicitée "
            "(code speaks for itself)"
        )
        self._add(
            "- Sois assertif — pas de hedging ni de caveats "
            "inutiles"
        )
        self._add(
            "- Si doute technique réel → ⚠️ DOUTE: [raison], "
            "pas de noyade dans les caveats"
        )
        self._add(
            '- Logger uniquement '
            '(`logging.getLogger("cartomorilles.<mod>")`) — '
            'jamais print'
        )
        self._add(
            "- Demande confirmation avant de modifier config.py "
            "(poids/seuils/éliminatoires)"
        )
        self._add(
            "- Bug hors scope → `# TODO: [desc]` dans le code, "
            "ne PAS corriger"
        )
        self._add(
            "- Rien de plus que ce qui est demandé — "
            "pas de features spontanées"
        )

        if self.session:
            self._add(
                f"- SESSION '{self.session.name}' : modifier "
                f"UNIQUEMENT {', '.join(self.session.focus_files)}"
            )

        self._add("</rules>")
        self._add("")

    def _section_forbidden(self) -> None:
        self._add("<forbidden>")

        if self.project.decisions:
            for d in self.project.decisions:
                self._add(
                    f"- [{d.id}] Ne PAS contredire : {d.decision}"
                )

        if self.project.rejected:
            self._add("")
            for r in self.project.rejected:
                self._add(
                    f"- Ne PAS re-proposer : {r.proposition} "
                    f"(rejeté : {r.reason})"
                )

        self._add("")
        self._add(
            "- Ne PAS utiliser Optional, Dict, List (types legacy)"
        )
        self._add("- Ne PAS utiliser print() au lieu du logger")
        self._add("- Ne PAS produire de code partiel / fragmenté")
        self._add("- Ne PAS ajouter de features non demandées")

        if self.session:
            forbidden_files = [
                f for f in self.project.modules
                if f not in self.session.focus_files
                and f not in self.session.read_only_files
                and f not in AUTO_GENERATED
            ]
            if forbidden_files:
                self._add(
                    f"- Ne PAS modifier : "
                    f"{', '.join(forbidden_files)}"
                )

        self._add("</forbidden>")
        self._add("")

    def _section_style_reference(self) -> None:
        samples = self.project.style_samples
        if not samples:
            return

        self._add("<style_reference>")
        self._add(
            "Ton style dans ce projet (à maintenir). "
            "Extraits auto-détectés :"
        )
        self._add("")

        for i, sample in enumerate(samples):
            self._add(
                f"# Extrait {i + 1}: {sample.function_name} "
                f"({sample.filename} L{sample.line_number}) "
                f"[score: {sample.score:.1f}]"
            )
            for line in sample.source.splitlines():
                self._add(f"    {line}")
            self._add("")

        self._add("Patterns récurrents détectés :")
        patterns_found = self._detect_global_patterns()
        for pattern_desc in patterns_found:
            self._add(f"- {pattern_desc}")

        self._add("</style_reference>")
        self._add("")

    def _detect_global_patterns(self) -> list[str]:
        """Détecte les patterns globaux dans tout le codebase."""
        patterns: list[str] = []
        all_source = "\n".join(
            m.filepath.read_text(encoding="utf-8", errors="replace")
            for m in self.project.modules.values()
            if m.filepath.exists()
        )

        checks = [
            (
                r"np\.asarray\(",
                "np.asarray() en entrée des fonctions de score",
            ),
            (
                r"np\.full_like\(",
                "np.full_like + masque valid pour NaN-safety",
            ),
            (
                r"np\.clip\(",
                "np.clip en sortie des scores [0, 1]",
            ),
            (
                r"np\.interp\(",
                "np.interp pour transitions linéaires entre seuils",
            ),
            (
                r'logger\.\w+\("',
                "Logger avec messages descriptifs pour chaque étape",
            ),
            (
                r"def \w+\(self.*?\)\s*->\s*\w+",
                "Type hints sur toutes les signatures publiques",
            ),
            (
                r"MappingProxyType|frozenset|tuple\(",
                "Immutabilité sur les constantes (MappingProxyType, "
                "frozenset, tuple)",
            ),
            (
                r"from __future__ import annotations",
                "from __future__ import annotations en tête",
            ),
            (
                r"isinstance\(.+?,\s*np\.ndarray\)",
                "isinstance guard avant accès .shape",
            ),
        ]

        for regex, description in checks:
            count = len(re.findall(regex, all_source))
            if count >= 2:
                patterns.append(f"{description} ({count}× trouvé)")

        return patterns

    def _section_project_identity(self) -> None:
        self._add("## IDENTITÉ DU PROJET")
        self._add("")
        self._add("| Champ | Valeur |")
        self._add("|---|---|")
        self._add("| Nom | Cartomorilles |")
        self._add(
            "| Objectif | Cartographie probabiliste multicritère "
            "des zones favorables aux morilles |"
        )
        self._add("| Emprise | 20×20 km centrée Grenoble |")
        self._add("| CRS | EPSG:2154 (Lambert-93) |")
        self._add(
            "| Centre L93 | (913_100, 6_458_800), rayon 10 km |"
        )
        self._add("| DEM | BD ALTI 25 m, 6000×6000 px |")
        self._add(
            f"| Version | {self.project.project_version} |"
        )
        self._add("")

        all_externals: set[str] = set()
        for mod in self.project.modules.values():
            all_externals.update(mod.imports_external)

        key_libs = sorted(all_externals & {
            "numpy", "pandas", "geopandas", "rasterio", "folium",
            "scipy", "shapely", "matplotlib", "fiona",
            "requests", "PIL", "skimage",
        })
        if key_libs:
            self._add(
                f"**Stack** : Python 3.12+, "
                f"{', '.join(key_libs)}"
            )
            self._add("")

    def _section_architecture(self) -> None:
        self._add("## ARBORESCENCE & ÉTAT")
        self._add("")
        self._add("```")
        self._add(f"{PROJECT_ROOT}\\")

        for filename, mod in self.project.modules.items():
            status = "✅"
            version = f" v{mod.version}" if mod.version else ""

            has_fixme = any(
                t.tag in ("FIXME", "HACK") for t in mod.todos
            )
            if has_fixme:
                status = "⚠️"

            focus_marker = ""
            if self.focus and filename in self.focus:
                focus_marker = " ◄ FOCUS"
            if (
                self.session
                and filename in self.session.focus_files
            ):
                focus_marker = " ◄ SESSION"

            if filename in self.exclude:
                status = "⏭️"
                focus_marker = " (hors scope)"

            doc_short = ""
            if mod.docstring:
                first_line = mod.docstring.strip().split("\n")[0]
                if len(first_line) > 50:
                    first_line = first_line[:47] + "..."
                doc_short = f" — {first_line}"

            self._add(
                f"├── {filename:<26s} {status}{version} "
                f"({mod.total_lines}L){doc_short}{focus_marker}"
            )

        if self.project.data_files:
            self._add("└── data/")
            for df in self.project.data_files:
                size = (
                    f"{df.size_mb:.1f}MB"
                    if df.size_mb > 0.1
                    else "<0.1MB"
                )
                self._add(
                    f"    ├── {df.filepath.name:<34s} ({size})"
                )

        self._add("```")
        self._add("")

    def _section_data_flow(self) -> None:
        self._add("## FLUX DE DONNÉES (PIPELINE)")
        self._add("")
        self._add("```")
        self._add("DEM (.tif) + Shapefiles (data/)")
        self._add("    ↓")
        self._add("data_loader.load_dem/forest/geology/hydro/urban()")
        self._add("    ↓")
        self._add(
            "grid_builder.compute_terrain() "
            "→ altitude, slope, aspect, roughness, TWI"
        )
        self._add(
            "grid_builder.score_*() → 11 scores [0,1]"
        )
        self._add("    ↓                      ↑")
        self._add(
            "species_enricher ──────────┘ (remplace unknown)"
        )
        self._add("    ↓")
        self._add(
            "apply_urban_mask()  ← AVANT micro-habitats"
        )
        self._add("score_canopy/ground/disturbance()")
        self._add("apply_water_mask()")
        self._add(
            "apply_landcover_mask() ← forest floor + green_clip"
        )
        self._add("    ↓")
        self._add("scoring.compute_weighted_score()")
        self._add(
            "scoring.apply_eliminatory_factors() ← 7 masques"
        )
        self._add("scoring.apply_spatial_smoothing()")
        self._add("scoring.classify_probability() → 6 classes")
        self._add("scoring.get_hotspots() → clusters 8-conn")
        self._add("    ↓")
        self._add("visualize → Folium + GeoTIFF + GPKG")
        self._add("```")
        self._add("")

    def _section_modules(self) -> None:
        self._add("## RÉSUMÉ FONCTIONNEL PAR MODULE")
        self._add("")

        for filename, mod in self.project.modules.items():
            if filename in self.exclude:
                self._add(
                    f"### {filename} ⏭️ (hors scope cette session)"
                )
                self._add("")
                continue

            version_str = (
                f" (v{mod.version})" if mod.version else ""
            )
            self._add(f"### {filename}{version_str}")
            self._add("")

            if mod.docstring:
                first_line = mod.docstring.strip().split("\n")[0]
                self._add(f"_{first_line}_")
                self._add("")

            self._add(
                f"📊 `{mod.total_lines}L | "
                f"{mod.code_lines}L code | "
                f"{len(mod.functions)}fn | "
                f"{len(mod.classes)}cls | "
                f"{mod.file_hash}`"
            )
            self._add("")

            if self.compact:
                self._module_compact(mod)
            else:
                self._module_detailed(mod)

            self._add("---")
            self._add("")

    def _module_compact(self, mod: ModuleInfo) -> None:
        """Résumé compact d'un module."""
        for cls in mod.classes:
            methods = ", ".join(m.name for m in cls.public_methods)
            self._add(f"**{cls.name}** : `{methods}`")

        pub = mod.public_functions
        if pub:
            names = ", ".join(f.name for f in pub)
            self._add(f"**Publiques** : `{names}`")

        if mod.imports_internal:
            self._add(
                f"**← imports** : "
                f"{', '.join(mod.imports_internal)}"
            )

        self._add("")

    def _module_detailed(self, mod: ModuleInfo) -> None:
        """Résumé détaillé d'un module."""
        for cls in mod.classes:
            bases_str = (
                f"({', '.join(cls.bases)})" if cls.bases else ""
            )
            self._add(f"**class {cls.name}{bases_str}**")
            if cls.docstring:
                doc_first = cls.docstring.strip().split("\n")[0]
                self._add(f"  _{doc_first}_")
            if cls.public_methods:
                methods_str = ", ".join(
                    m.signature_short for m in cls.public_methods
                )
                self._add(f"  Publiques : `{methods_str}`")
            if cls.private_methods:
                priv_names = ", ".join(
                    m.name for m in cls.private_methods
                )
                self._add(f"  Privées : `{priv_names}`")
            self._add("")

        pub_funcs = mod.public_functions
        if pub_funcs:
            self._add("**Fonctions publiques :**")
            for f in pub_funcs:
                doc = ""
                if f.docstring:
                    doc_line = f.docstring.strip().split("\n")[0]
                    if len(doc_line) > 60:
                        doc_line = doc_line[:57] + "..."
                    doc = f" — {doc_line}"
                self._add(f"- `{f.signature_short}`{doc}")
            self._add("")

        priv_funcs = mod.private_functions
        if priv_funcs:
            names = ", ".join(f.name for f in priv_funcs)
            self._add(f"**Privées** : `{names}`")
            self._add("")

        if mod.imports_internal:
            self._add(
                f"**← imports** : "
                f"{', '.join(mod.imports_internal)}"
            )
            self._add("")

    def _section_constants_dense(self) -> None:
        """Constantes en notation ultra-dense."""
        if "config.py" not in self.project.modules:
            return

        config_mod = self.project.modules["config.py"]
        self._add("## CONSTANTES CRITIQUES (config.py)")
        self._add("")

        groups: dict[str, list[ConstantInfo]] = defaultdict(list)
        category_keywords = {
            "Poids": ["WEIGHT"],
            "Altitude": ["ALTITUDE"],
            "Pente": ["SLOPE"],
            "TWI": ["TWI"],
            "Géologie": ["GEOLOGY", "ELIMINATORY_GEO"],
            "Essences": [
                "TREE", "SPECIES", "ELIMINATORY_SP", "VEGETATION",
            ],
            "Landcover": ["LANDCOVER", "FOREST_FLOOR", "CANOPY"],
            "Hydrographie": ["WATER", "DIST_WATER", "HYDRO"],
            "Urbain": ["URBAN", "BUFFER"],
            "Emprise": [
                "BBOX", "CENTER", "RADIUS", "CELL", "GRID",
            ],
        }

        for c in config_mod.constants:
            name = c.name.lstrip("_")
            categorized = False
            for cat, keywords in category_keywords.items():
                if any(kw in name for kw in keywords):
                    groups[cat].append(c)
                    categorized = True
                    break
            if not categorized:
                groups["Autres"].append(c)

        if self.compact:
            for group_name, consts in sorted(groups.items()):
                items = []
                for c in consts:
                    val = c.value_repr
                    if len(val) > 30:
                        val = val[:27] + "..."
                    items.append(f"{c.name}={val}")
                self._add(
                    f"**{group_name}** : {' | '.join(items)}"
                )
            self._add("")
        else:
            for group_name, consts in sorted(groups.items()):
                if not consts:
                    continue
                self._add(f"### {group_name}")
                self._add("| Constante | Valeur |")
                self._add("|---|---|")
                for c in consts:
                    val = c.value_repr
                    if len(val) > 60:
                        val = val[:57] + "..."
                    self._add(f"| `{c.name}` | `{val}` |")
                self._add("")

    def _section_dependencies(self) -> None:
        self._add("## DÉPENDANCES INTER-MODULES")
        self._add("")

        graph = self.project.dependency_graph
        if not graph:
            self._add("_Aucune dépendance détectée._")
            self._add("")
            return

        self._add("| Module | Importe depuis |")
        self._add("|---|---|")
        for module, deps in sorted(graph.items()):
            deps_str = ", ".join(f"`{d}`" for d in deps)
            self._add(f"| `{module}` | {deps_str} |")
        self._add("")

        import_count: dict[str, int] = defaultdict(int)
        for deps in graph.values():
            for d in deps:
                import_count[d] += 1
        if import_count:
            top = sorted(
                import_count.items(), key=lambda x: -x[1]
            )[:3]
            self._add(
                "**Hub** : "
                + ", ".join(
                    f"`{n}` ({c}×)" for n, c in top
                )
            )
            self._add("")

    def _section_conventions(self) -> None:
        self._add("## CONVENTIONS PYLANCE (P1-P10)")
        self._add("")

        conventions = [
            ("P1", "`from __future__ import annotations` en tête"),
            (
                "P2",
                "Types modernes : `dict` pas `Dict`, "
                "`str | None` pas `Optional`",
            ),
            (
                "P3",
                "Optional → variable locale + assert + type hint",
            ),
            (
                "P4",
                "`getattr()` + `isinstance()` pour attributs "
                "optionnels",
            ),
            (
                "P5",
                "`np.asarray()` autour de rasterize/gaussian/EDT. "
                "`np.any()` au lieu de `.any()`",
            ),
            ("P6", "Import direct si sous-module échoue"),
            (
                "P7",
                "`# type: ignore[…]` pour faux positifs documentés",
            ),
            ("P8", "Scores dict : isinstance guard avant .shape"),
            (
                "P9",
                'Logger `logging.getLogger("cartomorilles.<mod>")` '
                "— jamais print",
            ),
            (
                "P10",
                "Immutabilité : MappingProxyType, frozenset, tuple",
            ),
        ]

        for code, desc in conventions:
            self._add(f"- **{code}** : {desc}")
        self._add("")

    def _section_decisions(self) -> None:
        self._add("## DÉCISIONS VERROUILLÉES")
        self._add("")

        if self.project.decisions:
            self._add("| # | Décision | Justification |")
            self._add("|---|----------|---------------|")
            for d in self.project.decisions:
                self._add(
                    f"| {d.id} | {d.decision} | "
                    f"{d.justification} |"
                )
            self._add("")

        if self.project.rejected:
            self._add("**Approches REJETÉES :**")
            self._add("| Proposition | Raison |")
            self._add("|---|---|")
            for r in self.project.rejected:
                self._add(
                    f"| {r.proposition} | {r.reason} |"
                )
            self._add("")

        if not self.project.decisions and not self.project.rejected:
            self._add(
                "> ⚠️ `DECISIONS.md` non trouvé ou vide. "
                "Créez-le pour auto-remplir cette section."
            )
            self._add(">")
            self._add("> ```")
            self._add(
                "> | # | Décision | Justification |"
            )
            self._add("> |---|----------|---------------|")
            self._add(
                "> | D1 | TWI algo D8 | "
                "Suffisant pour 25m |"
            )
            self._add("> ```")
            self._add("")

    def _section_bugs_todos(self) -> None:
        self._add("## BUGS CONNUS & TODOs")
        self._add("")

        todos = self.project.todos
        if not todos:
            self._add("_Aucun TODO/FIXME trouvé dans le code._")
            self._add("")
            return

        by_tag: dict[str, list[TodoItem]] = defaultdict(list)
        for t in todos:
            by_tag[t.tag].append(t)

        self._add(
            "| Sév. | Tag | Fichier | L | Description |"
        )
        self._add("|---|---|---|---|---|")

        tag_order = ["FIXME", "HACK", "TODO", "XXX", "NOTE"]
        for tag in tag_order:
            if tag not in by_tag:
                continue
            emoji, _ = TODO_TAGS.get(tag, ("", ""))
            for t in by_tag[tag]:
                desc = (
                    t.text[:70] + "..."
                    if len(t.text) > 70
                    else t.text
                )
                self._add(
                    f"| {emoji} | {t.tag} | `{t.file}` | "
                    f"{t.line_number} | {desc} |"
                )

        self._add("")
        total_str = ", ".join(
            f"{len(v)} {k}" for k, v in sorted(by_tag.items())
        )
        self._add(f"**Total** : {len(todos)} ({total_str})")
        self._add("")

    def _section_stats(self) -> None:
        self._add("## STATISTIQUES")
        self._add("")

        total_lines = sum(
            m.total_lines for m in self.project.modules.values()
        )
        total_code = sum(
            m.code_lines for m in self.project.modules.values()
        )
        total_funcs = sum(
            len(m.functions) for m in self.project.modules.values()
        )
        total_classes = sum(
            len(m.classes) for m in self.project.modules.values()
        )
        total_data_mb = sum(
            df.size_mb for df in self.project.data_files
        )

        self._add(
            f"**Code** : {len(self.project.modules)} modules | "
            f"{total_lines:,}L total | "
            f"{total_code:,}L code | "
            f"{total_funcs} fn | "
            f"{total_classes} cls"
        )
        self._add(
            f"**Data** : {len(self.project.data_files)} fichiers | "
            f"{total_data_mb:.1f} MB"
        )
        self._add(
            f"**Santé** : {len(self.project.todos)} TODOs | "
            f"{len(self.project.decisions)} décisions verrouillées"
        )
        self._add("")

    def _section_focus(self) -> None:
        self._add("<current_focus>")

        if self.session:
            self._add(
                f"FOCUS DE CETTE SESSION : {self.session.name} — "
                f"{', '.join(self.session.focus_files)}"
            )
            if self.session.description:
                self._add(
                    f"DESCRIPTION : {self.session.description}"
                )
        elif self.focus:
            self._add(f"FOCUS DE CETTE SESSION : {self.focus}")
        else:
            self._add(
                "FOCUS : [à définir dans ton premier message]"
            )

        if self.session:
            excluded = ", ".join(sorted(
                f for f in self.session.exclude_files
                if f not in AUTO_GENERATED
            ))
            if excluded:
                self._add(
                    f"HORS SCOPE (ne pas toucher) : {excluded}"
                )
        elif self.exclude:
            excluded = ", ".join(sorted(self.exclude))
            self._add(
                f"HORS SCOPE (ne pas toucher) : {excluded}"
            )
        else:
            self._add(
                "HORS SCOPE : [aucun exclusion définie — "
                "préciser si nécessaire]"
            )

        self._add(
            "Si bug détecté hors scope → "
            "`# TODO: [desc]` mais ne PAS corriger."
        )
        self._add("</current_focus>")
        self._add("")

    def _section_checkpoint(self) -> None:
        self._add("<checkpoint>")
        self._add(
            "Avant de coder, remplis exactement ce template "
            "(pas plus) :"
        )
        self._add("")
        self._add(
            f"PROJET: Cartomorilles "
            f"v{self.project.project_version}"
        )
        self._add("TÂCHE: ___")
        self._add("FICHIER: ___")
        self._add("DÉPEND DE: ___")
        self._add("INTERDIT: ___")

        if self.session:
            self._add(f"SESSION: {self.session.name}")
            self._add(
                f"BRANCHE: {self.session.git_branch}"
            )

        self._add("</checkpoint>")

    # ── Helpers ────────────────────────────────

    def _add(self, line: str) -> None:
        self.lines.append(line)


# ─────────────────────────────────────────────
# Export JSON
# ─────────────────────────────────────────────

def _project_to_json(project: ProjectInfo) -> dict[str, Any]:
    """Convertit ProjectInfo en dict JSON-serializable."""
    modules_json: dict[str, Any] = {}
    for filename, mod in project.modules.items():
        modules_json[filename] = {
            "version": mod.version,
            "total_lines": mod.total_lines,
            "code_lines": mod.code_lines,
            "hash": mod.file_hash,
            "functions": [f.name for f in mod.functions],
            "classes": [c.name for c in mod.classes],
            "constants": [c.name for c in mod.constants],
            "imports_internal": mod.imports_internal,
            "imports_external": mod.imports_external,
            "todos": [
                {
                    "tag": t.tag,
                    "text": t.text,
                    "line": t.line_number,
                }
                for t in mod.todos
            ],
        }

    return {
        "timestamp": project.timestamp,
        "version": project.project_version,
        "modules": modules_json,
        "dependency_graph": project.dependency_graph,
        "data_files": [
            {
                "path": str(df.filepath),
                "size_mb": round(df.size_mb, 2),
            }
            for df in project.data_files
        ],
        "decisions": [
            {
                "id": d.id,
                "decision": d.decision,
                "justification": d.justification,
            }
            for d in project.decisions
        ],
        "rejected": [
            {
                "proposition": r.proposition,
                "reason": r.reason,
            }
            for r in project.rejected
        ],
        "style_samples": [
            {
                "function": s.function_name,
                "file": s.filename,
                "score": round(s.score, 1),
            }
            for s in project.style_samples
        ],
        "stats": {
            "total_modules": len(project.modules),
            "total_todos": len(project.todos),
            "total_decisions": len(project.decisions),
        },
    }


# ─────────────────────────────────────────────
# Git helper
# ─────────────────────────────────────────────

class GitHelper:
    """Interface simplifiée avec git."""

    def __init__(self, root: Path):
        self.root = root
        self._check_git()

    def _check_git(self) -> None:
        """Vérifie que git est disponible et le repo initialisé."""
        if not (self.root / ".git").exists():
            logger.info("Initialisation du repo git...")
            self._run("init")
            gitignore = self.root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "*.tif\n*.tiff\ndata/\noutput/\n"
                    "__pycache__/\n.sessions/backups/\n",
                    encoding="utf-8",
                )
            self._run("add", "-A")
            self._run("commit", "-m", "Initial commit")

    def _run(self, *args: str, check: bool = True) -> str:
        """Exécute une commande git."""
        cmd = ["git", "-C", str(self.root)] + list(args)
        logger.debug(f"  git {' '.join(args)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            logger.error(f"Git error: {error_msg}")
            raise RuntimeError(f"git {args[0]} failed: {error_msg}")
        return result.stdout.strip()

    def current_branch(self) -> str:
        return self._run("branch", "--show-current")

    def branch_exists(self, name: str) -> bool:
        result = self._run(
            "branch", "--list", name, check=False
        )
        return bool(result.strip())

    def create_branch(self, name: str) -> None:
        self._run("checkout", "-b", name)

    def switch_branch(self, name: str) -> None:
        self._run("checkout", name)

    def delete_branch(self, name: str) -> None:
        self._run("branch", "-D", name)

    def commit_all(self, message: str) -> None:
        self._run("add", "-A")
        status = self._run("status", "--porcelain")
        if status:
            self._run("commit", "-m", message)
        else:
            logger.debug("  Rien à committer")

    def merge_branch(
        self, branch: str, no_ff: bool = True
    ) -> tuple[bool, str]:
        """Merge une branche. Retourne (success, output)."""
        args = ["merge", branch]
        if no_ff:
            args.append("--no-ff")
        result = subprocess.run(
            ["git", "-C", str(self.root)] + args,
            capture_output=True,
            text=True,
            check=False,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output

    def abort_merge(self) -> None:
        self._run("merge", "--abort", check=False)

    def get_diff_files(self, branch_a: str, branch_b: str) -> list[str]:
        """Liste les fichiers modifiés entre deux branches."""
        output = self._run(
            "diff", "--name-only", f"{branch_a}...{branch_b}",
            check=False,
        )
        return [f for f in output.splitlines() if f.strip()]

    def get_file_hash(self, filepath: str) -> str:
        """Hash MD5 d'un fichier."""
        full_path = self.root / filepath
        if not full_path.exists():
            return "deleted"
        content = full_path.read_bytes()
        return hashlib.md5(content).hexdigest()[:10]

    def stash_save(self, message: str = "") -> None:
        self._run("stash", "save", message, check=False)

    def stash_pop(self) -> None:
        self._run("stash", "pop", check=False)

    def has_uncommitted_changes(self) -> bool:
        status = self._run("status", "--porcelain")
        return bool(status.strip())


# ─────────────────────────────────────────────
# Lock Manager
# ─────────────────────────────────────────────

class LockManager:
    """Gère les verrous sur les fichiers."""

    def __init__(self, git: GitHelper):
        self.git = git
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        BACKUP_DIR.mkdir(exist_ok=True)

    def _load_locks(self) -> dict[str, FileLock]:
        """Charge les verrous depuis le fichier JSON."""
        if not LOCKS_FILE.exists():
            return {}
        data = json.loads(LOCKS_FILE.read_text(encoding="utf-8"))
        return {
            k: FileLock(**v) for k, v in data.items()
        }

    def _save_locks(self, locks: dict[str, FileLock]) -> None:
        """Sauvegarde les verrous."""
        data = {k: asdict(v) for k, v in locks.items()}
        LOCKS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def acquire_locks(
        self,
        session_name: str,
        exclusive_files: list[str],
        read_only_files: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Tente d'acquérir les verrous pour une session."""
        locks = self._load_locks()
        errors: list[str] = []
        now = datetime.now(timezone.utc).isoformat()

        for filename in exclusive_files:
            if filename in AUTO_GENERATED:
                continue

            key = filename

            if key in locks:
                existing = locks[key]
                if existing.session_name != session_name:
                    if existing.lock_type == LockType.EXCLUSIVE:
                        errors.append(
                            f"❌ {filename} verrouillé "
                            f"exclusivement par "
                            f"session '{existing.session_name}'"
                        )
                    elif (
                        existing.lock_type == LockType.SHARED
                        and filename not in SHARED_READ_FILES
                    ):
                        errors.append(
                            f"⚠️ {filename} en accès partagé "
                            f"par '{existing.session_name}' — "
                            f"risque de conflit"
                        )

        if errors:
            return False, errors

        for filename in exclusive_files:
            if filename in AUTO_GENERATED:
                continue

            lock_type = (
                LockType.SHARED
                if filename in SHARED_READ_FILES
                else LockType.EXCLUSIVE
            )

            locks[filename] = FileLock(
                filename=filename,
                session_name=session_name,
                lock_type=lock_type,
                locked_at=now,
                file_hash_at_lock=self.git.get_file_hash(filename),
            )

        for filename in (read_only_files or []):
            if filename not in locks:
                locks[filename] = FileLock(
                    filename=filename,
                    session_name=session_name,
                    lock_type=LockType.READ_ONLY,
                    locked_at=now,
                    file_hash_at_lock=self.git.get_file_hash(
                        filename
                    ),
                )

        self._save_locks(locks)
        return True, []

    def release_locks(self, session_name: str) -> int:
        """Libère tous les verrous d'une session. Retourne le nb libéré."""
        locks = self._load_locks()
        to_remove = [
            k for k, v in locks.items()
            if v.session_name == session_name
        ]
        for k in to_remove:
            del locks[k]
        self._save_locks(locks)
        return len(to_remove)

    def get_session_locks(
        self, session_name: str
    ) -> list[FileLock]:
        """Retourne les verrous d'une session."""
        locks = self._load_locks()
        return [
            v for v in locks.values()
            if v.session_name == session_name
        ]

    def get_all_locks(self) -> dict[str, FileLock]:
        return self._load_locks()

    def check_file_modified_since_lock(
        self, filename: str
    ) -> bool:
        """Vérifie si un fichier a été modifié depuis le verrouillage."""
        locks = self._load_locks()
        if filename not in locks:
            return False
        lock = locks[filename]
        current_hash = self.git.get_file_hash(filename)
        return current_hash != lock.file_hash_at_lock

    def detect_cross_conflicts(self) -> list[MergeConflict]:
        """Détecte les conflits potentiels entre sessions actives."""
        locks = self._load_locks()
        conflicts: list[MergeConflict] = []

        file_sessions: dict[str, list[FileLock]] = {}
        for lock in locks.values():
            if lock.filename not in file_sessions:
                file_sessions[lock.filename] = []
            file_sessions[lock.filename].append(lock)

        for filename, file_locks in file_sessions.items():
            exclusive = [
                l for l in file_locks
                if l.lock_type == LockType.EXCLUSIVE
            ]
            if len(exclusive) > 1:
                conflicts.append(MergeConflict(
                    filename=filename,
                    session_a=exclusive[0].session_name,
                    session_b=exclusive[1].session_name,
                    conflict_type="dual_exclusive",
                    details=(
                        f"Deux sessions verrouillent "
                        f"exclusivement {filename}"
                    ),
                ))

        return conflicts


# ─────────────────────────────────────────────
# Session Manager
# ─────────────────────────────────────────────

class SessionManager:
    """Orchestrateur principal des sessions parallèles."""

    def __init__(self):
        self.git = GitHelper(PROJECT_ROOT)
        self.locks = LockManager(self.git)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        BACKUP_DIR.mkdir(exist_ok=True)

    def _session_file(self, name: str) -> Path:
        return SESSIONS_DIR / f"{name}.json"

    def _load_session(self, name: str) -> Session | None:
        path = self._session_file(name)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)

    def _save_session(self, session: Session) -> None:
        path = self._session_file(session.name)
        path.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_history(self) -> list[dict[str, Any]]:
        if not HISTORY_FILE.exists():
            return []
        return json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )

    def _append_history(self, entry: dict[str, Any]) -> None:
        history = self._load_history()
        history.append(entry)
        HISTORY_FILE.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Commandes principales ─────────────────

    def create(
        self,
        name: str,
        focus_files: list[str],
        description: str = "",
        read_only: list[str] | None = None,
    ) -> Session:
        """Crée une nouvelle session parallèle."""
        if self._session_file(name).exists():
            existing = self._load_session(name)
            if existing and existing.state == SessionState.ACTIVE:
                raise ValueError(
                    f"Session '{name}' existe déjà et est active. "
                    f"Utilisez 'session abort {name}' d'abord."
                )

        for f in focus_files:
            if not (PROJECT_ROOT / f).exists():
                raise FileNotFoundError(f"Fichier non trouvé : {f}")

        success, errors = self.locks.acquire_locks(
            name, focus_files, read_only
        )
        if not success:
            raise RuntimeError(
                "Impossible d'acquérir les verrous :\n"
                + "\n".join(errors)
            )

        all_py = [
            f.name for f in PROJECT_ROOT.glob("*.py")
            if f.name not in AUTO_GENERATED
        ]
        exclude = [
            f for f in all_py
            if f not in focus_files
            and f not in (read_only or [])
        ]

        current_branch = self.git.current_branch()
        if self.git.has_uncommitted_changes():
            self.git.commit_all(
                f"Auto-save before session '{name}'"
            )

        branch_name = f"session/{name}"
        if self.git.branch_exists(branch_name):
            self.git.delete_branch(branch_name)

        self.git.create_branch(branch_name)

        now = datetime.now(timezone.utc).isoformat()
        session = Session(
            name=name,
            state=SessionState.ACTIVE,
            created_at=now,
            focus_files=focus_files,
            exclude_files=exclude,
            read_only_files=read_only or [],
            description=description,
            git_branch=branch_name,
            context_file=f"AI_CONTEXT_{name}.md",
        )

        self._save_session(session)
        self._generate_session_context(session)
        self.git.switch_branch(current_branch)

        self._append_history({
            "action": "create",
            "session": name,
            "files": focus_files,
            "timestamp": now,
        })

        logger.info(f"✅ Session '{name}' créée")
        logger.info(f"   Branche : {branch_name}")
        logger.info(f"   Fichiers : {', '.join(focus_files)}")
        logger.info(f"   Contexte : {session.context_file}")

        return session

    def generate_context(self, name: str) -> Path:
        """Regénère le contexte pour une session."""
        session = self._load_session(name)
        if not session:
            raise ValueError(f"Session '{name}' introuvable")
        return self._generate_session_context(session)

    def _generate_session_context(self, session: Session) -> Path:
        """Génère AI_CONTEXT_<session>.md via le générateur intégré."""
        output_path = PROJECT_ROOT / session.context_file

        logger.info(
            f"Génération contexte pour session '{session.name}'..."
        )

        analyzer = ProjectAnalyzer(PROJECT_ROOT)
        project = analyzer.analyze()

        generator = ClaudeContextGenerator(
            project=project,
            compact=False,
            focus=", ".join(session.focus_files),
            exclude=session.exclude_files,
            session=session,
        )
        content = generator.generate()
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"✅ Contexte : {output_path}")
        return output_path

    def apply_changes(
        self,
        session_name: str,
        filename: str,
        new_content: str,
        description: str = "",
    ) -> FileChange:
        """Applique un changement de code à un fichier de la session."""
        session = self._load_session(session_name)
        if not session:
            raise ValueError(f"Session '{session_name}' introuvable")
        if session.state != SessionState.ACTIVE:
            raise RuntimeError(
                f"Session '{session_name}' n'est pas active "
                f"(état: {session.state.value})"
            )
        if filename not in session.focus_files:
            raise PermissionError(
                f"Fichier '{filename}' n'est pas dans le scope "
                f"de la session '{session_name}'. "
                f"Focus : {session.focus_files}"
            )

        filepath = PROJECT_ROOT / filename

        hash_before = self.git.get_file_hash(filename)

        if filepath.exists():
            backup_path = (
                BACKUP_DIR
                / f"{filename}.{session_name}"
                f".{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                f".bak"
            )
            shutil.copy2(filepath, backup_path)

        old_lines = (
            filepath.read_text(encoding="utf-8").splitlines()
            if filepath.exists() else []
        )
        new_lines = new_content.splitlines()

        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        ))
        lines_added = sum(
            1 for l in diff
            if l.startswith("+") and not l.startswith("+++")
        )
        lines_removed = sum(
            1 for l in diff
            if l.startswith("-") and not l.startswith("---")
        )

        original_branch = self.git.current_branch()
        self.git.switch_branch(session.git_branch)

        filepath.write_text(new_content, encoding="utf-8")

        hash_after = self.git.get_file_hash(filename)
        now = datetime.now(timezone.utc).isoformat()

        self.git.commit_all(
            f"[{session_name}] Update {filename}: {description}"
        )

        self.git.switch_branch(original_branch)

        change = FileChange(
            filename=filename,
            hash_before=hash_before,
            hash_after=hash_after,
            lines_added=lines_added,
            lines_removed=lines_removed,
            timestamp=now,
            description=description,
        )
        session.changes.append(change)
        self._save_session(session)

        logger.info(
            f"✅ {filename} mis à jour dans session '{session_name}' "
            f"(+{lines_added} -{lines_removed})"
        )

        return change

    def merge(self, session_name: str) -> MergeResult:
        """Fusionne une session dans main."""
        session = self._load_session(session_name)
        if not session:
            raise ValueError(f"Session '{session_name}' introuvable")
        if session.state not in (
            SessionState.ACTIVE, SessionState.CONFLICT
        ):
            raise RuntimeError(
                f"Session '{session_name}' ne peut pas être mergée "
                f"(état: {session.state.value})"
            )

        result = MergeResult(
            success=False,
            session_name=session_name,
        )

        logger.info(f"🔀 Fusion de la session '{session_name}'...")

        cross_conflicts = self.locks.detect_cross_conflicts()
        relevant = [
            c for c in cross_conflicts
            if c.session_a == session_name
            or c.session_b == session_name
        ]
        if relevant:
            for c in relevant:
                result.conflicts.append(c)
                logger.warning(
                    f"⚠️ Conflit potentiel : {c.details}"
                )

        for filename in session.read_only_files:
            if self.locks.check_file_modified_since_lock(filename):
                result.warnings.append(
                    f"⚠️ {filename} (read-only) a été modifié "
                    f"depuis le verrouillage"
                )

        original_branch = self.git.current_branch()
        if original_branch not in ("main", "master"):
            if self.git.branch_exists("main"):
                main_branch = "main"
            elif self.git.branch_exists("master"):
                main_branch = "master"
            else:
                main_branch = original_branch
        else:
            main_branch = original_branch

        if self.git.has_uncommitted_changes():
            self.git.commit_all("Auto-save before merge")

        self.git.switch_branch(main_branch)

        session.state = SessionState.MERGING
        self._save_session(session)

        success, output = self.git.merge_branch(
            session.git_branch
        )

        if success:
            result.success = True
            result.files_merged = [
                c.filename for c in session.changes
            ]

            session.state = SessionState.MERGED
            session.merged_at = (
                datetime.now(timezone.utc).isoformat()
            )
            self._save_session(session)

            released = self.locks.release_locks(session_name)

            self._append_history({
                "action": "merge",
                "session": session_name,
                "files": result.files_merged,
                "timestamp": session.merged_at,
            })

            logger.info(
                f"✅ Session '{session_name}' fusionnée avec succès"
            )
            logger.info(
                f"   Fichiers : {', '.join(result.files_merged)}"
            )
            logger.info(f"   Verrous libérés : {released}")

            self._regenerate_main_context()

        else:
            self.git.abort_merge()
            self.git.switch_branch(main_branch)

            session.state = SessionState.CONFLICT
            self._save_session(session)

            result.conflicts.append(MergeConflict(
                filename="multiple",
                session_a=session_name,
                session_b=main_branch,
                conflict_type="git_merge_conflict",
                details=output[:500],
            ))

            logger.error(
                f"❌ Conflit lors de la fusion de '{session_name}'"
            )
            logger.error(f"   Détails : {output[:200]}")
            logger.info(
                "   Résolvez le conflit manuellement puis "
                "relancez le merge."
            )

        return result

    def merge_all(self) -> list[MergeResult]:
        """Fusionne toutes les sessions actives (une par une)."""
        sessions = self.list_sessions(state=SessionState.ACTIVE)
        results: list[MergeResult] = []

        if not sessions:
            logger.info("Aucune session active à fusionner.")
            return results

        sessions.sort(key=lambda s: s.created_at)

        for session in sessions:
            logger.info(f"\n{'═' * 40}")
            logger.info(
                f"Fusion de '{session.name}' "
                f"({len(session.changes)} changements)..."
            )
            try:
                result = self.merge(session.name)
                results.append(result)
                if not result.success:
                    logger.error(
                        f"⛔ Arrêt du merge-all : conflit dans "
                        f"'{session.name}'"
                    )
                    break
            except Exception as e:
                logger.error(
                    f"⛔ Erreur lors du merge de "
                    f"'{session.name}': {e}"
                )
                break

        return results

    def abort(self, session_name: str) -> None:
        """Annule une session et libère les verrous."""
        session = self._load_session(session_name)
        if not session:
            raise ValueError(f"Session '{session_name}' introuvable")

        released = self.locks.release_locks(session_name)

        current = self.git.current_branch()
        if current == session.git_branch:
            if self.git.branch_exists("main"):
                self.git.switch_branch("main")
            elif self.git.branch_exists("master"):
                self.git.switch_branch("master")

        if self.git.branch_exists(session.git_branch):
            self.git.delete_branch(session.git_branch)

        session.state = SessionState.ABORTED
        self._save_session(session)

        context_path = PROJECT_ROOT / session.context_file
        if context_path.exists():
            context_path.unlink()

        self._append_history({
            "action": "abort",
            "session": session_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            f"🗑️ Session '{session_name}' annulée. "
            f"{released} verrou(s) libéré(s)."
        )

    def status(self, session_name: str) -> dict[str, Any]:
        """Retourne le statut détaillé d'une session."""
        session = self._load_session(session_name)
        if not session:
            raise ValueError(f"Session '{session_name}' introuvable")

        locks = self.locks.get_session_locks(session_name)

        modified_since_lock: list[str] = []
        for lock in locks:
            if self.locks.check_file_modified_since_lock(
                lock.filename
            ):
                modified_since_lock.append(lock.filename)

        return {
            "session": session.to_dict(),
            "locks": [asdict(l) for l in locks],
            "changes_count": len(session.changes),
            "modified_since_lock": modified_since_lock,
            "cross_conflicts": [
                asdict(c)
                for c in self.locks.detect_cross_conflicts()
                if c.session_a == session_name
                or c.session_b == session_name
            ],
        }

    def list_sessions(
        self, state: SessionState | None = None
    ) -> list[Session]:
        """Liste les sessions, optionnellement filtrées par état."""
        sessions: list[Session] = []
        for path in SESSIONS_DIR.glob("*.json"):
            if path.name in ("locks.json", "history.json"):
                continue
            session = self._load_session(path.stem)
            if session:
                if state is None or session.state == state:
                    sessions.append(session)
        return sessions

    def _regenerate_main_context(self) -> None:
        """Regénère AI_CONTEXT.md après un merge."""
        logger.info("Régénération de AI_CONTEXT.md...")
        analyzer = ProjectAnalyzer(PROJECT_ROOT)
        project = analyzer.analyze()
        generator = ClaudeContextGenerator(project=project)
        content = generator.generate()
        DEFAULT_OUTPUT.write_text(content, encoding="utf-8")
        logger.info(f"✅ AI_CONTEXT.md régénéré")

    def get_history(self) -> list[dict[str, Any]]:
        return self._load_history()


# ─────────────────────────────────────────────
# CLI — Fonctions d'affichage
# ─────────────────────────────────────────────

def _log_session_table(sessions: list[Session]) -> None:
    """Affiche un tableau de sessions via logger."""
    if not sessions:
        logger.info("  (aucune session)")
        return

    logger.info(
        f"  {'Nom':<20s} {'État':<10s} {'Fichiers':<40s} "
        f"{'Changes':<8s} {'Créée':<20s}"
    )
    logger.info(f"  {'─' * 98}")

    state_emoji = {
        SessionState.ACTIVE: "🟢",
        SessionState.MERGING: "🔄",
        SessionState.MERGED: "✅",
        SessionState.ABORTED: "🗑️",
        SessionState.CONFLICT: "🔴",
    }

    for s in sessions:
        emoji = state_emoji.get(s.state, "?")
        files = ", ".join(s.focus_files)
        if len(files) > 38:
            files = files[:35] + "..."
        created = s.created_at[:19] if s.created_at else "?"
        logger.info(
            f"  {s.name:<20s} {emoji} {s.state.value:<8s} "
            f"{files:<40s} {len(s.changes):<8d} {created:<20s}"
        )


def _log_locks_table(locks: dict[str, Any]) -> None:
    """Affiche un tableau de verrous via logger."""
    if not locks:
        logger.info("  (aucun verrou actif)")
        return

    lock_emoji = {
        "exclusive": "🔒",
        "read_only": "👁️",
        "shared": "🔓",
    }

    logger.info(
        f"  {'Fichier':<30s} {'Session':<20s} "
        f"{'Type':<12s} {'Hash':<12s}"
    )
    logger.info(f"  {'─' * 74}")

    for filename, lock_data in sorted(locks.items()):
        if isinstance(lock_data, dict):
            lt = lock_data.get("lock_type", "?")
            emoji = lock_emoji.get(lt, "?")
            logger.info(
                f"  {filename:<30s} "
                f"{lock_data.get('session_name', '?'):<20s} "
                f"{emoji} {lt:<10s} "
                f"{lock_data.get('file_hash_at_lock', '?'):<12s}"
            )


# ─────────────────────────────────────────────
# CLI — Commandes session
# ─────────────────────────────────────────────

def cli_session_create(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    try:
        session = mgr.create(
            name=args.name,
            focus_files=args.files,
            description=args.description or "",
            read_only=args.read_only,
        )
        logger.info(f"✅ Session '{session.name}' créée avec succès")
        logger.info(f"   Contexte → {session.context_file}")
        logger.info(
            f"   Copiez le contenu de {session.context_file} "
            f"dans le Playground."
        )
        logger.info(
            f"   Quand vous avez le code modifié, utilisez :"
        )
        logger.info(
            f"   python prepare_context.py session apply "
            f"{session.name} --file <fichier>"
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error(f"Erreur : {e}")
        sys.exit(1)


def cli_session_list(args: argparse.Namespace) -> None:
    mgr = SessionManager()

    state_filter = None
    if hasattr(args, "state") and args.state:
        state_filter = SessionState(args.state)

    sessions = mgr.list_sessions(state=state_filter)

    logger.info("═══ Sessions ═══")
    _log_session_table(sessions)

    logger.info("═══ Verrous ═══")
    _log_locks_table(
        {k: asdict(v) for k, v in mgr.locks.get_all_locks().items()}
    )

    conflicts = mgr.locks.detect_cross_conflicts()
    if conflicts:
        logger.info("═══ ⚠️ Conflits potentiels ═══")
        for c in conflicts:
            logger.warning(
                f"  🔴 {c.filename}: {c.session_a} ↔ "
                f"{c.session_b} ({c.conflict_type})"
            )


def cli_session_status(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    try:
        status = mgr.status(args.name)
        session_data = status["session"]

        logger.info(f"═══ Session '{args.name}' ═══")
        logger.info(f"  État        : {session_data['state']}")
        logger.info(
            f"  Créée       : {session_data['created_at'][:19]}"
        )
        logger.info(
            f"  Focus       : "
            f"{', '.join(session_data['focus_files'])}"
        )
        logger.info(
            f"  Branche     : {session_data['git_branch']}"
        )
        logger.info(
            f"  Changements : {status['changes_count']}"
        )

        if session_data["changes"]:
            logger.info("  Historique :")
            for c in session_data["changes"]:
                logger.info(
                    f"    {c['timestamp'][:19]} | "
                    f"{c['filename']} | "
                    f"+{c['lines_added']} -{c['lines_removed']} | "
                    f"{c['description']}"
                )

        if status["modified_since_lock"]:
            logger.warning(
                "  ⚠️ Fichiers modifiés depuis le lock :"
            )
            for f in status["modified_since_lock"]:
                logger.warning(f"    - {f}")

        if status["cross_conflicts"]:
            logger.error("  🔴 Conflits potentiels :")
            for c in status["cross_conflicts"]:
                logger.error(f"    - {c}")

    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


def cli_session_context(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    try:
        path = mgr.generate_context(args.name)
        logger.info(f"✅ Contexte régénéré : {path}")
        line_count = path.read_text(encoding="utf-8").count("\n")
        logger.info(f"   {line_count} lignes")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


def cli_session_apply(args: argparse.Namespace) -> None:
    mgr = SessionManager()

    filepath = PROJECT_ROOT / args.file
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Fichier non trouvé : {args.input}")
            sys.exit(1)
        new_content = input_path.read_text(encoding="utf-8")
    elif args.stdin:
        logger.info(
            "Collez le code puis Ctrl+D (Unix) ou Ctrl+Z (Windows) :"
        )
        new_content = sys.stdin.read()
    else:
        if not filepath.exists():
            logger.error(f"{args.file} non trouvé")
            sys.exit(1)
        new_content = filepath.read_text(encoding="utf-8")

    try:
        change = mgr.apply_changes(
            session_name=args.name,
            filename=args.file,
            new_content=new_content,
            description=args.description or "",
        )
        logger.info(
            f"✅ {args.file} mis à jour dans session '{args.name}'"
        )
        logger.info(
            f"   +{change.lines_added} -{change.lines_removed} lignes"
        )
        logger.info(
            f"   Hash: {change.hash_before} → {change.hash_after}"
        )
    except (ValueError, RuntimeError, PermissionError) as e:
        logger.error(str(e))
        sys.exit(1)


def cli_session_merge(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    try:
        result = mgr.merge(args.name)

        if result.success:
            logger.info(
                f"✅ Session '{args.name}' fusionnée avec succès"
            )
            logger.info(
                f"   Fichiers : "
                f"{', '.join(result.files_merged)}"
            )
            for w in result.warnings:
                logger.warning(f"   {w}")
        else:
            logger.error(f"❌ Fusion échouée pour '{args.name}'")
            for c in result.conflicts:
                logger.error(f"   Conflit : {c.details[:200]}")
    except (ValueError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)


def cli_session_merge_all(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    results = mgr.merge_all()

    logger.info("═══ Résultats merge-all ═══")
    for r in results:
        status = "✅" if r.success else "❌"
        files = ", ".join(r.files_merged) or "(aucun)"
        logger.info(f"  {status} {r.session_name}: {files}")
        for c in r.conflicts:
            logger.error(f"     🔴 {c.details[:100]}")

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        f"  Total : {succeeded}/{len(results)} "
        f"session(s) fusionnée(s)"
    )


def cli_session_abort(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    try:
        if not args.force:
            confirm = input(
                f"Annuler la session '{args.name}' ? "
                f"Les changements non mergés seront perdus. "
                f"[y/N] "
            )
            if confirm.lower() != "y":
                logger.info("Annulation abandonnée.")
                return

        mgr.abort(args.name)
        logger.info(f"🗑️ Session '{args.name}' annulée")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


def cli_session_history(args: argparse.Namespace) -> None:
    mgr = SessionManager()
    history = mgr.get_history()

    if not history:
        logger.info("  (aucun historique)")
        return

    logger.info(f"═══ Historique ({len(history)} entrées) ═══")
    action_emoji = {
        "create": "🆕",
        "merge": "🔀",
        "abort": "🗑️",
    }
    for entry in history[-20:]:
        emoji = action_emoji.get(entry["action"], "?")
        ts = entry.get("timestamp", "?")[:19]
        files = ", ".join(entry.get("files", []))
        logger.info(
            f"  {ts} {emoji} {entry['action']:<8s} "
            f"{entry.get('session', '?'):<15s} {files}"
        )


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

def _build_context_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Ajoute le sous-parseur 'context' (comportement par défaut)."""
    p_ctx = subparsers.add_parser(
        "context",
        help="Générer AI_CONTEXT.md (défaut)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python prepare_context.py context
              python prepare_context.py context --focus grid_builder.py
              python prepare_context.py context --compact --exclude visualize.py
              python prepare_context.py context --verbose --dry-run
              python prepare_context.py context --json
        """),
    )
    p_ctx.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Fichier de sortie (défaut: {DEFAULT_OUTPUT.name})",
    )
    p_ctx.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Version compressée (~600 lignes)",
    )
    p_ctx.add_argument(
        "--focus", "-f",
        type=str,
        default=None,
        help=(
            "Focus de la session : nom de fichier ou description "
            "de tâche"
        ),
    )
    p_ctx.add_argument(
        "--exclude", "-x",
        nargs="*",
        default=None,
        help="Fichiers hors scope (résumé minimal)",
    )
    p_ctx.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Affiche sans écrire de fichier",
    )
    p_ctx.add_argument(
        "--json",
        action="store_true",
        help="Exporte aussi les métadonnées en JSON",
    )
    p_ctx.add_argument(
        "--max-style-samples",
        type=int,
        default=3,
        help="Nombre max d'extraits de style (défaut: 3)",
    )
    p_ctx.add_argument(
        "--budget",
        type=int,
        default=1750,
        help="Budget max de lignes pour le prompt (défaut: 1750)",
    )
    p_ctx.set_defaults(func=_run_context)


def _build_session_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Ajoute le sous-parseur 'session' avec ses sous-commandes."""
    p_session = subparsers.add_parser(
        "session",
        help="Gestionnaire de sessions parallèles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Workflow type :
              1. session create  → Crée une session avec verrous
              2. session context → Copier le contexte dans le Playground
              3. session apply   → Appliquer le code reçu de Claude
              4. session merge   → Fusionner dans main

            Exemples :
              python prepare_context.py session create twi --files grid_builder.py
              python prepare_context.py session list
              python prepare_context.py session status twi
              python prepare_context.py session merge twi
        """),
    )

    session_sub = p_session.add_subparsers(
        dest="session_command", required=True
    )

    # ── create ────────────────────────────────
    p_create = session_sub.add_parser(
        "create", help="Créer une session parallèle"
    )
    p_create.add_argument("name", help="Nom de la session")
    p_create.add_argument(
        "--files", "-f", nargs="+", required=True,
        help="Fichiers à verrouiller (modifiables)",
    )
    p_create.add_argument(
        "--read-only", "-r", nargs="*", default=None,
        help="Fichiers en lecture seule",
    )
    p_create.add_argument(
        "--description", "-d", default="",
        help="Description de la session",
    )
    p_create.set_defaults(func=cli_session_create)

    # ── list ──────────────────────────────────
    p_list = session_sub.add_parser(
        "list", help="Lister les sessions"
    )
    p_list.add_argument(
        "--state", choices=[s.value for s in SessionState],
        help="Filtrer par état",
    )
    p_list.set_defaults(func=cli_session_list)

    # ── status ────────────────────────────────
    p_status = session_sub.add_parser(
        "status", help="Statut détaillé d'une session"
    )
    p_status.add_argument("name", help="Nom de la session")
    p_status.set_defaults(func=cli_session_status)

    # ── context ───────────────────────────────
    p_ctx = session_sub.add_parser(
        "context", help="Regénérer le contexte AI d'une session"
    )
    p_ctx.add_argument("name", help="Nom de la session")
    p_ctx.set_defaults(func=cli_session_context)

    # ── apply ─────────────────────────────────
    p_apply = session_sub.add_parser(
        "apply", help="Appliquer du code modifié"
    )
    p_apply.add_argument("name", help="Nom de la session")
    p_apply.add_argument(
        "--file", required=True,
        help="Fichier à mettre à jour",
    )
    p_apply.add_argument(
        "--input", "-i",
        help="Lire le nouveau code depuis ce fichier",
    )
    p_apply.add_argument(
        "--stdin", action="store_true",
        help="Lire le nouveau code depuis stdin",
    )
    p_apply.add_argument(
        "--description", "-d", default="",
        help="Description du changement",
    )
    p_apply.set_defaults(func=cli_session_apply)

    # ── merge ─────────────────────────────────
    p_merge = session_sub.add_parser(
        "merge", help="Fusionner une session dans main"
    )
    p_merge.add_argument("name", help="Nom de la session")
    p_merge.set_defaults(func=cli_session_merge)

    # ── merge-all ─────────────────────────────
    p_merge_all = session_sub.add_parser(
        "merge-all", help="Fusionner toutes les sessions actives"
    )
    p_merge_all.set_defaults(func=cli_session_merge_all)

    # ── abort ─────────────────────────────────
    p_abort = session_sub.add_parser(
        "abort", help="Annuler une session"
    )
    p_abort.add_argument("name", help="Nom de la session")
    p_abort.add_argument(
        "--force", action="store_true",
        help="Pas de confirmation",
    )
    p_abort.set_defaults(func=cli_session_abort)

    # ── history ───────────────────────────────
    p_history = session_sub.add_parser(
        "history", help="Historique des actions"
    )
    p_history.set_defaults(func=cli_session_history)


def _run_context(args: argparse.Namespace) -> None:
    """Exécute la génération de contexte."""
    logger.info(f"Projet : {PROJECT_ROOT}")
    logger.info(f"Sortie : {args.output}")
    if args.focus:
        logger.info(f"Focus  : {args.focus}")
    if args.exclude:
        logger.info(f"Exclus : {', '.join(args.exclude)}")

    # ── Analyse ───────────────────────────────
    analyzer = ProjectAnalyzer(PROJECT_ROOT)
    project = analyzer.analyze()

    if args.max_style_samples != 3:
        logger.info(
            f"Re-extraction style avec {args.max_style_samples} "
            f"samples..."
        )
        extractor = StyleExtractor(project.modules)
        project.style_samples = extractor.extract_best_samples(
            max_samples=args.max_style_samples
        )

    # ── Génération ────────────────────────────
    generator = ClaudeContextGenerator(
        project=project,
        compact=args.compact,
        focus=args.focus,
        exclude=args.exclude,
    )
    content = generator.generate()

    line_count = content.count("\n") + 1
    budget_used_pct = (line_count / args.budget) * 100
    budget_remaining = args.budget - line_count

    logger.info(f"Contexte généré : {line_count} lignes")
    logger.info(
        f"Budget : {line_count}/{args.budget} "
        f"({budget_used_pct:.0f}%) — "
        f"{budget_remaining} lignes restantes"
    )

    if line_count > args.budget:
        logger.error(
            f"🔴 DÉPASSEMENT : {line_count} > {args.budget} lignes ! "
            f"Utilisez --compact et/ou --exclude"
        )
    elif line_count > args.budget * 0.85:
        logger.warning(
            f"🟠 Serré : {budget_used_pct:.0f}% du budget utilisé. "
            f"Reste {budget_remaining} lignes pour prompt + code."
        )
    elif line_count > args.budget * 0.70:
        logger.info(
            f"🟡 Correct : {budget_remaining} lignes restantes."
        )
    else:
        logger.info(
            f"🟢 Confortable : {budget_remaining} lignes restantes."
        )

    # ── Sortie ────────────────────────────────
    if args.dry_run:
        sys.stdout.write(content)
        sys.stdout.write(f"\n{'═' * 50}\n")
        sys.stdout.write(f"  DRY RUN — {line_count} lignes\n")
        sys.stdout.write(
            f"  Budget : {line_count}/{args.budget} "
            f"({budget_used_pct:.0f}%)\n"
        )
        sys.stdout.write(f"{'═' * 50}\n")
    else:
        args.output.write_text(content, encoding="utf-8")
        logger.info(f"✅ Écrit : {args.output}")

    if args.json and not args.dry_run:
        json_path = args.output.with_suffix(".json")
        json_data = _project_to_json(project)
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"✅ JSON : {json_path}")

    # ── Résumé ────────────────────────────────
    logger.info("")
    logger.info("═" * 50)
    logger.info(f"  Version projet   : {project.project_version}")
    logger.info(f"  Modules analysés : {len(project.modules)}")
    logger.info(f"  Fichiers données : {len(project.data_files)}")
    logger.info(f"  TODOs trouvés    : {len(project.todos)}")
    logger.info(
        f"  Décisions chargées: {len(project.decisions)}"
    )
    logger.info(f"  Rejets chargés   : {len(project.rejected)}")
    logger.info(
        f"  Style samples    : {len(project.style_samples)}"
    )
    logger.info(f"  Lignes générées  : {line_count}")
    logger.info(
        f"  Budget restant   : {budget_remaining} / {args.budget}"
    )
    logger.info("═" * 50)

    # ── Conseils automatiques ─────────────────
    if not DECISIONS_FILE.exists():
        logger.warning(
            "⚠️  DECISIONS.md non trouvé ! La section <forbidden> "
            "sera incomplète."
        )
        logger.warning(
            "   Créez DECISIONS.md à la racine du projet."
        )

    fixme_count = sum(
        1 for t in project.todos if t.tag == "FIXME"
    )
    if fixme_count > 0:
        logger.warning(
            f"⚠️  {fixme_count} FIXME trouvé(s) dans le code — "
            f"à traiter en priorité"
        )

    if line_count > args.budget:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Générateur de contexte IA + Gestionnaire de sessions "
            "parallèles pour Cartomorilles"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Commandes :
              context   Générer AI_CONTEXT.md (défaut si aucun argument)
              session   Gestionnaire de sessions parallèles

            Exemples :
              python prepare_context.py
              python prepare_context.py context --focus grid_builder.py
              python prepare_context.py context --compact
              python prepare_context.py session create twi --files grid_builder.py
              python prepare_context.py session list
              python prepare_context.py session merge twi
        """),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux",
    )

    subparsers = parser.add_subparsers(dest="command")

    _build_context_parser(subparsers)
    _build_session_parser(subparsers)

    # ── Parse et dispatch ─────────────────────
    # Si aucun argument → mode contexte par défaut
    if len(sys.argv) == 1:
        sys.argv.append("context")
    elif (
        len(sys.argv) >= 2
        and sys.argv[1].startswith("-")
        and sys.argv[1] not in ("-h", "--help")
    ):
        # `prepare_context.py --compact` → `prepare_context.py context --compact`
        sys.argv.insert(1, "context")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
    )

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()