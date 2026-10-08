"""DatasetRegistry: everything the agent knows about each loaded dataset, built once at startup.

A dataset is servable when it has BOTH a folder (`datasets/<name>/dataset.yaml`) and an
`app.datasets` row (written by the loader). For each one the registry holds the validated config,
the catalog schema (tables, columns, comments, keys), `notes.md` and `fewshots.yaml`.
Nothing here knows any dataset by name (ADR-009).
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from psycopg import AsyncConnection
from psycopg.rows import tuple_row

from ask_data.datasets.config import DatasetCfg, load_dataset_cfg
from ask_data.schema.catalog import SchemaInfo, read_schema

log = logging.getLogger(__name__)

WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "by", "to", "and", "or", "is", "are", "was",
    "were", "what", "which", "who", "how", "many", "much", "me", "show", "list", "each", "with",
    "did", "do", "does", "has", "have", "most", "all", "per",
}  # fmt: skip


class UnknownDataset(LookupError):
    pass


@dataclass(frozen=True)
class Fewshot:
    question: str
    sql: str


@dataclass
class Dataset:
    cfg: DatasetCfg
    schema_info: SchemaInfo
    version: str
    loaded_at: datetime | None = None
    row_count: int = 0
    notes: str = ""
    fewshots: list[Fewshot] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def schema(self) -> str:
        return self.cfg.schema_name

    @property
    def title(self) -> str:
        return self.cfg.title

    @property
    def description(self) -> str:
        return " ".join(self.cfg.description.split())

    @property
    def tables(self) -> set[str]:
        """Tables that actually exist in the loaded schema (what guard() allows)."""
        return set(self.schema_info.tables)

    def schema_context(self, tables: list[str] | None = None) -> str:
        return self.schema_info.render(tables)

    def similar_fewshots(self, question: str, k: int = 3) -> list[Fewshot]:
        """Top-k few-shots by word overlap with the question (Jaccard on content words).

        Good enough for 3-8 examples; Phase 3 can switch to embeddings without changing callers.
        """
        q = _words(question)
        scored = [(_jaccard(q, _words(f.question)), i, f) for i, f in enumerate(self.fewshots)]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [f for score, _, f in scored[:k] if score > 0] or self.fewshots[:k]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "tables": sorted(self.tables),
            "examples": [f.question for f in self.fewshots[:4]],  # verified, safe to suggest
            "version": self.version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
        }


class DatasetRegistry:
    def __init__(self, datasets: dict[str, Dataset]):
        self._datasets = datasets

    def get(self, name: str) -> Dataset:
        try:
            return self._datasets[name]
        except KeyError:
            raise UnknownDataset(name) from None

    def __contains__(self, name: object) -> bool:
        return name in self._datasets

    def all(self) -> list[Dataset]:
        return [self._datasets[n] for n in sorted(self._datasets)]

    @classmethod
    async def load(cls, datasets_dir: str | Path, conn: AsyncConnection) -> "DatasetRegistry":
        datasets_dir = Path(datasets_dir)
        async with conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute("SELECT name, version, loaded_at, row_count FROM app.datasets")
            rows = await cur.fetchall()

        datasets: dict[str, Dataset] = {}
        for name, version, loaded_at, row_count in rows:
            folder = datasets_dir / name
            if not (folder / "dataset.yaml").exists():
                log.warning("dataset %r is loaded in Postgres but has no folder in %s; skipped",
                            name, datasets_dir)  # fmt: skip
                continue
            try:
                cfg = load_dataset_cfg(folder)
            except (ValueError, TypeError) as e:
                log.warning("dataset %r: invalid dataset.yaml, skipped: %s", name, e)
                continue
            schema_info = await read_schema(conn, cfg.schema_name)
            if not schema_info.tables:
                log.warning("dataset %r: schema %r has no tables; skipped", name, cfg.schema_name)
                continue
            datasets[name] = Dataset(
                cfg=cfg,
                schema_info=schema_info,
                version=version,
                loaded_at=loaded_at,
                row_count=row_count,
                notes=read_notes(folder),
                fewshots=read_fewshots(folder),
            )
        log.info("dataset registry: %s", ", ".join(sorted(datasets)) or "(empty)")
        return cls(datasets)


def read_notes(folder: Path) -> str:
    path = folder / "notes.md"
    return path.read_text().strip() if path.exists() else ""


def read_fewshots(folder: Path) -> list[Fewshot]:
    path = folder / "fewshots.yaml"
    if not path.exists():
        return []
    items = yaml.safe_load(path.read_text()) or []
    return [Fewshot(i["question"].strip(), i["sql"].strip()) for i in items]


def _words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
