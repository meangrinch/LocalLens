import hashlib
import os
import random
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

INDEX_SCHEMA_VERSION = "1"
INDEX_DB_FILENAME = "index.sqlite"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
ALL_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS


@dataclass(frozen=True)
class MediaRecord:
    media_id: str
    path: str
    path_key: str
    folder_id: str
    folder_path: str
    folder_key: str
    relative_path: str
    media_type: str
    mtime: float
    size: int

    def to_metadata(self) -> dict:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "media_id": self.media_id,
            "path": self.path,
            "path_key": self.path_key,
            "folder_id": self.folder_id,
            "folder_path": self.folder_path,
            "folder_key": self.folder_key,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "mtime": self.mtime,
            "size": self.size,
        }


def canonical_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def path_key(path: str) -> str:
    return os.path.normcase(canonical_path(path))


def stable_id(path_or_key: str) -> str:
    return hashlib.sha256(path_or_key.encode("utf-8")).hexdigest()


def folder_id_for_path(folder_path: str) -> str:
    return stable_id(path_key(folder_path))


def media_id_for_path(media_path: str) -> str:
    return stable_id(path_key(media_path))


def is_media_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ALL_EXTENSIONS


def media_type_for_path(path: str) -> str:
    return "video" if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS else "image"


def is_path_under_folder(path: str, folder_path: str) -> bool:
    try:
        child_key = path_key(path)
        parent_key = path_key(folder_path)
        return os.path.commonpath([child_key, parent_key]) == parent_key
    except ValueError:
        return False


def relative_caption(path: str, folder_path: str) -> str:
    if not is_path_under_folder(path, folder_path):
        return canonical_path(path)
    folder_abs = canonical_path(folder_path)
    folder_name = os.path.basename(folder_abs.rstrip(os.sep)) or folder_abs
    try:
        child_path = os.path.relpath(canonical_path(path), folder_abs)
        return os.path.join(folder_name, child_path)
    except ValueError:
        return canonical_path(path)


def scan_media_paths(folder_path: str) -> list[str]:
    root_folder = canonical_path(folder_path)
    media_paths = []
    for root, _, files in os.walk(root_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if is_media_file(file_path):
                media_paths.append(canonical_path(file_path))
    media_paths.sort(key=path_key)
    return media_paths


def build_media_record(media_path: str, folder_path: str) -> MediaRecord:
    media_abs = canonical_path(media_path)
    folder_abs = canonical_path(folder_path)
    media_key = path_key(media_abs)
    folder_key = path_key(folder_abs)
    try:
        stat_result = os.stat(media_abs)
        mtime = stat_result.st_mtime
        size = stat_result.st_size
    except OSError:
        mtime = 0.0
        size = 0
    return MediaRecord(
        media_id=stable_id(media_key),
        path=media_abs,
        path_key=media_key,
        folder_id=stable_id(folder_key),
        folder_path=folder_abs,
        folder_key=folder_key,
        relative_path=relative_caption(media_abs, folder_abs),
        media_type=media_type_for_path(media_abs),
        mtime=mtime,
        size=size,
    )


def build_media_records(folder_path: str) -> list[MediaRecord]:
    return [
        build_media_record(path, folder_path) for path in scan_media_paths(folder_path)
    ]


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class IndexStore:
    def __init__(self, db_path: str):
        self.db_path = canonical_path(db_path)
        self.sqlite_path = os.path.join(self.db_path, INDEX_DB_FILENAME)
        os.makedirs(self.db_path, exist_ok=True)
        self.ensure_schema()

    def _open_connection(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connect(self):
        conn = self._open_connection()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS folders (
                    folder_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    path_key TEXT NOT NULL UNIQUE,
                    added_at REAL NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media (
                    media_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    path_key TEXT NOT NULL UNIQUE,
                    folder_id TEXT NOT NULL,
                    folder_path TEXT NOT NULL,
                    folder_key TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(folder_id) REFERENCES folders(folder_id)
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_folder ON media(folder_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_path_key ON media(path_key)"
            )

    def schema_version(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            return row["value"] if row else None

    def mark_schema_current(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (INDEX_SCHEMA_VERSION,),
            )

    def upsert_folder(self, folder_path: str) -> str:
        folder_abs = canonical_path(folder_path)
        folder_key = path_key(folder_abs)
        folder_id = stable_id(folder_key)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO folders(folder_id, path, path_key, added_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path_key) DO UPDATE SET path = excluded.path
                """,
                (folder_id, folder_abs, folder_key, time.time()),
            )
        return folder_id

    def list_folders(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT path FROM folders ORDER BY path").fetchall()
            return [row["path"] for row in rows]

    def get_folder_id(self, folder_path: str) -> str | None:
        key = path_key(folder_path)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT folder_id FROM folders WHERE path_key = ?", (key,)
            ).fetchone()
            return row["folder_id"] if row else None

    def find_folder_for_path(self, media_path: str) -> str | None:
        folders = self.list_folders()
        matching = [
            folder for folder in folders if is_path_under_folder(media_path, folder)
        ]
        if not matching:
            return None
        matching.sort(key=lambda item: len(path_key(item)), reverse=True)
        return matching[0]

    def upsert_media_records(self, records: list[MediaRecord]) -> None:
        if not records:
            return
        now = time.time()
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO media(
                    media_id, path, path_key, folder_id, folder_path, folder_key,
                    relative_path, media_type, mtime, size, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    path = excluded.path,
                    path_key = excluded.path_key,
                    folder_id = excluded.folder_id,
                    folder_path = excluded.folder_path,
                    folder_key = excluded.folder_key,
                    relative_path = excluded.relative_path,
                    media_type = excluded.media_type,
                    mtime = excluded.mtime,
                    size = excluded.size,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        record.media_id,
                        record.path,
                        record.path_key,
                        record.folder_id,
                        record.folder_path,
                        record.folder_key,
                        record.relative_path,
                        record.media_type,
                        record.mtime,
                        record.size,
                        now,
                    )
                    for record in records
                ],
            )

    def get_media_by_folder(self, folder_path: str) -> list[MediaRecord]:
        folder_id = folder_id_for_path(folder_path)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM media WHERE folder_id = ? ORDER BY path", (folder_id,)
            ).fetchall()
            return [record_from_row(row) for row in rows]

    def get_media_by_ids(self, media_ids: list[str]) -> dict[str, MediaRecord]:
        if not media_ids:
            return {}
        result = {}
        with self.connect() as conn:
            for batch in batched(media_ids, 500):
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT * FROM media WHERE media_id IN ({placeholders})", batch
                ).fetchall()
                result.update({row["media_id"]: record_from_row(row) for row in rows})
        return result

    def get_media_by_paths(self, paths: list[str]) -> dict[str, MediaRecord]:
        if not paths:
            return {}
        keys = [path_key(path) for path in paths]
        result = {}
        with self.connect() as conn:
            for batch in batched(keys, 500):
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT * FROM media WHERE path_key IN ({placeholders})", batch
                ).fetchall()
                result.update({row["path_key"]: record_from_row(row) for row in rows})
        return result

    def records_needing_embedding(
        self, records: list[MediaRecord]
    ) -> list[MediaRecord]:
        existing = self.get_media_by_ids([record.media_id for record in records])
        needing = []
        for record in records:
            old = existing.get(record.media_id)
            if old is None or old.mtime != record.mtime or old.size != record.size:
                needing.append(record)
        return needing

    def delete_media_ids(self, media_ids: list[str]) -> None:
        if not media_ids:
            return
        with self.connect() as conn:
            for batch in batched(media_ids, 500):
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"DELETE FROM media WHERE media_id IN ({placeholders})", batch
                )

    def delete_folder(self, folder_path: str) -> tuple[bool, list[str]]:
        folder_id = folder_id_for_path(folder_path)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT media_id FROM media WHERE folder_id = ?", (folder_id,)
            ).fetchall()
            media_ids = [row["media_id"] for row in rows]
            folder_row = conn.execute(
                "SELECT folder_id FROM folders WHERE folder_id = ?", (folder_id,)
            ).fetchone()
            conn.execute("DELETE FROM media WHERE folder_id = ?", (folder_id,))
            conn.execute("DELETE FROM folders WHERE folder_id = ?", (folder_id,))
        return bool(folder_row or media_ids), media_ids

    def sample_media(
        self, limit: int, folder_path: str | None = None
    ) -> list[MediaRecord]:
        if limit <= 0:
            return []
        params: list = []
        where_clause = ""
        if folder_path and folder_path != "All":
            where_clause = "WHERE folder_id = ?"
            params.append(folder_id_for_path(folder_path))
        with self.connect() as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM media {where_clause}", params
            ).fetchone()
            total = count_row["total"] if count_row else 0
            if total == 0:
                return []
            limit = min(limit, total)
            offsets = sorted(random.sample(range(total), limit))
            records = []
            for offset in offsets:
                row = conn.execute(
                    f"SELECT * FROM media {where_clause} ORDER BY path LIMIT 1 OFFSET ?",
                    params + [offset],
                ).fetchone()
                if row:
                    records.append(record_from_row(row))
            random.shuffle(records)
            return records


def record_from_row(row: sqlite3.Row) -> MediaRecord:
    path = row["path"]
    folder_path = row["folder_path"]
    return MediaRecord(
        media_id=row["media_id"],
        path=path,
        path_key=row["path_key"],
        folder_id=row["folder_id"],
        folder_path=folder_path,
        folder_key=row["folder_key"],
        relative_path=relative_caption(path, folder_path),
        media_type=row["media_type"],
        mtime=row["mtime"],
        size=row["size"],
    )


def read_legacy_indexed_folders(db_path: str) -> list[str]:
    indexed_folders_file = os.path.join(db_path, "indexed_folders.txt")
    if not os.path.exists(indexed_folders_file):
        return []
    with open(indexed_folders_file, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_indexed_folders_mirror(db_path: str, folders: list[str]) -> None:
    os.makedirs(db_path, exist_ok=True)
    indexed_folders_file = os.path.join(db_path, "indexed_folders.txt")
    with open(indexed_folders_file, "w") as handle:
        handle.write("\n".join(folders))


def query_filter_for_folder(folder_path: str | None) -> dict | None:
    if not folder_path or folder_path == "All":
        return None
    return {"folder_id": folder_id_for_path(folder_path)}


def path_and_caption_from_result(
    document: str | None, metadata: dict | None
) -> tuple[str, str]:
    metadata = metadata or {}
    path = metadata.get("path") or document or ""
    folder_path = metadata.get("folder_path")
    caption = (
        relative_caption(path, folder_path)
        if folder_path
        else metadata.get("relative_path") or os.path.basename(path)
    )
    return path, caption


def migrate_legacy_db(db_path: str, collection, batch_size: int = 500) -> int:
    store = IndexStore(db_path)
    if store.schema_version() == INDEX_SCHEMA_VERSION:
        return 0

    legacy_folders = [
        canonical_path(folder) for folder in read_legacy_indexed_folders(db_path)
    ]
    for folder in legacy_folders:
        store.upsert_folder(folder)

    migrated = 0
    ids_to_delete = []
    total = 0
    try:
        total = collection.count()
    except Exception as e:
        print(f"Warning: could not count Chroma records during migration: {e}")

    offset = 0
    while offset < total:
        try:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "embeddings", "metadatas"],
            )
        except Exception as e:
            print(f"Warning: could not fetch legacy Chroma batch at {offset}: {e}")
            break

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        metadatas = result.get("metadatas") or []
        records = []
        new_ids = []
        new_documents = []
        new_embeddings = []
        new_metadatas = []
        legacy_ids = []

        for index, chroma_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None
            document = documents[index] if index < len(documents) else None
            current_path = (metadata or {}).get("path") or document or chroma_id
            folder = store.find_folder_for_path(current_path)
            if not folder:
                continue
            record = build_media_record(current_path, folder)
            records.append(record)
            if chroma_id == record.media_id:
                continue
            embedding = embeddings[index] if index < len(embeddings) else None
            if embedding is None:
                continue
            new_ids.append(record.media_id)
            new_documents.append(record.path)
            new_embeddings.append(embedding)
            new_metadatas.append(record.to_metadata())
            legacy_ids.append(chroma_id)

        if records:
            store.upsert_media_records(records)
        if new_ids:
            collection.upsert(
                ids=new_ids,
                embeddings=new_embeddings,
                documents=new_documents,
                metadatas=new_metadatas,
            )
            ids_to_delete.extend(legacy_ids)
            migrated += len(new_ids)
        offset += batch_size

    for batch in batched(ids_to_delete, batch_size):
        try:
            collection.delete(ids=batch)
        except Exception as e:
            print(f"Warning: could not delete legacy Chroma IDs: {e}")

    folders = store.list_folders()
    write_indexed_folders_mirror(db_path, folders)
    store.mark_schema_current()
    if migrated:
        print(f"Migrated {migrated} Chroma records to stable media IDs.")
    return migrated
