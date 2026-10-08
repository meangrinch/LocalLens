import ctypes
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

import cv2
import numpy as np
import send2trash
from PIL import Image

INDEX_SCHEMA_VERSION = "1"
INDEX_DB_FILENAME = "index.sqlite"
# schema_meta keys of one-time invalidations, set once they have run for an index.
FRAME_SAMPLING_KEY = "frame_sampling"
STILL_DECODING_KEY = "still_decoding"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
ALL_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
ORIENTATION_TAG = 0x0112
# Sampled frames at least this similar to an already kept frame are not stored.
FRAME_DISTINCT_THRESHOLD = 0.95


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


def distinct_frame_indices(
    embeddings: np.ndarray, threshold: float = FRAME_DISTINCT_THRESHOLD
) -> list[int]:
    """Indices of frames, in order, that differ from every earlier kept frame."""
    kept: list[int] = []
    for index, embedding in enumerate(embeddings):
        if all(float(embedding @ embeddings[k]) < threshold for k in kept):
            kept.append(index)
    return kept


def _decodes_differently(path: str) -> bool:
    """Whether EXIF orientation or multi-frame handling changes how `path` decodes.

    Reads headers only: PNG getexif() would decode the whole image to find eXIf.
    """
    if os.path.splitext(path)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        return False
    try:
        with Image.open(path) as img:
            exif = Image.Exif()
            if img.info.get("exif"):
                exif.load(img.info["exif"])
            return exif.get(ORIENTATION_TAG, 1) != 1 or getattr(
                img, "is_animated", False
            )
    except Exception:
        # Unreadable files fail at embedding instead; this must not abort the sync.
        return False


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def delete_media_embeddings(collection, media_ids: list[str]) -> None:
    """Deletes every Chroma record (all frames) belonging to the given media."""
    for batch in batched(list(media_ids), 500):
        collection.delete(where={"media_id": {"$in": batch}})


def kept_frame_rows(
    records: list[MediaRecord], embeddings: np.ndarray, frame_paths: list[str]
) -> dict[str, list[int]]:
    """Embedding rows of each record's distinct frames, keyed by media_id.

    `frame_paths` names the source file of each row; records without rows are left out.
    """
    rows_by_key: dict[str, list[int]] = {}
    for row, path in enumerate(frame_paths):
        rows_by_key.setdefault(path_key(path), []).append(row)
    kept = {}
    for record in records:
        rows = rows_by_key.get(record.path_key)
        if rows:
            frame_embeddings = embeddings[rows].astype(np.float32)
            kept[record.media_id] = [
                rows[i] for i in distinct_frame_indices(frame_embeddings)
            ]
    return kept


def write_media_embeddings(
    collection,
    records: list[MediaRecord],
    embeddings: np.ndarray,
    frame_paths: list[str],
) -> list[MediaRecord]:
    """Replaces each media file's Chroma records with its distinct frames.

    `frame_paths` names the source file of each embedding row. A file with one
    kept frame is stored under its media_id; several frames get `media_id:k`.
    Returns the records that had at least one embedding.
    """
    kept_by_id = kept_frame_rows(records, embeddings, frame_paths)
    written = [record for record in records if record.media_id in kept_by_id]
    if not written:
        return []

    delete_media_embeddings(collection, [record.media_id for record in written])
    ids, kept_rows, documents, metadatas = [], [], [], []
    for record in written:
        kept = kept_by_id[record.media_id]
        for frame, row in enumerate(kept):
            ids.append(
                record.media_id if len(kept) == 1 else f"{record.media_id}:{frame}"
            )
            kept_rows.append(row)
            documents.append(record.path)
            metadatas.append({**record.to_metadata(), "frame": frame})
    collection.add(
        ids=ids,
        embeddings=embeddings[kept_rows],
        documents=documents,
        metadatas=metadatas,
    )
    return written


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

    def folder_item_counts(self) -> tuple[list[tuple[str, int]], int]:
        """Returns ([(folder, items under it)], distinct items in the index)."""
        with self.connect() as conn:
            folders = conn.execute(
                "SELECT path, path_key FROM folders ORDER BY path"
            ).fetchall()
            counts = []
            for row in folders:
                # Count by path prefix so nested indexed folders include files owned by a parent.
                prefix = row["path_key"].rstrip(os.sep) + os.sep
                count = conn.execute(
                    "SELECT COUNT(*) FROM media WHERE substr(path_key, 1, ?) = ?",
                    (len(prefix), prefix),
                ).fetchone()[0]
                counts.append((row["path"], count))
            total = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
        return counts, total

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

    def _invalidate_once(self, key: str, is_stale) -> int:
        """Marks media whose path satisfies is_stale for re-embedding, once per index.

        Returns how many media were marked; 0 once `key` has run for this index.
        """
        with self.connect() as conn:
            done = conn.execute(
                "SELECT 1 FROM schema_meta WHERE key = ?", (key,)
            ).fetchone()
            if done:
                return 0
            rows = conn.execute("SELECT media_id, path FROM media").fetchall()
            stale = [(row["media_id"],) for row in rows if is_stale(row["path"])]
            # records_needing_embedding re-embeds any record whose mtime differs.
            conn.executemany("UPDATE media SET mtime = -1 WHERE media_id = ?", stale)
            conn.execute("INSERT INTO schema_meta(key, value) VALUES (?, '1')", (key,))
        return len(stale)

    def invalidate_animated_media(self) -> int:
        """Videos/GIFs/WebPs that 1.6 embedded from their first frame only."""
        animated = set(VIDEO_EXTENSIONS) | {".gif", ".webp"}
        return self._invalidate_once(
            FRAME_SAMPLING_KEY,
            lambda path: os.path.splitext(path)[1].lower() in animated,
        )

    def invalidate_redecoded_stills(self) -> int:
        """Images whose decoding changed since 1.6: EXIF-rotated, MPO and APNG files."""
        return self._invalidate_once(STILL_DECODING_KEY, _decodes_differently)

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
            # By path, so files owned by an enclosing indexed folder are included.
            prefix = path_key(folder_path).rstrip(os.sep) + os.sep
            where_clause = "WHERE substr(path_key, 1, ?) = ?"
            params += [len(prefix), prefix]
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM media {where_clause} ORDER BY RANDOM() LIMIT ?",
                params + [limit],
            ).fetchall()
            return [record_from_row(row) for row in rows]


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


def query_filter_for_folder(folder_path: str | None) -> dict:
    """Chroma query kwargs limiting results to media under `folder_path`.

    Matches on the stored path (the document) rather than folder_id, so files owned
    by an enclosing indexed folder are included.
    """
    if not folder_path or folder_path == "All":
        return {}
    pattern = "^" + re.escape(canonical_path(folder_path).rstrip(os.sep) + os.sep)
    if os.name == "nt":
        # Case-insensitive like path_key; Chroma ignores (?i), so use [aA]-style classes.
        pattern = "".join(
            f"[{ch.lower()}{ch.upper()}]" if ch.lower() != ch.upper() else ch
            for ch in pattern
        )
    return {"where_document": {"$regex": pattern}}


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


def get_media_file_info(
    file_path: str,
    base_folders: list[str] | str | None = None,
) -> dict:
    """Retrieve metadata for a media file including dimensions and formatted size."""
    canonical = canonical_path(file_path)

    caption = canonical
    if base_folders:
        folder_list = (
            [base_folders] if isinstance(base_folders, str) else list(base_folders)
        )
        folder_list.sort(key=len, reverse=True)
        for folder in folder_list:
            if is_path_under_folder(canonical, folder):
                caption = relative_caption(canonical, folder)
                break
    else:
        caption = os.path.basename(canonical)

    if not os.path.exists(canonical):
        return {
            "path": canonical,
            "exists": False,
            "caption": caption,
            "size": 0,
            "size_str": "0 B",
            "dimensions": None,
            "dim_str": "Unknown",
            "mtime": 0,
            "mtime_str": "Unknown",
        }

    try:
        stat = os.stat(canonical)
        size = stat.st_size
        mtime = stat.st_mtime
    except OSError:
        size = 0
        mtime = 0

    if size >= 1024 * 1024 * 1024:
        size_str = f"{size / (1024 * 1024 * 1024):.2f} GB"
    elif size >= 1024 * 1024:
        size_str = f"{size / (1024 * 1024):.2f} MB"
    elif size >= 1024:
        size_str = f"{size / 1024:.1f} KB"
    else:
        size_str = f"{size} B"

    try:
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    except (OSError, OverflowError, ValueError):
        mtime_str = "Unknown"

    dimensions = None
    dim_str = "Unknown"
    try:
        if media_type_for_path(canonical) == "video":
            cap = cv2.VideoCapture(canonical)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            with Image.open(canonical) as img:
                w, h = img.size
        if w and h:
            dimensions = (w, h)
            dim_str = f"{w} × {h}"
    except Exception:
        pass

    return {
        "path": canonical,
        "exists": True,
        "caption": caption,
        "size": size,
        "size_str": size_str,
        "dimensions": dimensions,
        "dim_str": dim_str,
        "mtime": mtime,
        "mtime_str": mtime_str,
    }


def _reveal_windows(path: str) -> bool:
    # The shell API behind `explorer /select`: unlike that command (which ignored the
    # selection when launched from here), it selects reliably and reuses an open window.
    ole32, shell32 = ctypes.windll.ole32, ctypes.windll.shell32
    shell32.ILCreateFromPathW.restype = ctypes.c_void_p
    shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
    shell32.SHOpenFolderAndSelectItems.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    shell32.ILFree.argtypes = [ctypes.c_void_p]
    # Gradio runs handlers on worker threads, which need their own COM init.
    com_initialized = ole32.CoInitializeEx(None, 2) >= 0  # COINIT_APARTMENTTHREADED
    try:
        pidl = shell32.ILCreateFromPathW(path)
        if not pidl:
            return False
        try:
            return shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0) == 0
        finally:
            shell32.ILFree(pidl)
    finally:
        if com_initialized:
            ole32.CoUninitialize()


def reveal_command(path: str, platform: str = sys.platform) -> list[str]:
    """Command that opens the macOS/Linux file manager with `path` selected."""
    if platform == "darwin":
        return ["open", "-R", path]
    # freedesktop FileManager1 (Nautilus, Dolphin, Nemo, Caja, ...).
    return [
        "dbus-send",
        "--session",
        "--print-reply",
        "--reply-timeout=10000",
        "--dest=org.freedesktop.FileManager1",
        "--type=method_call",
        "/org/freedesktop/FileManager1",
        "org.freedesktop.FileManager1.ShowItems",
        f"array:string:{PurePosixPath(path).as_uri()}",
        "string:",
    ]


def reveal_in_file_manager(path: str) -> bool:
    """Opens the OS file manager with `path` selected; returns False if it could not."""
    target = canonical_path(path)
    if not os.path.isfile(target):
        return False
    if sys.platform == "win32":
        return _reveal_windows(target)
    command = reveal_command(target)
    if sys.platform == "darwin":
        try:
            subprocess.Popen(command)
            return True
        except OSError:
            return False
    try:
        result = subprocess.run(command, capture_output=True, timeout=15)
        # NoReply: a cold-started file manager missed the reply timeout but still
        # handles the call, so opening the folder too would show two windows.
        if result.returncode == 0 or b"NoReply" in result.stderr:
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    # No FileManager1 service: open the containing folder without selecting the file.
    try:
        opener = subprocess.Popen(["xdg-open", os.path.dirname(target)])
        return opener.wait(timeout=5) == 0
    except subprocess.TimeoutExpired:
        # Some openers stay in the foreground while the file manager runs.
        return True
    except OSError:
        return False


def trash_and_delete_indexed_media(
    file_path: str,
    db_path: str | None = None,
    collection=None,
) -> bool:
    """Move a media file to the OS Recycle Bin / Trash and remove its record from SQLite and ChromaDB."""
    target_path = canonical_path(file_path)
    file_existed = os.path.exists(target_path)

    if file_existed:
        try:
            send2trash.send2trash(target_path)
        except Exception as e:
            print(f"Error moving '{target_path}' to Recycle Bin: {e}")
            return False

    mid = media_id_for_path(target_path)
    if db_path:
        try:
            store = IndexStore(db_path)
            store.delete_media_ids([mid])
        except Exception as e:
            print(f"Error removing media {mid} from IndexStore: {e}")

    if collection is not None:
        try:
            delete_media_embeddings(collection, [mid])
        except Exception as e:
            print(f"Error removing media {mid} from ChromaDB: {e}")

    return True
