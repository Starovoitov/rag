from __future__ import annotations

import shutil
from pathlib import Path

def cleanup_faiss_db(
    persist_directory: str = "data/faiss",
    index_name: str = ".",
    drop_persist_directory: bool = False,
) -> dict[str, bool]:
    """Clean up FAISS index folder and optionally remove all persisted FAISS data."""
    db_path = Path(persist_directory)
    index_path = db_path / index_name
    index_deleted = False
    directory_deleted = False

    if index_path.exists():
        shutil.rmtree(index_path)
        index_deleted = True

    if drop_persist_directory and db_path.exists():
        shutil.rmtree(db_path)
        directory_deleted = True

    return {"index_deleted": index_deleted, "directory_deleted": directory_deleted}
