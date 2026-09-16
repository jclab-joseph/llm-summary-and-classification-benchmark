"""Pinned-revision dataset downloads from the HuggingFace Hub.

Deliberately *not* using `datasets.load_dataset`: several of the source
datasets ship loading scripts, which recent `datasets` releases refuse to
execute, and script-based loading hides the exact revision behind a cache. A
direct `resolve/<commit-sha>/<path>` fetch pins the bytes, which is what
`dataset_revision` in the manifests is supposed to mean.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from llmbench.core.errors import DatasetError

__all__ = ["hf_resolve_url", "download_dataset_file"]

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def hf_resolve_url(repo: str, revision: str, path: str) -> str:
    return f"{HF_ENDPOINT}/datasets/{repo}/resolve/{revision}/{path}"


def download_dataset_file(
    repo: str,
    revision: str,
    path: str,
    dest_dir: Path,
    *,
    timeout: float = 300.0,
    force: bool = False,
    progress=None,
) -> Path:
    """Download one file at a pinned revision, caching it under ``dest_dir``.

    The local filename embeds the revision, so a revision bump never silently
    reuses stale bytes.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = repo.replace("/", "__")
    safe_path = path.replace("/", "__")
    dest = dest_dir / f"{safe_repo}@{revision[:12]}__{safe_path}"
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest

    url = hf_resolve_url(repo, revision, path)
    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=timeout, headers=headers
        ) as response:
            if response.status_code != 200:
                raise DatasetError(
                    f"failed to download {repo}@{revision}/{path}: HTTP {response.status_code}"
                )
            total = int(response.headers.get("content-length") or 0)
            task = progress.add_task(f"download {path}", total=total or None) if progress else None
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)
                    if progress and task is not None:
                        progress.update(task, advance=len(chunk))
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetError(f"failed to download {repo}@{revision}/{path}: {exc}") from exc

    tmp.replace(dest)
    return dest
