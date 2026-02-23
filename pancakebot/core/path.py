from pathlib import Path


def ensure_parent_dir(path: str) -> None:
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
