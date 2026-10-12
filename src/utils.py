from pathlib import Path

def find_root(marker: str = ".git") -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"No {marker} found above {p}")