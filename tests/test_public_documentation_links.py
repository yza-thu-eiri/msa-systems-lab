"""Publication gate: all relative Markdown file and figure links must resolve."""

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?(?:\\[[^\\]]*\\])\\(([^)]+)\\)")


def test_public_readme_and_study_links_resolve():
    pages = [ROOT / "README.md"]
    for section in ("docs", "benchmarks", "evidence"):
        pages.extend(sorted((ROOT / section).rglob("*.md")))
    broken = []
    for page in pages:
        for raw in MARKDOWN_LINK.findall(page.read_text(encoding="utf-8")):
            target = raw.strip().split("#", 1)[0].split("?", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            destination = (page.parent / unquote(target)).resolve()
            if not destination.is_relative_to(ROOT.resolve()) or not destination.exists():
                broken.append(f"{page.relative_to(ROOT)} -> {raw}")
    assert not broken, "Broken or escaping Markdown links:\\n" + "\\n".join(broken)
