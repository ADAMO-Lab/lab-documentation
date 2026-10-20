"""Split the handbook's llms.txt into one Quarto page per rule group.

Usage: python3 scripts/port_guidelines.py <lab_programming_data_guidelines dir> coding-guidelines
Run book.py in the guidelines dir first so web/llms.txt is current.
The handbook PDF is not copied: it names who vetoed each rule, and this site is public."""
import re, shutil, sys
from pathlib import Path

src_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
text = (src_dir / "web/llms.txt").read_text()
head, *sections = re.split(r"^## ", text, flags=re.M)
ratified = re.search(r"Ratified (\d{4}-\d{2}-\d{2})", head).group(1)
n_rules = re.search(r"(\d+) rules adopted", head).group(1)

links = []
for order, sec in enumerate(sections, start=1):
    name, body = sec.split("\n", 1)
    name = name.strip()
    if name == "How to use this file":  # agent-only instructions; kept in llms.txt
        continue
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    (out / f"{slug}.qmd").write_text(
        f'---\ntitle: "{name}"\norder: {order}\n---\n\n{body.strip()}\n')
    links.append(f"- [{name}]({slug}.qmd)")

(out / "index.qmd").write_text(f"""---
title: "Coding Guidelines"
order: 0
---

The lab's coding and data management rules. {n_rules} rules were adopted by lab vote on {ratified}. Each rule has an ID (S1, V3, ...). Cite the ID when you apply a rule or flag a problem.

{chr(10).join(links)}

## Download

- [All rules in one Markdown file (`llms.txt`)](llms.txt), for AI coding agents to read

::: {{.callout-note}}
These pages are generated from the voting sources (`gen.py`, `book.py`, `votes.json`). To change a rule, change it there, run `book.py`, then `scripts/port_guidelines.py`. Do not edit these pages directly.
:::
""")
shutil.copy(src_dir / "web/llms.txt", out / "llms.txt")
print(f"Wrote {len(links)} pages to {out}")
