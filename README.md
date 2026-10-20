# lab-documentation

ADAMO Lab documentation site, built with [Quarto](https://quarto.org).

| Folder | Section |
|---|---|
| `coding-guidelines/` | Adopted coding and data management rules |
| `ai-models/` | Model cards |
| `behavior-experiments/` | Task protocols |
| `datasets/` | Dataset cards |
| `eeg-eyetracking/` | EEG and eye-tracking setup and procedures |
| `hpc/` | Cluster access and Slurm recipes |

## Quick start

```bash
quarto preview      # live preview at http://localhost:4200 (or the port it prints)
quarto render       # build the site into _site/
```

To add a page, copy the `_template.qmd` in the section folder. See [`contributing.qmd`](contributing.qmd) for details.

## Publishing

`.github/workflows/publish.yml` renders the site on every push and pull request. It deploys to GitHub Pages only when the repository variable `PUBLISH_PAGES` is `true` and Pages is set to deploy from GitHub Actions.

A GitHub Pages site is public, even when the repository is private. Do not publish pages that contain internal paths or hostnames you want kept private.

## Coding guidelines source

The `coding-guidelines/` pages are generated from the lab's voting sources (`gen.py`, `book.py`, `votes.json`) by `scripts/port_guidelines.py`. Edit the rules there, not in these pages.
