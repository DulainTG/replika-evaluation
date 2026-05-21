# replika-evaluation

Evaluation workspace for comparing multiple reproductions of the solenoidal turbulence paper and keeping the associated code, reports, and selected result artifacts in one repository.

## Repository layout

- `replika_workspace/` — main reproduction workspace, including source code, tests, and committed output artifacts.
- `copilot_gemini_3_flash/` — Gemini Flash evaluation workspace and report artifacts.
- `copilot_gemini_3.1_pro/` — Gemini 3.1 Pro evaluation workspace and report artifacts.
- `raw_data_tdad/` — local raw VTK dataset directory used during evaluation; ignored by Git.

## What is tracked

This repository is intended to keep:

- evaluation reports and prompts
- reproduction code and tests
- selected result artifacts that are useful to review or share

## What is ignored

The root `.gitignore` excludes local-only or generated files such as:

- virtual environments like `venv/` and nested `.venv/`
- Python caches like `__pycache__/`, `.pytest_cache/`, and `.benchmarks/`
- local raw datasets such as `raw_data_tdad/`
- temporary smoke-test outputs and helper scripts that should not be committed from `replika_workspace/`

## Pushing

Typical flow:

```bash
git add .
git status --short
git commit -m "Initial commit"
git push -u origin main
```

Review `git status --short` before committing so you can confirm that only the intended reports, code, and result artifacts are staged.
