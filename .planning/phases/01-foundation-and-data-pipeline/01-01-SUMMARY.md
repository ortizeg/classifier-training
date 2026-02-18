---
phase: 01-foundation-and-data-pipeline
plan: "01"
subsystem: infra

tags: [pixi, flit_core, ruff, mypy, pre-commit, pytest, pytorch, lightning, pydantic]

# Dependency graph
requires: []
provides:
  - "pixi environment with conda-forge channel, osx-arm64+linux-64 platforms, Python 3.11"
  - "classifier_training src-layout package installed in editable mode via flit_core"
  - "ruff with full rule set (E/W/F/I/N/UP/B/SIM/S/A/C4/RUF), mypy strict mode"
  - "pre-commit hooks (trailing-whitespace, end-of-file-fixer, check-yaml, large-files, merge-conflict, detect-private-key, ruff, ruff-format)"
  - "pytest configured with tests/ directory; smoke test validates package import"
  - "pixi tasks: test, test-cov, lint, format, format-check, typecheck, precommit"
affects:
  - "all subsequent plans — every plan depends on this importable, lintable, type-checkable package"

# Tech tracking
tech-stack:
  added:
    - "pixi (conda-forge channel, osx-arm64+linux-64)"
    - "python 3.11"
    - "lightning"
    - "loguru"
    - "numpy<2.0.0"
    - "pydantic>=2.0"
    - "pytorch"
    - "torchvision"
    - "pillow"
    - "rich"
    - "pytest + pytest-cov"
    - "ruff"
    - "mypy (strict)"
    - "pre-commit"
    - "flit_core (build backend)"
  patterns:
    - "src-layout: src/classifier_training/ for package code"
    - "flit_core editable install via pixi pypi-dependencies"
    - "dev feature group in pixi.toml for all dev tooling"
    - "environments.default = [dev] ensures dev tools always available"

key-files:
  created:
    - "pixi.toml"
    - ".gitignore"
    - "pyproject.toml"
    - "src/classifier_training/__init__.py"
    - ".pre-commit-config.yaml"
    - "pytest.ini"
    - "tests/__init__.py"
    - "tests/test_package.py"
  modified: []

key-decisions:
  - "Python 3.11 pinned (matching object-detection-training sibling repo)"
  - "flit_core build backend (matching sibling repo convention)"
  - "dev feature always active via environments.default = [dev] — no separate env for tooling"
  - "Added tests/test_package.py smoke test to get pytest exit 0 (pytest 9 exits 5 on no tests)"

patterns-established:
  - "pixi tasks are the only entry point for all tooling (pixi run lint, pixi run test, etc.)"
  - "ruff lint + mypy typecheck must pass clean before every commit"
  - "third-party stub ignores in mypy: torch.*, torchvision.*, lightning.*, pydantic.*, PIL.*, loguru.*, numpy.*, rich.*"

# Metrics
duration: 3min
completed: 2026-02-18
---

# Phase 1 Plan 01: Project Scaffold Summary

**pixi + flit_core src-layout scaffold with ruff strict linting, mypy strict typing, and pre-commit hooks — all four tooling tasks (lint, typecheck, test, import) operational on osx-arm64**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-18T21:57:12Z
- **Completed:** 2026-02-18T22:00:39Z
- **Tasks:** 2
- **Files created:** 8

## Accomplishments

- pixi environment installed successfully (Python 3.11, pytorch, lightning, loguru, pydantic, torchvision, pillow, rich, and all dev tools)
- `classifier_training` package importable in editable mode via flit_core (`pixi run python -c "import classifier_training"` prints `0.0.1`)
- All four tooling tasks verified: `pixi run lint`, `pixi run typecheck`, `pixi run test` all exit 0
- Pre-commit config with 8 hooks covering whitespace, YAML, large files, merge conflicts, secrets, ruff lint+format

## Task Commits

Each task was committed atomically:

1. **Task 1: pixi.toml and .gitignore** - `5c19f4e` (chore)
2. **Task 2: pyproject.toml, src package, pre-commit, pytest config** - `0cd045f` (chore)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `pixi.toml` - Workspace definition: conda-forge channel, osx-arm64+linux-64, all runtime deps, dev feature, pixi tasks, editable install
- `.gitignore` - Python/pixi/model artifact patterns (pycache, .pixi, pixi.lock, *.onnx, *.ckpt, lightning_logs, outputs, etc.)
- `pyproject.toml` - flit_core backend, ruff with E/W/F/I/N/UP/B/SIM/S/A/C4/RUF rules, mypy strict with third-party overrides
- `src/classifier_training/__init__.py` - Package root with `__version__ = "0.0.1"` for flit_core editable install
- `.pre-commit-config.yaml` - trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files (5MB), check-merge-conflict, detect-private-key, ruff, ruff-format
- `pytest.ini` - testpaths = tests, addopts = -v
- `tests/__init__.py` - Empty package marker
- `tests/test_package.py` - Smoke test: import classifier_training, assert __version__ == "0.0.1"

## Decisions Made

- Python 3.11 pinned to match object-detection-training sibling repo conventions
- flit_core as build backend (matching sibling repo)
- `environments.default = ["dev"]` ensures dev tooling always available without separate env activation
- Added `tests/test_package.py` smoke test: pytest 9 returns exit code 5 (not 0) when zero tests are collected — a real import/version smoke test is appropriate and desirable

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added tests/test_package.py smoke test to get pytest exit 0**
- **Found during:** Task 2 (pytest verification)
- **Issue:** pytest 9 exits with code 5 (not 0) when no tests are collected. Plan required `pixi run test` exits 0. Empty `tests/` directory alone doesn't satisfy this.
- **Fix:** Added `tests/__init__.py` (package marker) and `tests/test_package.py` with a single test that imports `classifier_training` and asserts `__version__ == "0.0.1"` — a genuine smoke test, not a workaround.
- **Files modified:** `tests/__init__.py`, `tests/test_package.py`
- **Verification:** `pixi run test` now exits 0 with `1 passed in 0.00s`
- **Committed in:** `0cd045f` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary to satisfy success criteria. Smoke test adds genuine value — validates editable install end-to-end.

## Issues Encountered

None beyond the pytest exit-code deviation above.

## User Setup Required

None - no external service configuration required. pixi handles all dependencies.

## Next Phase Readiness

- Package scaffold complete; all subsequent plans can write source files under `src/classifier_training/`
- All tooling gates (lint, typecheck, test) operational — CI will enforce these from the start
- Ready for Plan 02 (data pipeline: ImageFolder dataset, LightningDataModule, transforms)

---
*Phase: 01-foundation-and-data-pipeline*
*Completed: 2026-02-18*

## Self-Check: PASSED

- All 8 files verified present on disk
- Commits 5c19f4e and 0cd045f verified in git log
- pixi.toml contains `conda-forge`
- pyproject.toml contains `flit_core`
- src/classifier_training/__init__.py contains `__version__`
- .pre-commit-config.yaml contains `ruff-pre-commit`
