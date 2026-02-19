# Phase 5: Infrastructure - Research

**Researched:** 2026-02-19
**Domain:** Docker / GCP Cloud Build / GitHub Actions CI + semantic release / pixi-based tooling
**Confidence:** HIGH (primary reference is the sibling repo — a working implementation of the exact target state)

---

## Summary

Phase 5 ports the CI/CD infrastructure from the sibling `object-detection-training` repo to `classifier-training`. The sibling repo is the authoritative reference: its Dockerfile, `cloudbuild.yaml`, `cloud-build.sh`, three GitHub Actions workflows, and GitHub community files all exist in a working state and need only mechanical name-substitution to apply here.

The key delta between the two repos is package name (`object_detection_training` → `classifier_training`), repo slug (`object-detection-training` → `classifier-training`), and one structural gap in `pixi.toml`: the target repo is missing a `prod` environment (no `[feature.prod.*]` / `[environments] prod = []` stanza) and the `[system-requirements] cuda = "12.1"` block needed by the Dockerfile's `CONDA_OVERRIDE_CUDA` pattern. Both gaps must be filled before the Dockerfile can work.

The semantic release tool is `python-semantic-release@v10.5.3` (already in the sibling workflow). Configuration lives in `[tool.semantic_release]` inside `pyproject.toml`. The pixi GitHub Action is `prefix-dev/setup-pixi@v0.8.1` (used by the sibling) or the current latest `v0.9.4`. Stick with the version the sibling uses unless there is a known reason to upgrade.

**Primary recommendation:** Mechanically port all sibling infrastructure files with name substitutions; add `prod` environment and `system-requirements.cuda` to `pixi.toml`; add `[tool.semantic_release]` block to `pyproject.toml`.

---

## Standard Stack

### Core
| Component | Version / Tag | Purpose | Why Standard |
|-----------|--------------|---------|--------------|
| `nvidia/cuda` base image | `12.1.0-cudnn8-runtime-ubuntu22.04` | GPU runtime for training | INFRA-01 requirement; matches sibling |
| pixi (in Docker) | installed via `curl -fsSL https://pixi.sh/install.sh \| bash` | Env/dep management inside container | pixi.toml already used in project |
| `CONDA_OVERRIDE_CUDA=12.1` | env var | Allows pixi to resolve CUDA pkgs without GPU on build host | Required for Cloud Build machines with no GPU |
| `prefix-dev/setup-pixi` | `v0.8.1` (sibling) / `v0.9.4` (latest) | Install pixi in GitHub Actions | Official pixi CI action |
| `python-semantic-release/python-semantic-release` | `v10.5.3` | Automated versioning + changelog | Used in sibling; latest v10 series |
| `python-semantic-release/publish-action` | `v10.5.3` | Publish GitHub Release artifacts | Paired with semantic release action |
| `pypa/gh-action-pypi-publish` | `release/v1` | Publish to PyPI (if desired) | Used by sibling; OIDC trusted publishing |
| `actions/checkout` | `v4` | Checkout in all workflows | Current standard |
| `codecov/codecov-action` | `v4` | Upload coverage | Used in sibling test workflow |
| GCP Artifact Registry | `us-docker.pkg.dev` | Container registry | Matches sibling `_REGION: us` |
| Cloud Build `E2_HIGHCPU_8` | machine type | Build compute | Matches sibling cloudbuild.yaml |

### Files to Create (complete inventory)
| File | Source Pattern |
|------|---------------|
| `Dockerfile` | Port from sibling; change `object_detection_training` → `classifier_training` |
| `cloudbuild.yaml` | Port from sibling; change `_REPO_NAME` + `_IMAGE_NAME` to `classifier-training` |
| `scripts/cloud-build.sh` | Port from sibling; change `REPO_NAME`/`IMAGE_NAME` vars |
| `scripts/build-docker.sh` | Port from sibling; change `IMAGE_NAME`/`REPO_NAME` vars |
| `.github/workflows/lint.yml` | Port verbatim (no name changes needed) |
| `.github/workflows/test.yml` | Port verbatim |
| `.github/workflows/release.yml` | Port verbatim |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Port verbatim |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Port verbatim |
| `.github/PULL_REQUEST_TEMPLATE.md` | Port verbatim |
| `.github/CODEOWNERS` | Port; update `/src/object_detection_training/conf/` → `/src/classifier_training/conf/` |

### pixi.toml Additions Required
| Addition | What | Why |
|----------|------|-----|
| `[system-requirements]` section | `cuda = "12.1"` | Enables pixi to solve CUDA deps for `prod` env |
| `[environments] prod = []` | Empty prod env (no dev features) | Dockerfile uses `pixi install --environment prod` |

### pyproject.toml Additions Required
| Section | Content | Why |
|---------|---------|-----|
| `[tool.semantic_release]` | `version_toml`, `build_command`, branch config | Required by python-semantic-release action |

---

## Architecture Patterns

### Recommended Project Structure (additions only)
```
classifier-training/
├── Dockerfile
├── cloudbuild.yaml
├── scripts/
│   ├── cloud-build.sh
│   └── build-docker.sh
└── .github/
    ├── CODEOWNERS
    ├── PULL_REQUEST_TEMPLATE.md
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.yml
    │   └── feature_request.yml
    └── workflows/
        ├── lint.yml
        ├── test.yml
        └── release.yml
```

### Pattern 1: Pixi in Docker (direct install, single-stage)

The sibling uses a simple single-stage build — pixi installs into the container and stays there. This is simpler than the multi-stage `shell-hook` pattern and appropriate when:
- The image is used for training (not serving), so size is less critical
- The `pixi run train` entrypoint is the runtime command

**What the pattern looks like:**
```dockerfile
# Source: sibling /object-detection-training/Dockerfile (verified working)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/ortizeg/classifier-training"
LABEL org.opencontainers.image.license="Apache-2.0"

WORKDIR /app

# Install curl and runtime libs needed by ML packages
RUN apt-get update && apt-get install -y curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

ENV PATH="/root/.pixi/bin:$PATH"

# Allow pixi to resolve CUDA packages without a GPU on the build host
ENV CONDA_OVERRIDE_CUDA=12.1

COPY pixi.toml pixi.lock* pyproject.toml ./

# Dummy package structure so flit/pixi can install without src code (cache layer)
RUN mkdir -p src/classifier_training && touch src/classifier_training/__init__.py

# Install only prod environment (no dev/test/lint tooling)
RUN pixi install --environment prod

# Copy real source code
COPY . .

ENTRYPOINT ["pixi", "run", "train"]
```

**Why the dummy `__init__.py` trick:** flit_core's editable install needs the package directory to exist. Creating it empty lets the dependency layer be cached separately from source changes.

### Pattern 2: Pixi in GitHub Actions

```yaml
# Source: sibling .github/workflows/lint.yml (verified working)
- name: Install pixi
  uses: prefix-dev/setup-pixi@v0.8.1
  with:
    pixi-version: latest
    cache: true

- name: Check formatting
  run: pixi run format-check

- name: Lint
  run: pixi run lint

- name: Type check
  run: pixi run typecheck
```

**Key env var:** All three workflows set `CONDA_OVERRIDE_CUDA: "12.1"` at the job level. This is required even in CI (ubuntu-latest runners have no GPU) so pixi can resolve pytorch/cuda packages from the lock file.

### Pattern 3: Semantic Release Configuration

In `pyproject.toml` (append to existing file):
```toml
# Source: sibling pyproject.toml [tool.semantic_release] section
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "pip install build && python -m build"

[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

The `develop` branch produces pre-releases with `dev` suffix (e.g. `1.2.0.dev1`). The `main` branch produces stable releases.

### Pattern 4: GCP Cloud Build Submission

```bash
# Source: sibling scripts/cloud-build.sh
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions=SHORT_SHA=${SHORT_SHA} \
    --project=${PROJECT_ID} \
    .
```

`SHORT_SHA` is computed locally via `git rev-parse --short HEAD` because `gcloud builds submit` (manual trigger) does not auto-populate Cloud Build substitution variables the way a push trigger does.

### Pattern 5: pixi.toml `prod` Environment

```toml
# Add to classifier-training/pixi.toml

[system-requirements]
cuda = "12.1"

[environments]
default = ["dev"]
prod = []          # Empty: only base [dependencies], no dev feature
```

The `prod = []` syntax means: no additional features, just the base `[dependencies]` block. This produces a minimal runtime environment without pytest, ruff, mypy, pre-commit.

### Anti-Patterns to Avoid
- **Omitting `CONDA_OVERRIDE_CUDA`:** Without it, pixi fails to resolve pytorch/cuda packages on a build host with no GPU (both Docker and CI).
- **`pixi install` without `--environment prod` in Dockerfile:** Installs dev env including all lint/test tooling; bloats the image significantly.
- **Omitting `pixi.lock*` in Dockerfile COPY:** The `*` makes it optional for first-time builds; always include the lock file when it exists for reproducibility.
- **Missing `fetch-depth: 0` in release workflow:** python-semantic-release needs full git history for commit analysis. The sibling uses `fetch-depth: 0` explicitly (note: PSR v10.5.0+ can auto-convert shallow clones, but the sibling's explicit approach is safer and portable).
- **Not setting `concurrency: release` in release job:** Multiple pushes can trigger concurrent releases; the `concurrency` key prevents duplicate/conflicting releases.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Version bumping | Custom version bump scripts | `python-semantic-release@v10.5.3` | Handles CHANGELOG, git tags, GitHub releases, PyPI publish atomically |
| Docker layer caching in Cloud Build | Custom cache scripts | `--cache-from` with `:latest` tag in `cloudbuild.yaml` | Standard Cloud Build pattern; already in sibling |
| pixi caching in CI | Custom cache actions | `prefix-dev/setup-pixi` with `cache: true` | Uses `pixi.lock` hash as cache key automatically |
| GitHub Release creation | Manual `gh release create` | `python-semantic-release/publish-action@v10.5.3` | Tied to semantic release output, handles assets |
| Coverage reporting | Custom coverage upload | `codecov/codecov-action@v4` | Reads `coverage.xml` that `pixi run test-cov` already produces |

**Key insight:** Every piece of this infrastructure has a turnkey solution already validated in the sibling repo. The work is port-and-rename, not design.

---

## Common Pitfalls

### Pitfall 1: Missing `prod` Environment in pixi.toml
**What goes wrong:** `docker build` fails at `RUN pixi install --environment prod` with "environment 'prod' not found".
**Why it happens:** The `classifier-training` pixi.toml currently only defines `default = ["dev"]` with no `prod` entry.
**How to avoid:** Add `prod = []` under `[environments]` AND add `[system-requirements] cuda = "12.1"` before writing the Dockerfile.
**Warning signs:** pixi error mentioning unknown environment name during Docker build.

### Pitfall 2: Missing `system-requirements.cuda` in pixi.toml
**What goes wrong:** `pixi install --environment prod` succeeds on macOS dev machine (which has the lock file) but produces a lock file that cannot resolve CUDA-enabled pytorch on linux-64 target.
**Why it happens:** pixi needs `cuda = "12.1"` in `[system-requirements]` to know it can select CUDA-enabled package variants for linux-64.
**How to avoid:** Add the `[system-requirements]` block before running `pixi install` to regenerate the lock file.
**Warning signs:** pytorch in lock file resolves to CPU-only variant on linux-64.

### Pitfall 3: `CONDA_OVERRIDE_CUDA` Not Set in CI Job Environment
**What goes wrong:** GitHub Actions job fails when pixi tries to install the default environment (which includes pytorch with CUDA dependency) on a runner with no GPU.
**Why it happens:** Pixi checks for actual CUDA virtual packages; without the override, it sees no CUDA and cannot satisfy the requirement.
**How to avoid:** Add `env: CONDA_OVERRIDE_CUDA: "12.1"` at the job level in all three workflow files.
**Warning signs:** pixi install error about unsatisfied CUDA system requirement in CI logs.

### Pitfall 4: Shallow Clone Breaks Semantic Release Commit Analysis
**What goes wrong:** `python-semantic-release` sees no commits to analyze; no release is created even after a conventional commit is merged.
**Why it happens:** `actions/checkout@v4` defaults to `fetch-depth: 1` (shallow clone).
**How to avoid:** Use `fetch-depth: 0` in the release job's checkout step (as the sibling does), then `git reset --hard ${{ github.sha }}` to ensure the HEAD matches the triggering commit.
**Warning signs:** Semantic release reports "no commits since last release" when there clearly are some.

### Pitfall 5: Branch Protection Blocks Semantic Release Push
**What goes wrong:** The semantic release workflow creates a version bump commit but cannot push it to `main` because branch protection requires PR reviews.
**Why it happens:** Branch protection `required_pull_request_reviews` applies to the bot's push too.
**How to avoid:** In branch protection configuration, allow the GitHub Actions bot (`github-actions[bot]`) to bypass PR requirements, OR configure semantic release to use a PAT with admin access. The sibling's approach uses `GITHUB_TOKEN` with `contents: write` permission — this works when the repository's branch protection does not enforce required reviews for admins/actions.
**Warning signs:** Workflow fails at the PSR push step with 403 Protected Branch error.

### Pitfall 6: Cloud Build SHORT_SHA Mismatch on Manual Submission
**What goes wrong:** Cloud Build image tags use a different SHA than the one computed locally in `cloud-build.sh`.
**Why it happens:** Cloud Build manual submissions do not auto-populate `$SHORT_SHA`; it must be passed via `--substitutions`.
**How to avoid:** Always use the `cloud-build.sh` script (not raw `gcloud builds submit`) — it computes `SHORT_SHA=$(git rev-parse --short HEAD)` and passes it explicitly.
**Warning signs:** Image tagged `:latest` but `:${SHORT_SHA}` tag is missing or wrong.

---

## Code Examples

### cloudbuild.yaml (classifier-training version)
```yaml
# Source: port of sibling /object-detection-training/cloudbuild.yaml
substitutions:
  _REGION: us
  _REPO_NAME: classifier-training
  _IMAGE_NAME: classifier-training

steps:
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args: ['-c', 'docker pull ${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPO_NAME}/${_IMAGE_NAME}:latest || true']

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--cache-from'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPO_NAME}/${_IMAGE_NAME}:latest'
      - '-t'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPO_NAME}/${_IMAGE_NAME}:${SHORT_SHA}'
      - '-t'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPO_NAME}/${_IMAGE_NAME}:latest'
      - '.'

images:
  - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPO_NAME}/${_IMAGE_NAME}:${SHORT_SHA}'
  - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPO_NAME}/${_IMAGE_NAME}:latest'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'

timeout: '1800s'
```

### GitHub Actions: lint.yml
```yaml
# Source: port of sibling .github/workflows/lint.yml
name: Lint & Format

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    env:
      CONDA_OVERRIDE_CUDA: "12.1"
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true

      - name: Check formatting
        run: pixi run format-check

      - name: Lint
        run: pixi run lint

      - name: Type check
        run: pixi run typecheck
```

### GitHub Actions: test.yml
```yaml
# Source: port of sibling .github/workflows/test.yml
name: Test Suite

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      CONDA_OVERRIDE_CUDA: "12.1"
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true

      - name: Run tests with coverage
        run: pixi run test-cov

      - name: Upload coverage
        if: always()
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

### GitHub Actions: release.yml (key section)
```yaml
# Source: port of sibling .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main, develop]

jobs:
  check:
    runs-on: ubuntu-latest
    env:
      CONDA_OVERRIDE_CUDA: "12.1"
    steps:
      - uses: actions/checkout@v4
      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true
      - name: Lint
        run: pixi run lint
      - name: Type check
        run: pixi run typecheck
      - name: Test
        run: pixi run test

  release:
    name: Semantic Release
    runs-on: ubuntu-latest
    concurrency: release
    environment: pypi
    needs: [check]
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop')
    permissions:
      id-token: write
      contents: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          ref: ${{ github.ref_name }}
          fetch-depth: 0

      - name: Reset to Workflow SHA
        run: git reset --hard ${{ github.sha }}

      - name: Python Semantic Release
        id: release
        uses: python-semantic-release/python-semantic-release@v10.5.3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}

      - name: Publish to PyPI
        if: steps.release.outputs.released == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}

      - name: Publish to GitHub Releases
        if: steps.release.outputs.released == 'true'
        uses: python-semantic-release/publish-action@v10.5.3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          tag: ${{ steps.release.outputs.tag }}
```

### CODEOWNERS (classifier-training version)
```
# Source: port of sibling .github/CODEOWNERS
* @ortizeg
/src/ @ortizeg
/src/classifier_training/conf/ @ortizeg
pixi.toml @ortizeg
pyproject.toml @ortizeg
/.github/ @ortizeg
/tests/ @ortizeg
```

### pixi.toml additions
```toml
# Add to existing pixi.toml

[system-requirements]
cuda = "12.1"

# Replace the existing [environments] block:
[environments]
default = ["dev"]
prod = []
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `setup-pixi@v0.8.1` (sibling uses) | `v0.9.4` (latest as of 2026-02) | v0.9.x adds Node 24, working directory setup; either works — match sibling for parity |
| `python-semantic-release` pre-v10 required `fetch-depth: 0` | v10.5.0+ auto-converts shallow clones | Sibling already uses v10.5.3 and explicit `fetch-depth: 0`; keep explicit for clarity |
| Multi-stage Docker (shell-hook pattern) | Single-stage with `pixi install --environment prod` | Single-stage is simpler for training images; multi-stage saves ~100MB for serving images |
| `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` | Newer tags available (12.9.1, 13.0.0) | INFRA-01 specifies 12.1 — do not change |

---

## Open Questions

1. **PyPI publish in release workflow**
   - What we know: The sibling publishes to PyPI via `pypa/gh-action-pypi-publish` with `PYPI_TOKEN` secret.
   - What's unclear: Does the user want `classifier-training` published to PyPI, or only GitHub Releases?
   - Recommendation: Port the full sibling workflow including PyPI publish; if not desired, the `if: steps.release.outputs.released == 'true'` step is simply omitted. Plan it as an optional step.

2. **GitHub Environments secret (`pypi` environment)**
   - What we know: The release job uses `environment: pypi` which requires a GitHub Environments configuration and `PYPI_TOKEN` secret.
   - What's unclear: Whether the `classifier-training` GitHub repo already has the `pypi` environment configured.
   - Recommendation: Document the setup requirement in the task; it is a one-time manual GitHub UI step.

3. **Branch protection configuration method**
   - What we know: INFRA-08 requires branch protection on `main` and `develop`. GitHub CLI (`gh api`) can configure this via REST API.
   - What's unclear: Whether to automate via script or document as a manual step.
   - Recommendation: Document as a manual GitHub UI step with the specific settings to enable (require PR reviews, require status checks for lint+test, do not restrict pushes for Actions bot). Low automation ROI for one-time setup.

4. **Gitflow branch `develop` existence**
   - What we know: INFRA-07 requires gitflow with `main` + `develop`. The current repo may only have `main`.
   - What's unclear: Whether `develop` branch exists; if not, it must be created before branch protection can be configured.
   - Recommendation: Create `develop` as part of INFRA-07 task; branch off current `main`.

---

## Sources

### Primary (HIGH confidence)
- **Sibling repo Dockerfile** — `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/Dockerfile` — verified working pattern, all Docker layer details
- **Sibling repo cloudbuild.yaml** — verified GCP Cloud Build configuration
- **Sibling repo scripts/cloud-build.sh + build-docker.sh** — verified invocation scripts
- **Sibling repo .github/workflows/lint.yml, test.yml, release.yml** — verified workflow YAML
- **Sibling repo pyproject.toml** `[tool.semantic_release]` block — verified semantic release config
- **Sibling repo pixi.toml** `[environments]` + `[system-requirements]` — verified pixi env structure
- **classifier-training pyproject.toml + pixi.toml** — confirmed current state, identified gaps

### Secondary (MEDIUM confidence)
- [setup-pixi GitHub Releases](https://github.com/prefix-dev/setup-pixi/releases) — confirmed v0.9.4 is latest as of 2026-02
- [python-semantic-release GitHub Actions docs](https://python-semantic-release.readthedocs.io/en/latest/configuration/automatic-releases/github-actions.html) — confirmed v10.5.3 workflow YAML
- [pixi GitHub Actions docs](https://pixi.prefix.dev/latest/integration/ci/github_actions/) — confirmed caching behavior and `CONDA_OVERRIDE_CUDA` pattern
- [nvidia/cuda Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) — confirmed `12.1.0-cudnn8-runtime-ubuntu22.04` tag still available

### Tertiary (LOW confidence)
- WebSearch results for multi-stage pixi Docker patterns — not used (sibling single-stage approach is verified and sufficient)

---

## Metadata

**Confidence breakdown:**
- Dockerfile pattern: HIGH — direct port of working sibling code
- cloudbuild.yaml + scripts: HIGH — direct port of working sibling code
- GitHub Actions workflows: HIGH — direct port of working sibling code
- semantic release config: HIGH — sibling pyproject.toml + readthedocs verified
- pixi.toml gaps (`prod` env, `system-requirements`): HIGH — structure verified in sibling
- setup-pixi action version: MEDIUM — v0.8.1 (sibling) vs v0.9.4 (latest); either works
- Branch protection configuration: MEDIUM — REST API approach confirmed, specific settings may need validation

**Research date:** 2026-02-19
**Valid until:** 2026-03-21 (30 days — stable infrastructure tooling)
