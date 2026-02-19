---
phase: 05-infrastructure
plan: 01
subsystem: infra
tags: [docker, cuda, gcp, cloud-build, artifact-registry, pixi, semantic-release]

# Dependency graph
requires:
  - phase: 04-training-configuration
    provides: pixi.toml, pyproject.toml with full classifier_training package installed

provides:
  - Dockerfile with CUDA 12.1 + pixi single-stage image for classifier_training
  - cloudbuild.yaml GCP Cloud Build pipeline for Artifact Registry push
  - scripts/cloud-build.sh for manual Cloud Build submissions with dry-run support
  - scripts/build-docker.sh for local builds and Artifact Registry pushes
  - pixi.toml prod environment and system-requirements.cuda=12.1
  - pyproject.toml semantic_release config for version bumps

affects: [phase-05-infrastructure, cloud-training, gpu-training]

# Tech tracking
tech-stack:
  added: [nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04, GCP Cloud Build, Artifact Registry, python-semantic-release]
  patterns: [pixi prod environment for Docker, CONDA_OVERRIDE_CUDA for GPU-less builds, SHORT_SHA tagging for image versioning]

key-files:
  created:
    - Dockerfile
    - cloudbuild.yaml
    - scripts/cloud-build.sh
    - scripts/build-docker.sh
  modified:
    - pixi.toml
    - pyproject.toml

key-decisions:
  - "Single-stage Docker image: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 base with pixi installed via curl"
  - "CONDA_OVERRIDE_CUDA=12.1 env var allows pixi to solve CUDA packages without attached GPU during build"
  - "Dummy src/classifier_training/__init__.py created before pixi install to cache dependency layer separately from source code"
  - "pixi.lock unchanged on macOS because CUDA system-requirements only affect linux-64 solver -- lockfile correct for both platforms"
  - "prod = [] environment in pixi.toml installs only base [dependencies], not [feature.dev.dependencies]"
  - "ENTRYPOINT uses pixi run train -- secrets injected via env vars at runtime, not baked into image"

patterns-established:
  - "Build scripts: cloud-build.sh handles --dry-run and uncommitted-changes guard; build-docker.sh handles --local vs push"
  - "Tagging: SHORT_SHA + latest tags for every image push to Artifact Registry"
  - "pixi tasks: build = cloud-build.sh, build-local = build-docker.sh --local for developer ergonomics"

# Metrics
duration: 2min
completed: 2026-02-19
---

# Phase 5 Plan 01: Docker and Cloud Build Infrastructure Summary

**CUDA 12.1 Docker image + GCP Cloud Build pipeline ported from object-detection-training sibling repo with classifier_training substitutions, pixi prod environment, and semantic_release config**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-19T06:32:06Z
- **Completed:** 2026-02-19T06:34:23Z
- **Tasks:** 2
- **Files modified:** 6 (2 modified, 4 created)

## Accomplishments

- Dockerfile builds a single-stage CUDA 12.1 + pixi image, installs only prod dependencies via `pixi install --environment prod`, and sets ENTRYPOINT to `pixi run train`
- cloudbuild.yaml defines a GCP Cloud Build pipeline with layer caching, SHA + latest tagging, E2_HIGHCPU_8 machine, and 30-min timeout
- cloud-build.sh and build-docker.sh shell scripts handle Cloud Build submission and local/registry Docker builds respectively, both referencing classifier-training names
- pixi.toml updated with prod environment, CUDA 12.1 system requirement, and build/build-local task shortcuts
- pyproject.toml updated with semantic_release config for automated version bumps on main/develop branches

## Task Commits

Each task was committed atomically:

1. **Task 1: Update pixi.toml and pyproject.toml for infrastructure** - `cdd0df0` (chore)
2. **Task 2: Create Dockerfile and build scripts** - `f474814` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `Dockerfile` - CUDA 12.1 + pixi single-stage training image for classifier_training
- `cloudbuild.yaml` - GCP Cloud Build pipeline: pull cache, build with SHA+latest tags, push to Artifact Registry
- `scripts/cloud-build.sh` - Cloud Build submission script with --dry-run flag and uncommitted-changes guard
- `scripts/build-docker.sh` - Local Docker build script with --local flag and Artifact Registry push path
- `pixi.toml` - Added [system-requirements] cuda="12.1", prod=[] env, build/build-local tasks
- `pyproject.toml` - Added [tool.semantic_release] config for main/develop branches

## Decisions Made

- **CONDA_OVERRIDE_CUDA=12.1** set as Docker ENV so pixi can solve CUDA-dependent packages without a GPU attached to the build machine.
- **pixi.lock unchanged on macOS** because CUDA system-requirements only affect the linux-64 solver; both platforms are covered correctly. The lockfile will diverge on linux-64 only after a `pixi install` there.
- **Dummy __init__.py pattern** caches the dependency install layer independently from source code — rebuilds on source-only changes skip the slow `pixi install --environment prod` step.
- **Ported from sibling repo** object-detection-training with name substitutions only (object_detection_training -> classifier_training, object-detection-training -> classifier-training).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required at this stage. When running cloud-build.sh or build-docker.sh (push mode), GCP project must be configured via `gcloud config set project <PROJECT_ID>`.

## Next Phase Readiness

- Docker infrastructure complete; image can be built locally with `pixi run build-local` once Docker daemon is available
- Cloud Build pipeline ready for `pixi run build` once GCP project is configured
- Phase 5 Plan 02 (cloud training jobs or GitHub Actions) can proceed

---
*Phase: 05-infrastructure*
*Completed: 2026-02-19*

## Self-Check: PASSED

- Dockerfile: FOUND
- cloudbuild.yaml: FOUND
- scripts/cloud-build.sh: FOUND
- scripts/build-docker.sh: FOUND
- 05-01-SUMMARY.md: FOUND
- Commit cdd0df0 (Task 1): FOUND
- Commit f474814 (Task 2): FOUND
