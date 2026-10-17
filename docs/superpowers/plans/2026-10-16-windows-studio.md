# Windows 8D Engine Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native execution or superpowers:subagent-driven-development if the user chooses delegated execution. Steps use checkbox syntax for tracking.

**Goal:** Deliver a local Windows studio that imports audio, saves projects, auditions processed excerpts, and reliably exports 8D renders.

**Architecture:** Build a separate FastAPI application around the existing streaming DSP. SQLite owns durable project/job state; a single worker renders immutable job settings. Serve an offline-capable HTML/CSS/JavaScript interface from the same local application.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, SQLite standard library, NumPy, SciPy, SoundFile, FFmpeg, vanilla browser JavaScript, pytest/httpx.

**Spec:** `docs/superpowers/specs/2026-10-16-windows-studio-design.md` (approved by user's “next”).

## Global Constraints

- Bind to `127.0.0.1` only.
- Store application data under `%LOCALAPPDATA%/8DEngineStudio`.
- Enforce an initial 200 MB upload limit and 60-minute decoded duration limit, with clear feedback.
- Bound queued jobs to 10 and reject excess submissions before starting work.
- Never overwrite source audio.
- One active worker initially.
- WAV and FLAC exports; MP3 when FFmpeg is available.
- Queued jobs remain queued for explicit restart.
- No production merge or deployment; use `rebuild/windows-studio`.
- First release excludes stem separation, cloud services, social accounts, payments and measured-HRTF claims.

## Review Focus

1. Unicode Windows paths, duplicate filenames and spaces must preserve each imported source independently (Task 2).
2. Empty, corrupt, NaN-bearing, silent and overlength audio must fail clearly or produce finite output as appropriate (Tasks 2–3).
3. Disk-full, decoder failure and process interruption must never produce a completed job pointing to partial output (Tasks 3–4).
4. Cancelling/retrying while editing settings must not mutate an existing render snapshot or create competing active jobs (Task 4).
5. Browser refresh, stale previews and missing optional codecs must expose correct current state rather than misleading playback/export success (Tasks 5–6).

## File ownership

`studio/config.py`: paths and runtime configuration. `studio/models.py`: validated settings and typed records. `studio/store.py`: database operations. `studio/assets.py`: imports, metadata and waveform peaks. `studio/renderer.py`: DSP adapter and output measurements. `studio/worker.py`: scheduling and cancellation. `studio/app.py`: API and local request guards. `studio/static/{index.html,studio.css,studio.js}`: interface. `studio/launch.py`: application startup. `requirements-studio.txt`, `requirements-studio-dev.txt`, `SETUP_STUDIO_WINDOWS.cmd`, `START_STUDIO_WINDOWS.cmd`: reproducible setup. Tests live in `tests/studio/`; release guide in `docs/windows-studio.md`.

### Task 1: Persistent projects and validated settings

**Files:** Create `studio/__init__.py`, `studio/config.py`, `studio/models.py`, `studio/store.py`, requirement files and `tests/studio/test_store.py`.

**Interfaces:** `StudioConfig(data_dir: Path)`; `MixSettings(rotation_cpm=5.78, room_size=0.18, crossover_hz=150, motion_depth=1, panning_preset='binaural_8d')`; `Store(db_path: Path)` with `create_project(title: str, asset: dict) -> dict`, `get_project(project_id: str) -> dict`, `list_projects() -> list[dict]`, `update_settings(project_id: str, settings: MixSettings) -> dict`.

- [ ] Record the existing DSP baseline with `python -m pytest tests/test_dsp.py tests/test_streaming_render.py tests/test_premium_quality_pipeline.py -q`; record missing dependencies separately from test failures.
- [ ] Add `test_project_survives_reopen` asserting saved title/source/settings survive closing and reopening SQLite; `test_settings_reject_nonfinite_and_out_of_range` asserting invalid values raise validation errors.
- [ ] Run `python -m pytest tests/studio/test_store.py -q`; expect missing-module failure before implementation.
- [ ] Implement schema version 1, UUID identifiers, parameterized SQL, per-operation connections and transactions. Validate finite rotation 0.1–30 CPM, room 0–0.5, crossover 40–300 Hz, motion depth 0–1; allow only existing preset names. Store timestamps in UTC and paths relative to data root.
- [ ] Install the minimal studio dependencies, run the store tests, and commit the passing unit. Keep heavy Demucs/social dependencies out of the studio requirements.

### Task 2: Managed audio import and waveform assets

**Files:** Create `studio/assets.py`, `studio/app.py`, `tests/studio/test_assets.py`, `tests/studio/test_api.py`.

**Interfaces:** `import_audio(stream: BinaryIO, filename: str, config: StudioConfig) -> dict`; `waveform_peaks(path: Path, bins: int = 1200) -> list[list[float]]`; `create_app(config: StudioConfig) -> FastAPI`. Asset record: ID, relative path, original name, sample rate, frames, duration and channel count.

- [ ] Add import tests for duplicate Unicode names, empty/corrupt files and byte/duration overflow; assert independent asset IDs, source preservation and no abandoned partial file. Test overflow with small injected test limits, plus assert production limits equal `200 * 1024 * 1024` and `3600`.
- [ ] Add API tests asserting an untrusted Host/Origin or missing modification token is rejected and arbitrary path traversal cannot retrieve files; run these tests and confirm missing implementation failures.
- [ ] Implement chunked writes, metadata inspection through existing decoding helpers, UUID storage filenames and bounded waveform reduction. Inspect duration before waveform work; reject invalid sample rates and zero frames. Clean temporary/failed imports.
- [ ] Implement same-origin routes: `GET /api/health`, `GET /api/projects`, `POST /api/projects` (multipart), `GET /api/projects/{id}`, `PATCH /api/projects/{id}/settings`, `GET /api/assets/{id}`, `GET /api/projects/{id}/waveform`. Stream file responses with media types supporting browser playback.
- [ ] Implement exact allowed loopback Host/Origin checks and a random per-launch request token embedded in the locally served page. All mutations require the token; deny foreign Origin, including null. Run import/API tests and commit.

### Task 3: Shared preview/final rendering adapter

**Files:** Create `studio/renderer.py`, `tests/studio/test_renderer.py`; modify DSP only if needed for cleanup/cancellation hooks, preserving existing signatures.

**Interfaces:** `RenderRequest(source: Path, output: Path, settings: MixSettings, format: str, start_seconds: float | None = None, duration_seconds: float | None = None)`; `render(request: RenderRequest, progress: Callable[[float], None], cancelled: Callable[[], bool]) -> dict`. Result contains output metadata, sample peak, correlation and optional measured loudness.

- [ ] Add synthetic stereo tests asserting finite output, equal sample rate, expected frame count and peak at most 1; compare preview and final settings mapping. Add silence, mono, corrupt source, cancellation and forced write-failure tests. Run to establish failure before implementation.
- [ ] Adapt `render_8d_file_to_wav` using explicit DSP parameters and `preserve_quality=True`. Preview defaults to 15 seconds, clamped to remaining duration; validate selected start. Extract the excerpt to a temporary WAV and process with the same settings mapping. Document that excerpt phase/mastering may differ from a full render because the excerpt starts its own render timeline.
- [ ] Render into a per-job temporary directory and atomically finalize output. Convert compressed output through argument-array FFmpeg subprocess calls, with timeouts. Remove temporary WAV/rawtmp files in `finally`; never mark a cancelled operation successful.
- [ ] Measure final exported audio in bounded blocks. Use the existing optional loudness measurement helper, returning null when unavailable. Check cancellation before decoding, at progress boundaries, before encoding and before finalization; document non-interruptible stages honestly.
- [ ] Run renderer and existing streaming/DSP tests, record runtime and peak memory for a reproducible synthetic fixture, and commit.

### Task 4: Durable render queue and recovery

**Files:** Create `studio/worker.py`, `tests/studio/test_jobs.py`; extend `studio/store.py` and `studio/app.py`.

**Interfaces:** Store methods `enqueue(project_id: str, kind: str, settings: MixSettings, options: dict) -> dict`, `get_job(job_id: str) -> dict`, `list_jobs(project_id: str | None = None) -> list[dict]`, `transition(job_id: str, expected: str, target: str, **updates) -> bool`; `Worker(store: Store, config: StudioConfig)` with `start()`, `stop()`, `cancel(job_id: str)`, `resume_queue()`.

- [ ] Add tests for 10 queued jobs, rejection of the eleventh, one active renderer, snapshot isolation after project edits, queued/running cancellation, failed retry and startup recovery. Assert running → interrupted after reopen, queued jobs remain paused until explicit resume, and incomplete outputs never become downloadable.
- [ ] Run `python -m pytest tests/studio/test_jobs.py -q` and confirm failures before implementing transactional state transitions and single-worker ownership. Use an injected renderer stub for deterministic concurrency tests.
- [ ] Implement terminal states completed/failed/cancelled/interrupted; active cancellation records a request until the renderer acknowledges it. Retry creates a new job from the prior immutable snapshot. Job deletion is not needed for this release.
- [ ] Add `POST /api/projects/{id}/renders`, `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/retry`, `POST /api/queue/resume`, `GET /api/jobs/{id}/download`. Final/preview jobs share the queue; downloads require completed state and validated asset paths.
- [ ] Run queue and API tests, including real adapter smoke render, and commit.

### Task 5: Usable browser studio

**Files:** Create `studio/static/index.html`, `studio/static/studio.css`, `studio/static/studio.js`; extend static route in `studio/app.py`.

**Interfaces:** Consume Task 2/4 routes, send `X-Studio-Token` on mutations, and use project/job IDs as the sole asset lookup keys. No external libraries/CDNs required.

- [ ] Implement the dark responsive layout with project sidebar, waveform/transport center, labeled settings panel and render queue. Include numeric inputs, preset/reset actions, drag/drop plus standard file picker and format selection.
- [ ] Implement project restore/save, bounded polling while jobs are active, explicit queue resume after restart, and actionable API errors. Disable export until completed. Preserve original upload names in display using text content, never injected HTML.
- [ ] Implement 15-second excerpt selection and original/processed A/B playback. Use decoded browser audio or a compatible local audition WAV; compressed-format playback failure must show a usable message. Maintain excerpt-relative playhead on switching, pause the prior source, and require user initiation. Invalidate previews whenever settings change; do not label stale output as current.
- [ ] Verify in a browser: import a fixture, change controls, preview, switch A/B, render/export, refresh and reopen the saved project. Exercise missing codec and failed-job displays, keyboard focus, small viewport and cancellation. Record observed results; if browser tooling is unavailable, explicitly record the verification gap.
- [ ] Run the integration suite and commit the working interface.

### Task 6: Windows startup and release handoff

**Files:** Create `studio/launch.py`, `SETUP_STUDIO_WINDOWS.cmd`, `START_STUDIO_WINDOWS.cmd`, `docs/windows-studio.md`, `tests/studio/test_launch.py`; update README with the separate Studio entrypoint.

**Interfaces:** `python -m studio.launch` runs loopback port 8765; `EIGHTD_STUDIO_DATA_DIR` overrides storage for tests/development. Launcher probes readiness before opening a browser; if the port is occupied by an unrelated service, it reports the conflict and exits.

- [ ] Add readiness/port-conflict tests and ensure the launcher never opens a failed service; run and confirm pre-implementation failure.
- [ ] Implement Python 3.11 setup detection, quoted Windows paths, virtual-environment setup and clear exit-code handling. Setup pins a tested dependency set; start reports missing setup without silently performing installs. Keep errors visible with `pause` in command scripts.
- [ ] Document first install, subsequent start, data location, MP3/FFmpeg requirements, preview versus full render, optional future stems, recovery, logs and export retrieval. Distinguish tested host behavior from actual Windows acceptance.
- [ ] Run `python -m pytest tests/studio tests/test_dsp.py tests/test_streaming_render.py tests/test_premium_quality_pipeline.py -q` plus relevant existing audio tests. Run an end-to-end synthetic import/preview/render/export/restart check and report actual results.
- [ ] Review changes against the spec, inspect git diff for unintended production changes, commit, and push the separate branch with a draft PR if GitHub write access succeeds. Report blockers truthfully if it does not.
- [ ] Deliver exact Windows launcher instructions and note that Windows installation and real-track listening still require Windows verification if this environment cannot perform them.

## Self-review and execution handoff

All first-release spec requirements map to Tasks 1–6. Stem mixing and new DSP algorithms are explicitly later milestones, not missing first-release deliverables. Preview is processed rather than instantaneous; restart means recoverable job records, not resuming midway through samples. A/B listening and Windows launch acceptance remain distinct from automated DSP correctness.

Recommended execution: native implementation in this session because the six tasks share tightly coupled store/renderer/API interfaces. User may instead choose subagent-driven execution. Implementation begins after review of this plan and selection of execution method.
