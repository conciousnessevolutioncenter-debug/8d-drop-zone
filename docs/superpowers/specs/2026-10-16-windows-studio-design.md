# 8D Engine Studio — Windows-first rebuild

Status: proposed design for review; implementation has not started.

## Purpose and agreed direction

Build Christ's personal 8D audio production studio on Windows first. The user confirmed personal production before a customer platform, and Windows before online hosting. Preserve the original repository while developing on `rebuild/windows-studio`.

Success means an understandable import → audition → adjust → render → export workflow, reproducible settings, persistent projects, and verified audio output. “A thousand times better” expresses ambition, not a measured performance claim.

## Approach

Recommended: a local browser interface served by a separate FastAPI studio application, reusing the existing Python audio engine through a small renderer adapter. SQLite stores projects and render jobs. This allows a richer interface while retaining the working DSP investment.

Alternatives: extending Tkinter would require less infrastructure but constrain the waveform and audition experience; an Electron rewrite would add packaging and runtime complexity without improving the audio itself. Neither is selected.

## First release

1. Local studio with project list, drag-and-drop audio import, waveform overview, playback, preset selection, and controls for rotation speed, spatial depth, room amount, and bass crossover where supported by the engine.
2. Save project settings and restore them on reopening. Each render records an immutable settings snapshot so later edits do not change its meaning.
3. Render a selected short excerpt for before/after audition using the same DSP path as the final export. Preview requires processing; it is not an instant real-time DSP promise. Switching A/B retains the excerpt playback position where browser playback permits.
4. Durable render history with queued, running, completed, failed, cancelled, and interrupted states. One active worker initially. On startup, previously running jobs become interrupted and can be retried from the beginning; no claim of sample-level resume. Queued jobs remain queued for explicit restart.
5. WAV and FLAC exports; MP3 when FFmpeg is available. Completed exports display sample rate, duration, peak and stereo correlation. Optional loudness measurements are clearly unavailable if unsupported, never invented.
6. Windows setup and start scripts with readable diagnostics. Setup creates a virtual environment and installs the core dependency set. Start opens the browser only after the local health endpoint responds. Missing Python, missing codecs, occupied ports, and installation failures produce actionable messages and keep the console visible.

## Scope sequencing

The first milestone delivers the full import-to-export workflow above. A second milestone adds local stem separation and per-stem mix/render controls after the core workflow passes verification. Existing stem processing is retained, but Demucs is an optional installation because of its large dependencies and hardware cost. Imported stems should be supported before requiring automatic separation.

Later work: measured-HRTF convolution after choosing a redistributable dataset, improved spatial algorithms evaluated through listening comparisons, standalone installer, and optional customer hosting. Social feeds, payments, cloud AI, and accounts are outside the local studio's first release. The existing hosted app remains independently runnable.

## Architecture and data

- `studio/app.py`: application setup and local API routes; no import of the existing social/billing application.
- `studio/store.py`: SQLite schema, project records, immutable render settings, and transactional job transitions.
- `studio/worker.py`: bounded single-worker scheduling, progress, interruption handling, cancellation checks, and output finalization.
- `studio/renderer.py`: adapter to existing decoding, streaming DSP, and format conversion; shared preview/final settings validation.
- `studio/static/`: separate HTML, CSS and JavaScript for accessible studio controls and playback; no remote CDN dependency.
- Windows launcher scripts and a separate minimal requirements file.

Store application data under `%LOCALAPPDATA%/8DEngineStudio`, including SQLite, managed copies of imported sources, project assets, previews, and renders. Use a configurable data directory on non-Windows systems for development and tests. Never overwrite source audio. Write outputs to temporary names and atomically finalize only successful renders.

Each project has an ID, title, source asset metadata, current settings and timestamps. Each job has an ID, project ID, settings snapshot, render kind, state, progress, timestamps, relative output path and sanitized error message. Resolve file downloads by asset ID within the data directory, never an arbitrary client-supplied path.

## Local access and resource limits

Bind to `127.0.0.1` only. Validate Host and browser Origin and require a per-launch token for modifying requests. Do not expose the source directory through a public static mount. Do not fetch remote audio URLs in the first local release.

Enforce an initial 200 MB upload limit and 60-minute decoded duration limit, with clear feedback. Bound queued jobs to 10 and reject excess submissions before starting work. Decode/render off the request loop. Cancellation is checked at supported rendering boundaries; the UI must distinguish a cancellation request from a worker that has actually stopped. Remove partial outputs on failure/cancellation. Retain originals and successful exports until the user deletes them.

## Sound quality and claims

Retain the existing renderer as the baseline. The current technique is HRTF-inspired panning/delay/filtering, not personalized measured HRTF. Do not rename existing processing as a newly proven audio improvement. Preview and final rendering must preserve source sample rate and channels normalized to stereo. Prevent non-finite output and overload, preserve duration, and report the actual output measurements. Bass-centering and spatial behavior are verified with signal tests.

Evaluate future DSP changes using level-matched listening excerpts chosen by Christ, alongside peak, correlation, duration and runtime measurements. Subjective improvement requires listening approval; correctness tests alone cannot establish it.

## Interface

Dark studio theme with restrained accent colors. Left: saved projects. Center: track name, waveform, transport and original/processed audition. Right: labeled spatial controls with numeric values and reset. Bottom: render queue and exports. Empty, loading, failed, interrupted and completed states must be explicit. Keyboard-operable controls, visible focus, and no autoplay. The primary action progresses from Import to Preview to Render; export becomes available only when the file is complete.

## Verification and release criteria

- Existing relevant DSP tests pass before and after adapter integration, with any pre-existing failures recorded.
- API integration tests cover importing, project persistence, invalid settings, queue limits, worker failure, restart recovery, cancellation, and file access boundaries.
- Synthetic audio runs through import → preview → full render → export, producing finite stereo audio of the expected duration and sample rate.
- Opening a saved project restores settings and completed renders after an application restart.
- Browser checks cover transport, A/B preview, controls, render progress and actionable errors.
- Record baseline and revised render duration and peak memory on the same input and machine; no speed multiplier claimed without measurement.
- Windows scripts are reviewed here; actual Windows installation, browser launch and real-track listening require a Windows execution environment. Linux verification must never be reported as Windows verification.

## Delivery

Keep the rebuild on a separate branch and prepare reviewable commits and a draft pull request when implementation is verified. Provide a Windows startup guide and exact launcher filenames. Do not replace the production deployment or merge the rebuild as part of local development.
