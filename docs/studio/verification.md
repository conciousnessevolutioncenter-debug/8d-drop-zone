# Studio verification record

Development environment: Linux, Python 3.12. This does not certify Windows installation.

- Original DSP baseline: 19 tests passed before changes.
- Studio suite after review fixes: 28 tests passed (storage, settings, import, limits, local request guards, renderer, cancellation, queue/recovery, real API export/restart, launcher).
- Whole repository after installing hosted-app dependencies: 96 passed, 5 failed before adding the final five regression tests.
- The same five failures were reproduced against an untouched `main` archive in the same environment:
  - `tests/test_live_fast_upload.py::test_job_status_exposes_download_when_background_render_completes`
  - `tests/test_live_fast_upload.py::test_ai_stem_mode_uses_separation_and_stem_renderer`
  - `tests/test_live_fast_upload.py::test_rejects_uploaded_tracks_longer_than_20_minutes`
  - `tests/test_live_fast_upload.py::test_homepage_exposes_ai_stem_mode_choice_and_20_minute_limit`
  - `tests/test_tracks.py::test_track_routes_registered_and_publish_exists`
- Initial suite run also lacked hosted-app optional packages; those environment failures were resolved before the comparison above.
- JavaScript syntax: checked with `node --check studio/static/studio.js`.
- Browser E2E: not completed. Playwright is present but Chromium is absent; browser installation was blocked by network approval cancellation. Visual layout, playback, A/B and responsive browser checks remain outstanding.
- Windows setup/launch: scripts provided and reviewed; execution on actual Windows remains outstanding.
- Real music listening: outstanding. No subjective quality improvement is claimed.

## Measured render smoke test

20-second synthetic stereo WAV, 44,100 Hz, current default Studio settings:

- elapsed: 0.959 seconds
- process peak RSS: 178,176 KiB (includes interpreter/imports)
- output: 882,000 frames, stereo, 44,100 Hz
- sample peak: 0.2900
- correlation: 0.7773
- measured loudness: -19.0 LUFS; true peak: -10.75 dB

This is a single host measurement, not a speedup comparison or a Windows estimate.

## Independent review fixes

Fixed project-switch and submit races, stale exports on empty projects, preview offset loss, Windows UTF-8 loading, request-size limits before and during multipart parsing, bounded compressed decoding with finite-sample validation, and transient storage-error recovery. Each issue has a regression test observed failing before its fix. Four Node UI-state regression tests pass; these simulate state and do not replace browser E2E.
