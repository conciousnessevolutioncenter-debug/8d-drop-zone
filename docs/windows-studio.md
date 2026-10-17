# 8D Engine Studio — Windows quick start

## First launch

1. Download the `rebuild/windows-studio` branch as ZIP and extract it completely.
2. Install **64-bit Python 3.12**, including the Python launcher, if it is not already installed.
3. Double-click **SETUP_STUDIO_WINDOWS.cmd**. Setup needs internet access to install dependencies.
4. When setup reports success, double-click **START_STUDIO_WINDOWS.cmd**.
5. Keep the console window open. Studio opens at **http://127.0.0.1:8765** when ready.

For subsequent sessions, use START_STUDIO_WINDOWS.cmd. Close Studio with Ctrl+C in the console.

## Produce a track

Import a WAV or FLAC, adjust movement and depth, then create a 15-second preview. Click **8D preview** and Play to audition it; use **Original** for comparison. Preview starts at the specified seconds into the track. It is a separately processed excerpt: spatial phase and mastering can differ from the full-track output. It is not a sample-exact preview of the full render.

Choose WAV (float), FLAC (24-bit), or MP3 (320 kbps), then **Render full track**. Download appears only after successful completion. Settings save automatically and each render retains its own settings snapshot. Changing controls invalidates the current preview.

MP3/M4A import and MP3 export require **FFmpeg** on PATH. WAV/FLAC do not. After installing FFmpeg, reopen Studio. The MP3 export option is disabled when FFmpeg is unavailable. The studio never uploads your audio to a cloud service.

## Storage and restart

Projects, source copies, decoded audition WAV files, render history and exports live in `%LOCALAPPDATA%\8DEngineStudio`. Keep free disk space available: a decoded WAV can be significantly larger than an MP3. Original files are not overwritten. Back up this folder while Studio is closed. Developer/test override: `EIGHTD_STUDIO_DATA_DIR`.

After an interrupted session, running jobs are marked **interrupted** and can be retried from the beginning. Queued jobs stay paused until you click **Resume queue**. Cancellation during DSP is checked at rendering boundaries; decoding/mastering/encoding may finish their current stage before cancellation is acknowledged.

## Troubleshooting

- **Missing Python:** install Python 3.12 with its launcher, then run setup again.
- **Setup failed:** keep the console visible and check its final error. Retry setup after restoring internet connectivity.
- **Port 8765 in use:** close the previous Studio console or the application using that port. The launcher never takes over an unrelated service.
- **Cannot decode:** try WAV/FLAC, or install FFmpeg for compressed formats.
- **Session expired:** refresh the browser after restarting the server.
- **Render failed:** check source audio, free disk space and `%LOCALAPPDATA%\8DEngineStudio\studio.log`.
- **Browser audio unavailable:** download the WAV and audition it in a local player.

## Release scope and verification

This is a first Windows-first Studio build, with the existing HRTF-inspired renderer. It does not claim personalized measured-HRTF rendering, proven subjective sound improvement, or a measured speed multiplier. Stem mixing and a standalone installer are later milestones. The existing hosted app remains separate.

Automated tests and Linux browser verification are recorded in `docs/studio/verification.md`. Actual Windows installation and real-track listening are separate acceptance steps; they must not be inferred from Linux tests.
