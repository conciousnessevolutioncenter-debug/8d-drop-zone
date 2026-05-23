"""Dark drag-and-drop desktop GUI for The 8D Drop-Zone.

The UI is intentionally simple: drop an MP3/WAV/audio file on the window and the
application automatically performs smart analysis, renders professional 8D audio,
and writes `<original>_8D_Final.wav` next to the source file.
"""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

try:  # Native OS drag-and-drop support. App still launches with a helpful note if missing.
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    DND_AVAILABLE = True
except Exception:  # pragma: no cover - environment-specific optional dependency
    DND_FILES = "DND_Files"
    TkinterDnD = None  # type: ignore[assignment]
    DND_AVAILABLE = False

from .audio_io import SUPPORTED_INPUTS, export_audio, load_audio
from .dsp import (
    analyze_correlation,
    bpm_to_two_bar_rotation_cpm,
    estimate_bpm,
    process_8d,
)


BaseWindow = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk


class DropZoneApp(BaseWindow):
    """One-action drop-zone front-end for automatic 8D conversion."""

    BG = "#0b0f17"
    PANEL = "#111827"
    PANEL_HOVER = "#16324f"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    ACCENT = "#38bdf8"
    SUCCESS = "#22c55e"
    ERROR = "#ef4444"

    def __init__(self) -> None:
        super().__init__()
        self.title("8D Drop-Zone")
        self.geometry("760x520")
        self.minsize(680, 460)
        self.configure(bg=self.BG)

        self._messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self._busy = False

        self.headline = tk.StringVar(value="Drop Audio Here")
        self.subhead = tk.StringVar(value="MP3, WAV, FLAC, OGG, AIFF, or M4A → automatic professional 8D render")
        self.status = tk.StringVar(value="Ready. Drag a file onto the window.")
        self.analysis = tk.StringVar(value="Smart engine: BPM lock • 150 Hz mono bass protection • binaural orbit • YouTube-safe master")
        self.output_text = tk.StringVar(value="Output will save beside the source as *_8D_Final.wav")

        self._build_style()
        self._build_ui()
        self._enable_drop_target()
        self.after(120, self._drain_messages)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#020617",
            background=self.ACCENT,
            bordercolor=self.PANEL,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
        )

    def _build_ui(self) -> None:
        shell = tk.Frame(self, bg=self.BG)
        shell.pack(fill="both", expand=True, padx=26, pady=24)

        title = tk.Label(
            shell,
            text="8D Drop-Zone",
            bg=self.BG,
            fg=self.TEXT,
            font=("TkDefaultFont", 28, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            shell,
            text="Drag. Analyze. Convert. No file dialogs, no setup decisions.",
            bg=self.BG,
            fg=self.MUTED,
            font=("TkDefaultFont", 12),
        )
        subtitle.pack(anchor="w", pady=(4, 18))

        self.drop_frame = tk.Frame(
            shell,
            bg=self.PANEL,
            highlightbackground="#334155",
            highlightcolor=self.ACCENT,
            highlightthickness=2,
            bd=0,
        )
        self.drop_frame.pack(fill="both", expand=True)

        self.drop_icon = tk.Label(
            self.drop_frame,
            text="⬇",
            bg=self.PANEL,
            fg=self.ACCENT,
            font=("TkDefaultFont", 54, "bold"),
        )
        self.drop_icon.pack(pady=(46, 6))

        self.headline_label = tk.Label(
            self.drop_frame,
            textvariable=self.headline,
            bg=self.PANEL,
            fg=self.TEXT,
            font=("TkDefaultFont", 24, "bold"),
        )
        self.headline_label.pack()

        self.subhead_label = tk.Label(
            self.drop_frame,
            textvariable=self.subhead,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("TkDefaultFont", 12),
            wraplength=600,
            justify="center",
        )
        self.subhead_label.pack(pady=(8, 18))

        self.progress = ttk.Progressbar(self.drop_frame, mode="indeterminate", length=440)
        self.progress.pack(pady=(2, 18))

        self.status_label = tk.Label(
            self.drop_frame,
            textvariable=self.status,
            bg=self.PANEL,
            fg=self.TEXT,
            font=("TkDefaultFont", 12, "bold"),
            wraplength=630,
            justify="center",
        )
        self.status_label.pack(pady=(2, 8))

        self.analysis_label = tk.Label(
            self.drop_frame,
            textvariable=self.analysis,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("TkDefaultFont", 10),
            wraplength=630,
            justify="center",
        )
        self.analysis_label.pack(pady=(0, 8))

        self.output_label = tk.Label(
            self.drop_frame,
            textvariable=self.output_text,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("TkDefaultFont", 10),
            wraplength=630,
            justify="center",
        )
        self.output_label.pack(pady=(0, 28))

        footer = tk.Label(
            shell,
            text=(
                "DSP chain: stereo input → 150 Hz crossover → mono/static bass → "
                "mid/high HRTF-inspired spherical orbit with ITD/ILD/rear shading → subtle room → -13 dB RMS master"
            ),
            bg=self.BG,
            fg=self.MUTED,
            font=("TkDefaultFont", 9),
            wraplength=700,
            justify="left",
        )
        footer.pack(anchor="w", pady=(14, 0))

        if not DND_AVAILABLE:
            self._set_error(
                "Drag-and-drop library missing. Install it with: pip install tkinterdnd2"
            )

    def _enable_drop_target(self) -> None:
        if not DND_AVAILABLE:
            return
        # Register the whole window and visible drop card so native hover/drop works reliably.
        for widget in (self, self.drop_frame):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _set_drop_colors(self, panel_color: str, accent_color: str | None = None) -> None:
        accent = accent_color or self.ACCENT
        for widget in (
            self.drop_frame,
            self.drop_icon,
            self.headline_label,
            self.subhead_label,
            self.status_label,
            self.analysis_label,
            self.output_label,
        ):
            widget.configure(bg=panel_color)
        self.drop_icon.configure(fg=accent)
        self.drop_frame.configure(highlightbackground=accent, highlightcolor=accent)

    def _on_drop_enter(self, _event: object) -> str:
        if not self._busy:
            self.headline.set("Release to Convert")
            self.subhead.set("The 8D engine will analyze tempo, protect bass, render motion, and auto-export.")
            self._set_drop_colors(self.PANEL_HOVER)
        return "copy"

    def _on_drop_leave(self, _event: object) -> str:
        if not self._busy:
            self._reset_ready_state()
        return "copy"

    def _on_drop(self, event: object) -> str:
        if self._busy:
            return "copy"
        raw_data = getattr(event, "data", "")
        paths = [Path(p) for p in self.tk.splitlist(raw_data)]
        audio_paths = [p for p in paths if p.suffix.lower() in SUPPORTED_INPUTS and p.exists()]
        if not audio_paths:
            self._set_error("Unsupported drop. Please drop an MP3/WAV/FLAC/OGG/AIFF/M4A audio file.")
            return "copy"
        self._start_conversion(audio_paths[0])
        return "copy"

    def _start_conversion(self, input_path: Path) -> None:
        output_path = input_path.with_name(f"{input_path.stem}_8D_Final.wav")
        self._busy = True
        self.headline.set("Processing…")
        self.subhead.set(input_path.name)
        self.status.set("Loading audio and detecting BPM…")
        self.analysis.set("Please wait. Long files can take a moment; the UI will remain responsive.")
        self.output_text.set(f"Export target: {output_path.name}")
        self._set_drop_colors(self.PANEL, self.ACCENT)
        self.progress.start(12)

        thread = threading.Thread(target=self._process_worker, args=(input_path, output_path), daemon=True)
        thread.start()

    def _process_worker(self, input_path: Path, output_path: Path) -> None:
        try:
            audio = load_audio(input_path)
            bpm = estimate_bpm(audio)
            rotation_cpm = bpm_to_two_bar_rotation_cpm(bpm)
            rendered = process_8d(
                audio,
                rotation_cpm=rotation_cpm,
                room_size=0.22,
                crossover_hz=150.0,
                motion_depth=1.05,
                high_emphasis=0.45,
                youtube_master=True,
                section_automation=True,
            )
            report = analyze_correlation(rendered.samples)
            export_audio(rendered, output_path)
        except Exception as exc:  # pragma: no cover - GUI path
            self._messages.put(("error", str(exc)))
        else:
            payload = "|".join(
                [
                    str(output_path),
                    f"{bpm:.1f}",
                    f"{rotation_cpm:.2f}",
                    f"{report.correlation:.3f}",
                    f"{report.side_mid_ratio:.3f}",
                    "phase warning" if report.phase_warning else "phase safe",
                ]
            )
            self._messages.put(("done", payload))

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self._messages.get_nowait()
                self.progress.stop()
                self._busy = False
                if kind == "done":
                    output, bpm, cpm, corr, side_mid, phase = payload.split("|", maxsplit=5)
                    self.headline.set("Success!")
                    self.subhead.set(Path(output).name)
                    self.status.set(f"Saved: {output}")
                    self.analysis.set(
                        f"Detected BPM: {bpm} • 2-bar rotation: {cpm} cycles/min • "
                        f"Correlation: {corr} • Side/Mid: {side_mid} • {phase}"
                    )
                    self.output_text.set("Drop another file to convert again.")
                    self._set_drop_colors(self.PANEL, self.SUCCESS)
                    messagebox.showinfo("8D conversion complete", f"Saved:\n{output}")
                else:
                    self._set_error(payload)
        except queue.Empty:
            pass
        self.after(120, self._drain_messages)

    def _reset_ready_state(self) -> None:
        self.headline.set("Drop Audio Here")
        self.subhead.set("MP3, WAV, FLAC, OGG, AIFF, or M4A → automatic professional 8D render")
        self.status.set("Ready. Drag a file onto the window.")
        self.analysis.set("Smart engine: BPM lock • 150 Hz mono bass protection • binaural orbit • YouTube-safe master")
        self.output_text.set("Output will save beside the source as *_8D_Final.wav")
        self._set_drop_colors(self.PANEL, self.ACCENT)

    def _set_error(self, message: str) -> None:
        self.headline.set("Cannot Convert")
        self.subhead.set("Fix the issue below, then drop the file again.")
        self.status.set(message)
        self.analysis.set("No audio was exported.")
        self.output_text.set("Supported inputs: MP3, WAV, FLAC, OGG, AIFF, M4A")
        self._set_drop_colors(self.PANEL, self.ERROR)
        if self._busy:
            self._busy = False
            self.progress.stop()


# Backward-compatible alias for older imports/tests.
EightDEngineApp = DropZoneApp


def main() -> None:
    app = DropZoneApp()
    app.mainloop()


if __name__ == "__main__":
    main()
