"""Static ASCII art. Kept as plain strings so the app needs no figlet dependency.

Art is decoration around real metrics — it never fakes telemetry.
"""

# The Daish wordmark. Shown at >=140 columns and >=42 rows.
WORDMARK = """\
 ____    _    ___ ____  _   _
|  _ \\  / \\  |_ _/ ___|| | | |
| | | |/ _ \\  | |\\___ \\| |_| |
| |_| / ___ \\ | | ___) |  _  |
|____/_/   \\_\\___|____/|_| |_|"""

# GPU board / "engine room" silhouette for the wide Hardware view.
# The two fan outlines are static; only the bars and numeric labels
# respond to measured data.
def gpu_board(width: int, vram_f: float, load_f: float, temp: str, watts: str,
              ascii_only: bool = False) -> list[str]:
    """Render the engine-room board with live memory/load bars.

    vram_f and load_f are fractions in [0, 1]. Unknown values render N/A.
    """
    def bar(f: float, n: int) -> str:
        if f is None:
            return "." * n
        f = max(0.0, min(1.0, f))
        filled = int(round(n * f))
        return "#" * filled + "." * (n - filled)

    def bar_ascii(f: float, n: int) -> str:
        if f is None:
            return "." * n
        f = max(0.0, min(1.0, f))
        filled = int(round(n * f))
        return ("#" * filled).ljust(n, ".")

    b1 = (bar_ascii if ascii_only else bar)(vram_f, 12)
    b2 = (bar_ascii if ascii_only else bar)(load_f, 12)
    vram_label = "VRAM " if vram_f is not None else "VRAM "
    temp_label = temp if temp else "N/A"
    watt_label = watts if watts else "N/A"
    lines = [
        "+-------- GPU / ENGINE ROOM --------+",
        "| .----.  .----.  VRAM  [%s] |" % b1,
        "| | () |  | () |  LOAD  [%s] |" % b2,
        "| '----'  '----'  TEMP %s  PWR %s |" % (temp_label, watt_label),
        "+--|||||||||||||||------------------+",
    ]
    return lines
