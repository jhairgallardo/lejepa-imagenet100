"""Utilities for logging experiment output."""

import json
import os
import sys


def save_args(args, save_dir):
    """Write the parsed arguments to ``save_dir/args.json``."""
    with open(os.path.join(save_dir, "args.json"), "w") as file:
        json.dump(vars(args), file, indent=2)


def format_duration(seconds):
    """Format a duration in seconds as H:MM:SS.ss."""
    # Round to hundredths first so seconds never display as 60.00.
    minutes, centiseconds = divmod(round(seconds * 100), 6000)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{centiseconds / 100:05.2f}"


class Tee:
    """Mirror stdout writes to both the terminal and a log file.

    Usage:
        sys.stdout = Tee(log_path)

    Only activate on rank 0 in DDP — other ranks share the same terminal
    but should not write to the same file.
    """

    def __init__(self, log_path: str):
        # Line buffering flushes the file after each newline.
        self.file = open(log_path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, data: str):
        """Write text to the log file and the original stdout stream."""
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        """Flush both output streams."""
        self.file.flush()
        self.stdout.flush()

    def close(self):
        """Close the log file."""
        self.file.close()

