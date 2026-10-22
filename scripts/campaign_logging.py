"""Stream child output immediately while retaining a complete per-worker log."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time


def emit_event(event, **fields):
    print(
        json.dumps(
            dict(
                event=event,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                **fields,
            ),
            allow_nan=False,
        ),
        flush=True,
    )


class LoggedWorker:
    def __init__(self, command, path, index, env=None, sink=None):
        self.index = index
        self.last_output_at = time.monotonic()
        self.reader_error = None
        self.sink = sink or (lambda message: print(message, flush=True))
        self.path = Path(path)
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            with self.path.open("w", encoding="utf-8") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    self.last_output_at = time.monotonic()
                    self.sink(f"[worker {self.index}] {line.rstrip()}")
        except Exception as error:
            self.reader_error = str(error)
        finally:
            self.process.stdout.close()

    def stop(self, grace_seconds=30):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=5)

    def status(self):
        return dict(
            worker=self.index,
            pid=self.process.pid,
            exit_code=self.process.poll(),
            seconds_since_output=round(time.monotonic() - self.last_output_at, 1),
            reader_error=self.reader_error,
        )


def gpu_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rows = []
        for line in result.stdout.splitlines():
            index, name, utilization, used, total = [
                part.strip() for part in line.split(",")
            ]
            rows.append(
                dict(
                    index=int(index),
                    name=name,
                    utilization_percent=int(utilization),
                    memory_used_mib=int(used),
                    memory_total_mib=int(total),
                )
            )
        return rows
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return dict(unavailable=type(error).__name__)
