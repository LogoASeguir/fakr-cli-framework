from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import uuid


@dataclass
class Job:
    id: str
    type: str
    params: Dict[str, Any]
    status: str = "queued"   # queued | running | done | failed

    @staticmethod
    def new(job_type: str, params: Dict[str, Any]) -> "Job":
        return Job(id=str(uuid.uuid4()), type=job_type, params=params)


class JobQueue:
    def __init__(self) -> None:
        self._queue: List[Job] = []

    def enqueue(self, job: Job) -> None:
        self._queue.append(job)

    def dequeue(self) -> Job | None:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def __len__(self) -> int:
        return len(self._queue)
