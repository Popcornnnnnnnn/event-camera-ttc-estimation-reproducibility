from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .formal_loader import (
    BoundingBoxRecord,
    Calibration,
    EventSlice,
    FormalSequenceReader,
    IntervalSample,
    ProtocolSpec,
    SequenceMeta,
)


@dataclass(frozen=True)
class NativeSampleBundle:
    sequence_meta: SequenceMeta
    protocol: ProtocolSpec
    calibration: Calibration
    interval_sample: IntervalSample
    event_slice: EventSlice | None
    bbox_records: tuple[BoundingBoxRecord, ...] = ()
    interval_count: int = 0

    @property
    def sequence_id(self) -> str:
        return self.sequence_meta.sequence_id

    @property
    def sample_id(self) -> str:
        return self.interval_sample.sample_id


class NativeSampleAdapter:
    def __init__(
        self,
        sequence_id: str,
        *,
        formal_root: Path | None = None,
    ) -> None:
        self.reader = FormalSequenceReader(sequence_id, formal_root=formal_root)
        self._bbox_records: tuple[BoundingBoxRecord, ...] | None = None

    def __enter__(self) -> NativeSampleAdapter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.reader.close()

    def summary(self) -> dict[str, object]:
        payload = self.reader.summary()
        payload["adapter"] = "NativeSampleAdapter"
        payload["supports_lazy_events"] = True
        return payload

    @property
    def bbox_records(self) -> tuple[BoundingBoxRecord, ...]:
        if self._bbox_records is None:
            self._bbox_records = tuple(self.reader.bbox_records)
        return self._bbox_records

    def get_sample(
        self,
        *,
        interval_idx: int | None = None,
        sample_id: str | None = None,
        include_events: bool = True,
    ) -> NativeSampleBundle:
        sample = self.reader.get_interval_sample(interval_idx=interval_idx, sample_id=sample_id)
        event_slice = self.reader.read_event_slice(sample=sample) if include_events else None
        return NativeSampleBundle(
            sequence_meta=self.reader.sequence_meta,
            protocol=self.reader.protocol,
            calibration=self.reader.calibration,
            interval_sample=sample,
            event_slice=event_slice,
            bbox_records=self.bbox_records,
            interval_count=len(self.reader.interval_samples),
        )

    def iter_samples(
        self,
        *,
        include_events: bool = False,
        start_interval_idx: int = 0,
        max_samples: int | None = None,
    ) -> Iterator[NativeSampleBundle]:
        emitted = 0
        for sample in self.reader.iter_interval_samples():
            if sample.interval_idx < start_interval_idx:
                continue
            yield self.get_sample(sample_id=sample.sample_id, include_events=include_events)
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                break


__all__ = ["NativeSampleAdapter", "NativeSampleBundle"]
