from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set
import pandas as pd
import numpy as np


@dataclass
class UserBaseline:
    common_devices: Set[str]
    common_cities: Set[str]
    mean_bytes: float
    std_bytes: float
    work_start: int
    work_end: int


def fit_user_baselines(normal_logs: pd.DataFrame) -> Dict[str, UserBaseline]:
    baselines: Dict[str, UserBaseline] = {}

    for user_id, g in normal_logs.groupby("user_id"):
        g = g.sort_values("timestamp").copy()
        top_devices = set(g["device_id"].value_counts().head(3).index.tolist())
        top_cities = set(g["city"].value_counts().head(2).index.tolist())
        mean_bytes = float(g["bytes"].mean()) if not g.empty else 0.0
        std_bytes = float(g["bytes"].std()) if len(g) > 1 else 1.0
        if std_bytes == 0:
            std_bytes = 1.0

        hours = g["timestamp"].dt.hour
        q10 = int(np.percentile(hours, 10)) if not g.empty else 8
        q90 = int(np.percentile(hours, 90)) if not g.empty else 17

        baselines[user_id] = UserBaseline(
            common_devices=top_devices,
            common_cities=top_cities,
            mean_bytes=mean_bytes,
            std_bytes=std_bytes,
            work_start=q10,
            work_end=q90,
        )

    return baselines


def add_context_features(df: pd.DataFrame, baselines: Dict[str, UserBaseline]) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["weekday"] = out["timestamp"].dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    new_device_flags = []
    new_city_flags = []
    off_hours_flags = []
    byte_zscores = []

    for _, row in out.iterrows():
        b = baselines.get(row["user_id"])
        if b is None:
            new_device_flags.append(0)
            new_city_flags.append(0)
            off_hours_flags.append(0)
            byte_zscores.append(0.0)
            continue

        new_device_flags.append(0 if row["device_id"] in b.common_devices else 1)
        new_city_flags.append(0 if row["city"] in b.common_cities else 1)

        off_hour = 1 if (row["hour"] < b.work_start or row["hour"] > b.work_end) else 0
        off_hours_flags.append(off_hour)

        byte_zscores.append((row["bytes"] - b.mean_bytes) / b.std_bytes)

    out["new_device"] = new_device_flags
    out["new_city"] = new_city_flags
    out["off_hours"] = off_hours_flags
    out["bytes_zscore"] = byte_zscores
    out["is_external_event"] = out["share_scope"].isin(["external", "public"]).astype(int)
    return out


def compute_window_features(
    df: pd.DataFrame,
    short_minutes: int = 10,
    long_minutes: int = 30,
) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).copy()
    result_frames: List[pd.DataFrame] = []

    short_delta = pd.Timedelta(minutes=short_minutes)
    long_delta = pd.Timedelta(minutes=long_minutes)

    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()

        short_q = deque()
        long_q = deque()

        events_10m = []
        views_10m = []
        downloads_10m = []
        edits_10m = []
        deletes_10m = []
        renames_10m = []
        bytes_10m = []
        files_touched_10m = []

        shares_30m = []
        external_shares_30m = []
        permission_changes_30m = []

        for i, row in g.iterrows():
            current_ts = row["timestamp"]

            while short_q and (current_ts - short_q[0]["timestamp"]) > short_delta:
                short_q.popleft()

            while long_q and (current_ts - long_q[0]["timestamp"]) > long_delta:
                long_q.popleft()

            current_event = {
                "timestamp": current_ts,
                "action": row["action"],
                "bytes": row["bytes"],
                "file_id": row["file_id"],
                "is_external_event": row["is_external_event"],
                "permission_change": row["permission_change"],
            }

            short_q.append(current_event)
            long_q.append(current_event)

            short_events = list(short_q)
            long_events = list(long_q)

            events_10m.append(len(short_events))
            views_10m.append(sum(1 for e in short_events if e["action"] == "view"))
            downloads_10m.append(sum(1 for e in short_events if e["action"] == "download"))
            edits_10m.append(sum(1 for e in short_events if e["action"] == "edit"))
            deletes_10m.append(sum(1 for e in short_events if e["action"] == "delete"))
            renames_10m.append(sum(1 for e in short_events if e["action"] == "rename"))
            bytes_10m.append(sum(e["bytes"] for e in short_events))
            files_touched_10m.append(len(set(e["file_id"] for e in short_events)))

            shares_30m.append(sum(1 for e in long_events if e["action"] == "share"))
            external_shares_30m.append(sum(int(e["is_external_event"]) for e in long_events))
            permission_changes_30m.append(sum(int(e["permission_change"]) for e in long_events))

        g["events_10m"] = events_10m
        g["views_10m"] = views_10m
        g["downloads_10m"] = downloads_10m
        g["edits_10m"] = edits_10m
        g["deletes_10m"] = deletes_10m
        g["renames_10m"] = renames_10m
        g["bytes_10m"] = bytes_10m
        g["files_touched_10m"] = files_touched_10m
        g["shares_30m"] = shares_30m
        g["external_shares_30m"] = external_shares_30m
        g["permission_changes_30m"] = permission_changes_30m

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


def build_subject_features(
    df: pd.DataFrame,
    baselines: Dict[str, UserBaseline],
    short_minutes: int = 10,
    long_minutes: int = 30,
) -> pd.DataFrame:
    out = add_context_features(df, baselines)
    out = compute_window_features(out, short_minutes=short_minutes, long_minutes=long_minutes)
    return out