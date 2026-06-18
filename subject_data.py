from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
import numpy as np
import pandas as pd


ACTIONS = ["view", "edit", "download", "share", "delete", "rename", "permission_change"]
FILE_TYPES = ["pdf", "docx", "xlsx", "pptx", "txt", "csv"]


@dataclass
class PersonaTemplate:
    role: str
    department: str
    daily_event_range: Tuple[int, int]
    work_hour_range: Tuple[int, int]
    action_probs: Dict[str, float]
    external_share_prob: float
    permission_change_prob: float
    typical_devices: int
    typical_locations: int
    app_probs: Dict[str, float]


@dataclass
class UserProfile:
    user_id: str
    role: str
    department: str
    devices: List[str] = field(default_factory=list)
    locations: List[Tuple[str, str]] = field(default_factory=list)
    action_probs: Dict[str, float] = field(default_factory=dict)
    external_share_prob: float = 0.0
    permission_change_prob: float = 0.0
    work_hour_range: Tuple[int, int] = (9, 17)
    daily_event_range: Tuple[int, int] = (10, 20)
    app_probs: Dict[str, float] = field(default_factory=dict)


def build_persona_templates() -> Dict[str, PersonaTemplate]:
    return {
        "office_staff": PersonaTemplate(
            role="office_staff",
            department="operations",
            daily_event_range=(20, 60),
            work_hour_range=(8, 17),
            action_probs={
                "view": 0.45,
                "edit": 0.25,
                "download": 0.15,
                "share": 0.10,
                "delete": 0.02,
                "rename": 0.01,
                "permission_change": 0.02,
            },
            external_share_prob=0.05,
            permission_change_prob=0.02,
            typical_devices=2,
            typical_locations=2,
            app_probs={"web": 0.65, "desktop": 0.30, "mobile": 0.05},
        ),
        "manager": PersonaTemplate(
            role="manager",
            department="management",
            daily_event_range=(15, 50),
            work_hour_range=(8, 19),
            action_probs={
                "view": 0.35,
                "edit": 0.20,
                "download": 0.15,
                "share": 0.20,
                "delete": 0.03,
                "rename": 0.02,
                "permission_change": 0.05,
            },
            external_share_prob=0.12,
            permission_change_prob=0.05,
            typical_devices=3,
            typical_locations=3,
            app_probs={"web": 0.55, "desktop": 0.35, "mobile": 0.10},
        ),
        "contractor": PersonaTemplate(
            role="contractor",
            department="external",
            daily_event_range=(5, 30),
            work_hour_range=(9, 18),
            action_probs={
                "view": 0.50,
                "edit": 0.20,
                "download": 0.10,
                "share": 0.12,
                "delete": 0.03,
                "rename": 0.02,
                "permission_change": 0.03,
            },
            external_share_prob=0.08,
            permission_change_prob=0.03,
            typical_devices=2,
            typical_locations=3,
            app_probs={"web": 0.75, "desktop": 0.15, "mobile": 0.10},
        ),
        "it_admin": PersonaTemplate(
            role="it_admin",
            department="it",
            daily_event_range=(10, 40),
            work_hour_range=(7, 19),
            action_probs={
                "view": 0.25,
                "edit": 0.10,
                "download": 0.10,
                "share": 0.10,
                "delete": 0.05,
                "rename": 0.05,
                "permission_change": 0.35,
            },
            external_share_prob=0.03,
            permission_change_prob=0.35,
            typical_devices=3,
            typical_locations=2,
            app_probs={"web": 0.50, "desktop": 0.45, "mobile": 0.05},
        ),
    }


def create_user_profiles(user_counts: Dict[str, int], seed: int = 42) -> List[UserProfile]:
    random.seed(seed)
    templates = build_persona_templates()

    possible_locations = [
        ("NZ", "Auckland"),
        ("NZ", "Wellington"),
        ("NZ", "Christchurch"),
        ("AU", "Sydney"),
    ]

    users: List[UserProfile] = []
    user_idx = 1

    for role, count in user_counts.items():
        t = templates[role]
        for _ in range(count):
            devices = [f"{role}_device_{user_idx}_{i}" for i in range(t.typical_devices)]
            locations = random.sample(possible_locations, k=t.typical_locations)

            users.append(
                UserProfile(
                    user_id=f"user_{user_idx:03d}",
                    role=t.role,
                    department=t.department,
                    devices=devices,
                    locations=locations,
                    action_probs=t.action_probs,
                    external_share_prob=t.external_share_prob,
                    permission_change_prob=t.permission_change_prob,
                    work_hour_range=t.work_hour_range,
                    daily_event_range=t.daily_event_range,
                    app_probs=t.app_probs,
                )
            )
            user_idx += 1

    return users


def create_file_catalog(users: List[UserProfile], files_per_department: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    files = []
    file_idx = 1
    dept_users: Dict[str, List[str]] = {}
    for u in users:
        dept_users.setdefault(u.department, []).append(u.user_id)

    for dept, owner_ids in dept_users.items():
        for _ in range(files_per_department):
            file_type = random.choices(FILE_TYPES, weights=[0.2, 0.25, 0.2, 0.1, 0.15, 0.1], k=1)[0]
            sensitivity = random.choices(["low", "medium", "high"], weights=[0.5, 0.35, 0.15], k=1)[0]
            owner_id = random.choice(owner_ids)

            files.append({
                "file_id": f"file_{file_idx:05d}",
                "department": dept,
                "owner_user_id": owner_id,
                "file_type": file_type,
                "sensitivity": sensitivity,
            })
            file_idx += 1

    return pd.DataFrame(files)


def _weighted_choice(prob_dict: Dict[str, float]) -> str:
    return random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()), k=1)[0]


def _sample_timestamp_for_user(day: datetime, user: UserProfile) -> datetime:
    start_hour, end_hour = user.work_hour_range
    if random.random() < 0.9:
        hour = random.randint(start_hour, max(start_hour, end_hour - 1))
    else:
        hour = random.choice([max(0, start_hour - 2), min(23, end_hour + 1)])
    return day.replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0,
    )


def _sample_bytes(action: str) -> int:
    if action == "view":
        return int(np.random.randint(500, 10_000))
    if action == "edit":
        return int(np.random.randint(2_000, 50_000))
    if action == "download":
        return int(np.random.randint(20_000, 3_000_000))
    if action == "share":
        return int(np.random.randint(0, 3_000))
    if action == "permission_change":
        return int(np.random.randint(0, 1_000))
    if action in {"delete", "rename"}:
        return int(np.random.randint(0, 2_000))
    return 0


def _choose_file_for_user(user: UserProfile, file_catalog: pd.DataFrame) -> pd.Series:
    same_dept = file_catalog[file_catalog["department"] == user.department]
    other_dept = file_catalog[file_catalog["department"] != user.department]

    if random.random() < 0.8 or other_dept.empty:
        return same_dept.sample(1).iloc[0]
    return other_dept.sample(1).iloc[0]


def generate_normal_logs(
    users: List[UserProfile],
    file_catalog: pd.DataFrame,
    days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    events: List[dict] = []
    event_counter = 1

    for day_offset in range(days):
        current_day = start_day + timedelta(days=day_offset)
        is_weekend = current_day.weekday() >= 5

        for user in users:
            min_events, max_events = user.daily_event_range
            if is_weekend:
                min_events = max(0, int(min_events * 0.25))
                max_events = max(1, int(max_events * 0.40))

            event_count = random.randint(min_events, max_events)

            for _ in range(event_count):
                action = _weighted_choice(user.action_probs)
                ts = _sample_timestamp_for_user(current_day, user)
                country, city = random.choice(user.locations)
                device_id = random.choice(user.devices)
                app = _weighted_choice(user.app_probs)
                file_row = _choose_file_for_user(user, file_catalog)

                share_scope = "none"
                is_external_share = 0
                permission_change = 0

                if action == "share":
                    if random.random() < user.external_share_prob:
                        share_scope = random.choice(["external", "public"])
                        is_external_share = 1
                    else:
                        share_scope = "internal"

                if action == "permission_change":
                    permission_change = 1

                events.append({
                    "event_id": f"evt_{event_counter:08d}",
                    "timestamp": ts,
                    "user_id": user.user_id,
                    "role": user.role,
                    "department": user.department,
                    "file_id": file_row["file_id"],
                    "file_type": file_row["file_type"],
                    "file_sensitivity": file_row["sensitivity"],
                    "action": action,
                    "device_id": device_id,
                    "country": country,
                    "city": city,
                    "app": app,
                    "is_external_share": is_external_share,
                    "share_scope": share_scope,
                    "permission_change": permission_change,
                    "bytes": _sample_bytes(action),
                    "label": 0,
                    "intent": "normal",
                    "scenario_id": "",
                })
                event_counter += 1

    df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    return df


def inject_attack_scenarios(
    logs: pd.DataFrame,
    users: List[UserProfile],
    file_catalog: pd.DataFrame,
    attack_counts: Dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    logs = logs.copy()
    new_events: List[dict] = []
    next_event_num = len(logs) + 1

    users_by_id = {u.user_id: u for u in users}
    start_day = logs["timestamp"].min().normalize()
    end_day = logs["timestamp"].max().normalize()

    possible_days = pd.date_range(start=start_day, end=end_day, freq="D")

    def choose_attack_time() -> datetime:
        base_day = random.choice(list(possible_days))
        # Attack events lean toward off-hours for stronger signal
        return pd.Timestamp(base_day).replace(
            hour=random.choice([1, 2, 3, 20, 21, 22]),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        ).to_pydatetime()

    def build_event(
        ts: datetime,
        user: UserProfile,
        action: str,
        file_id: str,
        file_type: str,
        file_sensitivity: str,
        device_id: str,
        country: str,
        city: str,
        app: str,
        is_external_share: int,
        share_scope: str,
        permission_change: int,
        bytes_value: int,
        intent: str,
        scenario_id: str,
    ) -> dict:
        nonlocal next_event_num
        event = {
            "event_id": f"evt_{next_event_num:08d}",
            "timestamp": ts,
            "user_id": user.user_id,
            "role": user.role,
            "department": user.department,
            "file_id": file_id,
            "file_type": file_type,
            "file_sensitivity": file_sensitivity,
            "action": action,
            "device_id": device_id,
            "country": country,
            "city": city,
            "app": app,
            "is_external_share": is_external_share,
            "share_scope": share_scope,
            "permission_change": permission_change,
            "bytes": bytes_value,
            "label": 1,
            "intent": intent,
            "scenario_id": scenario_id,
        }
        next_event_num += 1
        return event

    def new_unseen_context(user: UserProfile) -> tuple[str, str, str]:
        # choose device and location that do NOT belong to user’s common set
        attack_device = f"{user.user_id}_unknown_device"
        attack_country, attack_city = random.choice([("SG", "Singapore"), ("US", "Seattle"), ("GB", "London")])
        return attack_device, attack_country, attack_city

    # Exfiltration
    for i in range(attack_counts.get("exfiltration", 0)):
        user = random.choice([u for u in users if u.role != "it_admin"])
        ts0 = choose_attack_time()
        scenario_id = f"scenario_exfil_{i+1:03d}"
        device_id, country, city = new_unseen_context(user)
        app = "web"

        high_value = file_catalog[file_catalog["sensitivity"].isin(["medium", "high"])]
        chosen = high_value.sample(n=min(25, len(high_value)), replace=False)

        for j, (_, file_row) in enumerate(chosen.iterrows()):
            ts = ts0 + timedelta(seconds=j * 12)
            new_events.append(build_event(
                ts=ts,
                user=user,
                action="download",
                file_id=file_row["file_id"],
                file_type=file_row["file_type"],
                file_sensitivity=file_row["sensitivity"],
                device_id=device_id,
                country=country,
                city=city,
                app=app,
                is_external_share=0,
                share_scope="none",
                permission_change=0,
                bytes_value=int(np.random.randint(1_000_000, 8_000_000)),
                intent="exfiltration",
                scenario_id=scenario_id,
            ))

        for k in range(3):
            file_row = chosen.sample(1).iloc[0]
            ts = ts0 + timedelta(minutes=6, seconds=k * 20)
            new_events.append(build_event(
                ts=ts,
                user=user,
                action="share",
                file_id=file_row["file_id"],
                file_type=file_row["file_type"],
                file_sensitivity=file_row["sensitivity"],
                device_id=device_id,
                country=country,
                city=city,
                app=app,
                is_external_share=1,
                share_scope=random.choice(["external", "public"]),
                permission_change=0,
                bytes_value=int(np.random.randint(0, 2_000)),
                intent="exfiltration",
                scenario_id=scenario_id,
            ))

    # Recon
    for i in range(attack_counts.get("recon", 0)):
        user = random.choice([u for u in users if u.role != "it_admin"])
        ts0 = choose_attack_time()
        scenario_id = f"scenario_recon_{i+1:03d}"
        device_id, country, city = new_unseen_context(user)
        app = "web"

        chosen = file_catalog.sample(n=min(35, len(file_catalog)), replace=False)
        for j, (_, file_row) in enumerate(chosen.iterrows()):
            ts = ts0 + timedelta(seconds=j * 8)
            new_events.append(build_event(
                ts=ts,
                user=user,
                action="view",
                file_id=file_row["file_id"],
                file_type=file_row["file_type"],
                file_sensitivity=file_row["sensitivity"],
                device_id=device_id,
                country=country,
                city=city,
                app=app,
                is_external_share=0,
                share_scope="none",
                permission_change=0,
                bytes_value=int(np.random.randint(100, 2_000)),
                intent="recon",
                scenario_id=scenario_id,
            ))

    # Privilege misuse
    for i in range(attack_counts.get("privilege_misuse", 0)):
        user = random.choice(users)
        ts0 = choose_attack_time()
        scenario_id = f"scenario_priv_{i+1:03d}"
        device_id, country, city = new_unseen_context(user)
        app = "web"

        chosen = file_catalog.sample(n=min(8, len(file_catalog)), replace=False)
        for j, (_, file_row) in enumerate(chosen.iterrows()):
            ts = ts0 + timedelta(seconds=j * 30)
            new_events.append(build_event(
                ts=ts,
                user=user,
                action="permission_change",
                file_id=file_row["file_id"],
                file_type=file_row["file_type"],
                file_sensitivity=file_row["sensitivity"],
                device_id=device_id,
                country=country,
                city=city,
                app=app,
                is_external_share=0,
                share_scope="none",
                permission_change=1,
                bytes_value=int(np.random.randint(0, 500)),
                intent="privilege_misuse",
                scenario_id=scenario_id,
            ))

    # Tamper-like
    for i in range(attack_counts.get("tamper_like", 0)):
        user = random.choice([u for u in users if u.role != "manager"])
        ts0 = choose_attack_time()
        scenario_id = f"scenario_tamper_{i+1:03d}"
        device_id, country, city = new_unseen_context(user)
        app = "desktop"

        chosen = file_catalog.sample(n=min(18, len(file_catalog)), replace=False)
        for j, (_, file_row) in enumerate(chosen.iterrows()):
            ts = ts0 + timedelta(seconds=j * 10)
            action = random.choice(["rename", "delete", "edit"])
            new_events.append(build_event(
                ts=ts,
                user=user,
                action=action,
                file_id=file_row["file_id"],
                file_type=file_row["file_type"],
                file_sensitivity=file_row["sensitivity"],
                device_id=device_id,
                country=country,
                city=city,
                app=app,
                is_external_share=0,
                share_scope="none",
                permission_change=0,
                bytes_value=int(np.random.randint(0, 30_000)),
                intent="tamper_like",
                scenario_id=scenario_id,
            ))

    all_logs = pd.concat([logs, pd.DataFrame(new_events)], ignore_index=True)
    all_logs["timestamp"] = pd.to_datetime(all_logs["timestamp"])
    all_logs = all_logs.sort_values("timestamp").reset_index(drop=True)
    return all_logs