from __future__ import annotations

import logging
import os
import time as _time
from collections import defaultdict
from dataclasses import dataclass

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Room:
    room_id: int
    capacity: int
    room_type: str | None = None


@dataclass(frozen=True)
class TimeSlot:
    time_slot_id: int
    day_of_week: int
    start_minute: int
    end_minute: int


@dataclass(frozen=True)
class Session:
    session_id: int
    instructor_id: int | None
    group_ids: tuple[int, ...]
    student_count: int
    activity_type: str | None = None
    friday_available: bool = False
    online: bool = False
    course_id: int | None = None


@dataclass(frozen=True)
class Assignment:
    session_id: int
    room_id: int | None
    time_slot_id: int


class ScheduleInfeasibleError(ValueError):
    pass


class ScheduleTimeoutError(TimeoutError):
    pass


MAX_DAILY_SESSIONS = 4
MIN_DAILY_COURSES = 3
MAX_ATTENDANCE_DAYS = 5
# Total wall-clock budget for the whole call (model building + every solve).
MAX_SOLVE_SECONDS = float(os.getenv("TIMETABLE_MAX_SOLVE_SECONDS", "120"))
FRIDAY = 5
SATURDAY = 6


def is_seconded_lecture(session: Session) -> bool:
    return session.friday_available and session.activity_type == "lecture"


def _room_class_key(room: Room) -> tuple[int, str | None]:
    # Two rooms with the same capacity and type are interchangeable for every
    # session, so the model only decides *how many* rooms of a class are busy.
    return (
        room.capacity,
        room.room_type.casefold() if room.room_type is not None else None,
    )


def _room_fits(room_key: tuple[int, str | None], session: Session) -> bool:
    capacity, room_type = room_key
    return capacity >= session.student_count and (
        room_type is None
        or session.activity_type is None
        or room_type == session.activity_type.casefold()
    )


def _room_type_fits(room_key: tuple[int, str | None], session: Session) -> bool:
    _, room_type = room_key
    return (
        room_type is None
        or session.activity_type is None
        or room_type == session.activity_type.casefold()
    )


def _worker_count() -> int:
    configured = os.getenv("TIMETABLE_SOLVER_WORKERS")
    if configured and configured.isdigit() and int(configured) > 0:
        return int(configured)
    # 8 is the size of CP-SAT's default search portfolio; more rarely helps and
    # oversubscribes small containers that report the host's CPU count.
    return min(max(os.cpu_count() or 4, 4), 8)


def _make_solver(time_limit: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit, 1.0)
    try:
        solver.parameters.num_workers = _worker_count()
    except AttributeError:  # older OR-Tools
        solver.parameters.num_search_workers = _worker_count()
    solver.parameters.random_seed = 0
    if os.getenv("TIMETABLE_SOLVER_LOG") == "1":
        solver.parameters.log_search_progress = True
    return solver


def _assign_rooms(
    placements: list[tuple[Session, TimeSlot, TimeSlot | None, tuple[int, str | None]]],
    rooms_by_class: dict[tuple[int, str | None], list[Room]],
) -> dict[int, int]:
    """Give each session a concrete room.

    The model guarantees that, per room class, no more sessions are busy in any
    period than there are rooms. Interval graphs are perfect, so a greedy sweep
    by start time always finds a conflict-free assignment.
    """
    by_class: dict[tuple[int, str | None], list[tuple[int, int, int]]] = defaultdict(list)
    for session, slot, second_slot, class_key in placements:
        base = slot.day_of_week * 1440
        end_slot = second_slot or slot
        by_class[class_key].append(
            (base + slot.start_minute, base + end_slot.end_minute, session.session_id)
        )

    room_for_session: dict[int, int] = {}
    for class_key, items in by_class.items():
        rooms = rooms_by_class[class_key]
        free_at = {room.room_id: -1 for room in rooms}
        for start, end, session_id in sorted(items):
            for room in rooms:
                if free_at[room.room_id] <= start:
                    free_at[room.room_id] = end
                    room_for_session[session_id] = room.room_id
                    break
            else:  # pragma: no cover - the model makes this unreachable
                raise ScheduleInfeasibleError(
                    "Could not assign rooms to the generated timetable."
                )
    return room_for_session


def generate_schedule(
    sessions: list[Session], rooms: list[Room], time_slots: list[TimeSlot]
) -> list[Assignment]:
    """Assign each session a compatible room and time without resource conflicts."""
    if not sessions:
        return []

    started = _time.monotonic()
    deadline = started + MAX_SOLVE_SECONDS

    slots_by_day: dict[int, list[TimeSlot]] = {}
    for slot in time_slots:
        slots_by_day.setdefault(slot.day_of_week, []).append(slot)
    for day_slots in slots_by_day.values():
        day_slots.sort(key=lambda slot: (slot.start_minute, slot.end_minute))
    next_slot_by_id = {
        first.time_slot_id: second
        for day_slots in slots_by_day.values()
        for first, second in zip(day_slots, day_slots[1:])
    }

    def course_of(session: Session) -> int:
        return session.course_id if session.course_id is not None else session.session_id

    # ------------------------------------------------------------------
    # Eligibility (identical rules to the previous implementation)
    # ------------------------------------------------------------------
    slots_by_session: dict[int, list[TimeSlot]] = {}
    second_slots_by_session: dict[int, dict[int, TimeSlot]] = {}
    for session in sessions:
        eligible_slots = [
            slot
            for slot in time_slots
            if (
                slot.day_of_week in (FRIDAY, SATURDAY)
                if is_seconded_lecture(session) and not session.online
                else slot.day_of_week != FRIDAY or session.friday_available
            )
        ]
        second_slots = (
            {
                slot.time_slot_id: next_slot_by_id[slot.time_slot_id]
                for slot in eligible_slots
                if slot.time_slot_id in next_slot_by_id
            }
            if is_seconded_lecture(session) and not session.online
            else {}
        )
        if is_seconded_lecture(session) and not session.online:
            eligible_slots = [
                slot for slot in eligible_slots if slot.time_slot_id in second_slots
            ]
        if not eligible_slots:
            detail = (
                f"Session {session.session_id} has no eligible teaching slots; "
                "Friday is available only to seconded instructors."
            )
            if is_seconded_lecture(session) and not session.online:
                detail += (
                    " Offline seconded lectures must be on Friday or Saturday "
                    "and need two consecutive periods."
                )
            raise ScheduleInfeasibleError(detail)
        slots_by_session[session.session_id] = eligible_slots
        second_slots_by_session[session.session_id] = second_slots

    # Room classes: rooms with identical (capacity, type) are interchangeable.
    rooms_by_class: dict[tuple[int, str | None], list[Room]] = defaultdict(list)
    for room in rooms:
        rooms_by_class[_room_class_key(room)].append(room)
    classes_by_session: dict[int, list[tuple[int, str | None]]] = {}
    for session in sessions:
        if session.online:
            classes_by_session[session.session_id] = []
            continue
        compatible = [key for key in rooms_by_class if _room_fits(key, session)]
        if not compatible:
            type_compatible = [
                key for key in rooms_by_class if _room_type_fits(key, session)
            ]
            if type_compatible:
                largest_capacity = max(key[0] for key in type_compatible)
                raise ScheduleInfeasibleError(
                    f"Session {session.session_id} ({session.activity_type or 'activity'}, "
                    f"{session.student_count} students) needs a compatible room with "
                    f"capacity >= {session.student_count}; the largest compatible room "
                    f"has capacity {largest_capacity}."
                )
            raise ScheduleInfeasibleError(
                f"Session {session.session_id} ({session.activity_type or 'activity'}) "
                "has no room of a compatible type."
            )
        classes_by_session[session.session_id] = compatible

    # Periods a session occupies when it starts in `slot` (the slot itself,
    # plus the following one for two-period lectures).
    def occupied_ids(session: Session, slot: TimeSlot) -> tuple[int, ...]:
        second = second_slots_by_session[session.session_id].get(slot.time_slot_id)
        if second is None:
            return (slot.time_slot_id,)
        return (slot.time_slot_id, second.time_slot_id)

    # ------------------------------------------------------------------
    # Indexes (built once instead of re-scanning every session per group/day)
    # ------------------------------------------------------------------
    sessions_by_group: dict[int, list[Session]] = defaultdict(list)
    weekly_group_courses: dict[int, set[int]] = defaultdict(set)
    for session in sessions:
        for group_id in set(session.group_ids):
            sessions_by_group[group_id].append(session)
            weekly_group_courses[group_id].add(course_of(session))

    model = cp_model.CpModel()
    slot_variables: dict[int, dict[int, cp_model.IntVar]] = {}
    # (session_id, slot_id) -> [(class_key, var)]
    room_variables: dict[tuple[int, int], list[tuple[tuple[int, str | None], cp_model.IntVar]]] = {}
    group_slot_variables: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)
    instructor_slot_variables: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)
    class_slot_variables: dict[tuple[tuple[int, str | None], int], list[cp_model.IntVar]] = defaultdict(list)
    group_day_variables: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)
    group_course_day_variables: dict[tuple[int, int, int], list[cp_model.IntVar]] = defaultdict(list)
    instructor_day_variables: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)

    for session in sessions:
        session_id = session.session_id
        course_id = course_of(session)
        slot_variables[session_id] = {}
        choices = []
        for slot in slots_by_session[session_id]:
            selected = model.NewBoolVar("")
            slot_variables[session_id][slot.time_slot_id] = selected
            choices.append(selected)
            covered = occupied_ids(session, slot)
            day = slot.day_of_week

            if session.instructor_id is not None:
                instructor_day_variables[(session.instructor_id, day)].append(selected)
                for slot_id in covered:
                    instructor_slot_variables[(session.instructor_id, slot_id)].append(selected)
            for group_id in set(session.group_ids):
                group_day_variables[(group_id, day)].append(selected)
                group_course_day_variables[(group_id, course_id, day)].append(selected)
                for slot_id in covered:
                    group_slot_variables[(group_id, slot_id)].append(selected)

            if not session.online:
                options = []
                for class_key in classes_by_session[session_id]:
                    room_var = model.NewBoolVar("")
                    options.append(room_var)
                    room_variables.setdefault((session_id, slot.time_slot_id), []).append(
                        (class_key, room_var)
                    )
                    for slot_id in covered:
                        class_slot_variables[(class_key, slot_id)].append(room_var)
                # Picking this period means picking exactly one room class.
                model.Add(sum(options) == selected)
        model.AddExactlyOne(choices)

    # ------------------------------------------------------------------
    # Resource conflicts: at most one session per period (booleans only)
    # ------------------------------------------------------------------
    for variables in group_slot_variables.values():
        if len(variables) > 1:
            model.AddAtMostOne(variables)
    for variables in instructor_slot_variables.values():
        if len(variables) > 1:
            model.AddAtMostOne(variables)
    for (class_key, _), variables in class_slot_variables.items():
        room_count = len(rooms_by_class[class_key])
        if len(variables) > room_count:
            if room_count == 1:
                model.AddAtMostOne(variables)
            else:
                model.Add(sum(variables) <= room_count)
    for variables in instructor_day_variables.values():
        if len(variables) > MAX_DAILY_SESSIONS:
            model.Add(sum(variables) <= MAX_DAILY_SESSIONS)

    # ------------------------------------------------------------------
    # Student-group daily rules
    # ------------------------------------------------------------------
    for group_id in sessions_by_group:
        group_attendance_days = []
        group_courses = weekly_group_courses[group_id]
        for day, day_slots in slots_by_day.items():
            daily_session_variables = group_day_variables.get((group_id, day))
            if not daily_session_variables:
                continue

            # present <=> at least one session on this day (as clauses)
            present = model.NewBoolVar("")
            for variable in daily_session_variables:
                model.AddImplication(variable, present)
            model.AddBoolOr([present.Not(), *daily_session_variables])
            if len(daily_session_variables) > MAX_DAILY_SESSIONS:
                model.Add(sum(daily_session_variables) <= MAX_DAILY_SESSIONS)
            group_attendance_days.append(present)

            # active[slot] <=> the group is busy in that period
            active_by_slot: list[cp_model.IntVar] = []
            for slot in day_slots:
                possible = group_slot_variables.get((group_id, slot.time_slot_id))
                active = model.NewBoolVar("")
                if possible:
                    for variable in possible:
                        model.AddImplication(variable, active)
                    model.AddBoolOr([active.Not(), *possible])
                else:
                    model.Add(active == 0)
                active_by_slot.append(active)

            # No idle gaps inside a day: busy, free, busy is forbidden.
            for first in range(len(active_by_slot)):
                for last in range(first + 2, len(active_by_slot)):
                    for missing in range(first + 1, last):
                        model.AddBoolOr(
                            [
                                active_by_slot[first].Not(),
                                active_by_slot[last].Not(),
                                active_by_slot[missing],
                            ]
                        )

            # Minimum number of distinct courses on a attended day.
            if day != FRIDAY and len(group_courses) >= MIN_DAILY_COURSES:
                daily_course_variables = []
                for course_id in group_courses:
                    course_variables = group_course_day_variables.get(
                        (group_id, course_id, day)
                    )
                    if not course_variables:
                        continue
                    course_present = model.NewBoolVar("")
                    for variable in course_variables:
                        model.AddImplication(variable, course_present)
                    model.AddBoolOr([course_present.Not(), *course_variables])
                    daily_course_variables.append(course_present)
                model.Add(sum(daily_course_variables) >= MIN_DAILY_COURSES).OnlyEnforceIf(
                    present
                )

        if group_attendance_days:
            model.Add(sum(group_attendance_days) <= MAX_ATTENDANCE_DAYS)

    built = _time.monotonic()
    logger.info(
        "timetable model built in %.2fs (%d sessions)", built - started, len(sessions)
    )

    def timeout_error() -> ScheduleTimeoutError:
        return ScheduleTimeoutError(
            f"Could not find a valid timetable within {MAX_SOLVE_SECONDS:g} seconds. "
            "Reduce the number of sessions or add more eligible rooms and time slots, "
            "then try again."
        )

    remaining = deadline - built
    if remaining <= 0:
        raise timeout_error()

    # Seconded lectures may go on Friday or Saturday freely: single solve.
    solver = _make_solver(remaining)
    status = solver.Solve(model)
    logger.info("timetable solve: %s in %.2fs", solver.StatusName(status), solver.WallTime())

    if status == cp_model.UNKNOWN:
        raise timeout_error()
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        detail = "No valid timetable exists for these sessions, rooms, and time slots."
        if any(is_seconded_lecture(session) for session in sessions):
            detail += (
                " Seconded lectures are restricted to Friday or Saturday, and "
                "offline lectures need two consecutive periods."
            )
        raise ScheduleInfeasibleError(detail)

    # ------------------------------------------------------------------
    # Read the solution, then turn room classes into concrete rooms
    # ------------------------------------------------------------------
    chosen_slot: dict[int, TimeSlot] = {}
    placements = []
    for session in sessions:
        slot = next(
            slot
            for slot in slots_by_session[session.session_id]
            if solver.BooleanValue(slot_variables[session.session_id][slot.time_slot_id])
        )
        chosen_slot[session.session_id] = slot
        if session.online:
            continue
        class_key = next(
            key
            for key, variable in room_variables[(session.session_id, slot.time_slot_id)]
            if solver.BooleanValue(variable)
        )
        second = second_slots_by_session[session.session_id].get(slot.time_slot_id)
        placements.append((session, slot, second, class_key))

    room_for_session = _assign_rooms(placements, rooms_by_class)
    logger.info("timetable total %.2fs", _time.monotonic() - started)
    return [
        Assignment(
            session.session_id,
            room_for_session.get(session.session_id),
            chosen_slot[session.session_id].time_slot_id,
        )
        for session in sessions
    ]
