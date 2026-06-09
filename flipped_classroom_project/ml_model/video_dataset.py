"""
============================================================
 FlipLearn – Video Watch Dataset Updater
 Author  : Uttam Vitthal Bhise

 SCHEMA (video_dataset.csv)
 ───────────────────────────
 One row per student per video watched.

   record_id          – DB_<student_id>_<video_id>
   usn                – student roll number
   student_name       – full name
   subject_code       – subject code (DS, PY, …)
   subject_name       – subject full name
   video_title        – video title
   video_duration_min – video total duration (VideoLecture.duration_minutes)
   watch_duration_min – how long the student actually watched
   completed          – True / False
   completion_pct     – watch_duration / video_duration * 100 (capped at 100)
   watched_at         – ISO timestamp of first watch
   engagement_label   – Full / Partial / Minimal based on completion_pct

 Thread safety: threading.Lock protects CSV I/O.
============================================================
"""

import pathlib
import threading
import logging

import pandas as pd

logger = logging.getLogger('fliplearn.video_dataset')

ML_DIR    = pathlib.Path(__file__).parent
VIDEO_CSV = ML_DIR / 'video_dataset.csv'

_COLUMNS = [
    'record_id', 'usn', 'student_name',
    'subject_code', 'subject_name', 'video_title',
    'video_duration_min', 'watch_duration_min',
    'completed', 'completion_pct', 'watched_at', 'engagement_label',
]

_lock = threading.Lock()


# ── Helpers ─────────────────────────────────────────────────────

def _engagement_label(completion_pct: float) -> str:
    if completion_pct >= 90: return 'Full'
    if completion_pct >= 50: return 'Partial'
    return 'Minimal'


def _read() -> pd.DataFrame:
    if VIDEO_CSV.exists():
        try:
            return pd.read_csv(VIDEO_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=_COLUMNS)


def _write(df: pd.DataFrame) -> None:
    VIDEO_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = VIDEO_CSV.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(VIDEO_CSV)


# ── Core: upsert one video watch record ─────────────────────────

def upsert_video_row(history) -> bool:
    """
    Insert or update a row in video_dataset.csv for a VideoWatchHistory instance.
    Returns True if a new row was inserted.
    """
    try:
        student = history.student
        video   = history.video
        subject = video.subject

        try:
            usn = student.student_profile.roll_number
        except Exception:
            usn = f'USN_{student.id}'

        student_name   = student.get_full_name().strip() or student.username
        video_duration = float(video.duration_minutes) if video.duration_minutes else 0.0
        watch_duration = float(history.watch_duration_minutes)

        if video_duration > 0:
            completion_pct = round(min(100.0, watch_duration / video_duration * 100), 1)
        else:
            completion_pct = 100.0 if history.completed else 0.0

        row = {
            'record_id':          f'DB_{student.id}_{video.id}',
            'usn':                usn,
            'student_name':       student_name,
            'subject_code':       subject.code,
            'subject_name':       subject.name,
            'video_title':        video.title,
            'video_duration_min': round(video_duration, 1),
            'watch_duration_min': round(watch_duration, 1),
            'completed':          bool(history.completed),
            'completion_pct':     completion_pct,
            'watched_at':         str(history.watched_at),
            'engagement_label':   _engagement_label(completion_pct),
        }

        key    = str(row['record_id'])
        is_new = False

        with _lock:
            df = _read()
            if 'record_id' in df.columns and key in df['record_id'].astype(str).values:
                idx = df.index[df['record_id'].astype(str) == key][0]
                for col, val in row.items():
                    df.at[idx, col] = val
                logger.info(f'[VideoDS] Updated {key}')
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                logger.info(f'[VideoDS] Inserted {key} (total={len(df)})')
                is_new = True
            _write(df)

        return is_new

    except Exception as e:
        logger.error(f'[VideoDS] upsert_video_row error: {e}')
        return False


# ── Bulk seed from DB ─────────────────────────────────────────────

def seed_from_db() -> int:
    """
    Pull all VideoWatchHistory records from the DB and write video_dataset.csv.
    Safe to call repeatedly (upserts, not duplicates).
    Returns number of rows written.
    """
    try:
        from flipped_app.models import VideoWatchHistory
        histories = VideoWatchHistory.objects.select_related(
            'student', 'student__student_profile',
            'video', 'video__subject'
        ).all()
        count = 0
        for history in histories:
            upsert_video_row(history)
            count += 1
        logger.info(f'[VideoDS] Seeded {count} rows from DB.')
        return count
    except Exception as e:
        logger.error(f'[VideoDS] seed_from_db error: {e}')
        return 0
