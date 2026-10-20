"""SQLite 数据库模块：保存学习记录与分享链接。"""
import os
import json
import uuid
import sqlite3
import datetime
from typing import List, Dict, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "learning.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                words TEXT NOT NULL,
                articles TEXT NOT NULL,
                translations TEXT,
                questions TEXT,
                coverage REAL,
                matched_words TEXT,
                missing_words TEXT,
                audio_paths TEXT,
                demo_mode INTEGER DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS share_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                record_id INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                answers TEXT NOT NULL,
                correct_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                accuracy REAL NOT NULL
            )
        """)
        # 旧库迁移：records 表补充 images 列
        cols = [r[1] for r in c.execute("PRAGMA table_info(records)").fetchall()]
        if "images" not in cols:
            c.execute("ALTER TABLE records ADD COLUMN images TEXT")
        c.commit()


def save_record(result: Dict) -> int:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO records
               (created_at, words, articles, translations, questions, coverage,
                matched_words, missing_words, audio_paths, demo_mode, images)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                json.dumps(result.get("words", []), ensure_ascii=False),
                json.dumps(result.get("articles", []), ensure_ascii=False),
                json.dumps(result.get("translations", []), ensure_ascii=False),
                json.dumps(result.get("questions", []), ensure_ascii=False),
                result.get("coverage", 0.0),
                json.dumps(result.get("matched_words", []), ensure_ascii=False),
                json.dumps(result.get("missing_words", []), ensure_ascii=False),
                json.dumps(result.get("audio_paths", []), ensure_ascii=False),
                1 if result.get("demo_mode") else 0,
                json.dumps(result.get("images", []), ensure_ascii=False),
            ),
        )
        c.commit()
        return cur.lastrowid


def list_records() -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, created_at, words, coverage, matched_words, missing_words, demo_mode "
            "FROM records ORDER BY id DESC"
        ).fetchall()
    out = []
    for r in rows:
        item = {
            "id": r["id"],
            "created_at": r["created_at"],
            "words": json.loads(r["words"] or "[]"),
            "coverage": r["coverage"],
            "matched_words": json.loads(r["matched_words"] or "[]"),
            "missing_words": json.loads(r["missing_words"] or "[]"),
            "demo_mode": bool(r["demo_mode"]),
        }
        # 获取最新答题记录
        attempt = get_latest_quiz_attempt(r["id"])
        item["accuracy"] = attempt["accuracy"] if attempt else None
        item["correct_count"] = attempt["correct_count"] if attempt else 0
        item["total_count"] = attempt["total_count"] if attempt else 0
        out.append(item)
    return out


def get_record(record_id: int) -> Optional[Dict]:
    with _conn() as c:
        r = c.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "words": json.loads(r["words"] or "[]"),
        "articles": json.loads(r["articles"] or "[]"),
        "translations": json.loads(r["translations"] or "[]"),
        "questions": json.loads(r["questions"] or "[]"),
        "coverage": r["coverage"],
        "matched_words": json.loads(r["matched_words"] or "[]"),
        "missing_words": json.loads(r["missing_words"] or "[]"),
        "audio_paths": json.loads(r["audio_paths"] or "[]"),
        "images": json.loads(r["images"] or "[]"),
        "demo_mode": bool(r["demo_mode"]),
    }


def delete_record(record_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM records WHERE id = ?", (record_id,))
        c.commit()
        return cur.rowcount > 0


# ---------- 分享链接 ----------
def create_share_token(record_id: int) -> str:
    token = uuid.uuid4().hex[:16]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        c.execute(
            "INSERT INTO share_links (token, record_id, created_at) VALUES (?, ?, ?)",
            (token, record_id, now),
        )
        c.commit()
    return token


def get_record_by_share_token(token: str) -> Optional[Dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT record_id FROM share_links WHERE token = ?", (token,)
        ).fetchone()
    if not row:
        return None
    return get_record(row["record_id"])


# ---------- 答题记录 ----------
def save_quiz_attempt(record_id: int, answers: List[int], correct_count: int, total_count: int) -> int:
    """保存一次答题记录，answers 为各题选中的选项索引列表（-1 表示未作答）。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    accuracy = round(correct_count / total_count * 100, 2) if total_count > 0 else 0.0
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO quiz_attempts (record_id, created_at, answers, correct_count, total_count, accuracy)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (record_id, now, json.dumps(answers), correct_count, total_count, accuracy),
        )
        c.commit()
        return cur.lastrowid


def get_latest_quiz_attempt(record_id: int) -> Optional[Dict]:
    """获取某条记录的最新一次答题记录。"""
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM quiz_attempts WHERE record_id = ? ORDER BY id DESC LIMIT 1",
            (record_id,),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "record_id": r["record_id"],
        "created_at": r["created_at"],
        "answers": json.loads(r["answers"] or "[]"),
        "correct_count": r["correct_count"],
        "total_count": r["total_count"],
        "accuracy": r["accuracy"],
    }
