import os

from dotenv import load_dotenv
import psycopg

load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    print("DUPLICATES")
    rows = conn.execute(
        """
        SELECT course_id, academic_term_id, group_number, COUNT(*),
               array_agg(id), array_agg(offering_id)
        FROM student_groups
        GROUP BY course_id, academic_term_id, group_number
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 15
        """
    ).fetchall()
    print("count", len(rows))
    for row in rows:
        print(row)

    print("GROUP SAMPLE")
    rows = conn.execute(
        """
        SELECT id, offering_id, course_id, academic_term_id, group_number, student_count
        FROM student_groups
        ORDER BY id DESC
        LIMIT 8
        """
    ).fetchall()
    for row in rows:
        print(row)

    print("NULL OFFERING", conn.execute(
        "SELECT COUNT(*) FROM student_groups WHERE offering_id IS NULL"
    ).fetchone())
