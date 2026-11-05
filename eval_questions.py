"""בדיקת איכות האחזור וכיוון SIMILARITY_CUTOFF. לא קורא למודל התשובה.
הרצה: python eval_questions.py
"""
import chromadb

import config
from rag_engine import RagError, analyze, retrieve

# expect: רשימת (רמה, תת-מחרוזת של שם הסעיף); None = רמה כלשהי
# answerable=False: אין תשובה במסמכים. clarify=True: מצופה שאלת הבהרה.
HIST = [{"role": "user", "content": "כמה עולה לוגו פשוט למתחילה מאוד?"},
        {"role": "assistant", "content": "לוגו פשוט למתחילה מאוד עולה 100 עד 300 ש\"ח."}]

QUESTIONS = [
    {"q": "כמה עולה יצירת תמונת AI אחת למתחילה מאוד?", "expect": [("very_beginner", "שילוב AI")]},
    {"q": "מה המחיר של לוגו פשוט ברמת מתחילה מאוד?", "expect": [("very_beginner", "רגילות")]},
    {"q": "כמה עולה עריכת ריל לטיקטוק?", "expect": [(None, "רגילות")]},
    {"q": "כמה לגבות לשעה כמתחילה מאוד?", "expect": [("very_beginner", "מחיר לשעה")]},
    {"q": "מה ההבדל במחיר תמונת AI בין מתחילה מאוד למתקדמת?",
     "expect": [("very_beginner", "הפער"), ("advanced", "הפער")]},
    {"q": "השווה את מחיר חבילת תוכן בין כל הרמות", "expect": [(None, "הפער")]},
    {"q": "איזה טיפים יש למתחילה מאוד?", "expect": [("very_beginner", "טיפים")]},
    {"q": "אילו טעויות נפוצות יש בתמחור?", "expect": [(None, "טעויות")]},
    {"q": "תן לי דוגמה להצעת מחיר", "expect": [(None, "דוגמה")]},
    {"q": "כמה מקדמה כדאי לבקש מלקוח?", "expect": [(None, "טיפים")]},
    {"q": "מי נחשבת מתחילה מאוד?", "expect": [("very_beginner", "מי נחשבת")]},
    {"q": "מה שעות הפעילות של המשרד?", "answerable": False},
    {"q": "איך מכינים שקשוקה?", "answerable": False},
    {"q": "כמה עולה לבנות אתר אינטרנט?", "answerable": False},
    {"q": "ומה לגבי מתקדמת?", "history": HIST, "expect": [("advanced", "רגילות")]},
    {"q": "מחירים", "clarify": True},
]


def indexed_levels():
    col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION_NAME)
    return {m["level"] for m in col.get(include=["metadatas"])["metadatas"]}


def main():
    try:
        have = indexed_levels()
    except Exception:
        raise SystemExit("האינדקס לא קיים. הריצו: python build_index.py")
    print(f"רמות באינדקס: {sorted(have)}\n")
    hits = total = 0
    good_tops, bad_tops = [], []
    try:
        for item in QUESTIONS:
            q = item["q"]
            plan = analyze(q, item.get("history"))
            print("=" * 70)
            print(f"שאלה: {q}")
            if item.get("clarify"):
                print("  ->", "שאלת הבהרה (תקין)" if plan.clarify else "נכשל: לא נשאלה הבהרה")
                continue
            if plan.clarify:
                print("  -> נשאלה הבהרה")
                continue
            print(f"  חיפוש: {plan.search_query!r} רמות={plan.levels} k={plan.k} השוואה={plan.compare}")
            chunks = retrieve(plan, apply_cutoff=False)
            for c in chunks:
                print(f"  {c['score']:.3f}  [{c['level']}] {c['section']}")
            top = chunks[0]["score"] if chunks else 0.0
            if item.get("answerable") is False:
                bad_tops.append(top)
                print(f"  (אין תשובה במסמכים) ציון עליון={top:.3f}")
                continue
            good_tops.append(top)
            expect = [(l, s) for l, s in item["expect"] if l is None or l in have]
            if not expect:
                print("  דילוג: הרמה הצפויה לא באינדקס")
                continue
            total += 1
            ok = all(any((l is None or c["level"] == l) and s in c["section"] for c in chunks[:plan.k * max(1, len(plan.levels))])
                     for l, s in expect)
            hits += ok
            print("  צ'אנק צפוי בין ה-k הראשונים:", "כן" if ok else "לא")
    except RagError as e:
        raise SystemExit(f"שגיאה: {e}")

    print("\n" + "=" * 70)
    print(f"אחזור נכון: {hits}/{total}")
    if good_tops and bad_tops:
        lo, hi = min(good_tops), max(bad_tops)
        print(f"ציון עליון נמוך ביותר בשאלות רלוונטיות: {lo:.3f}")
        print(f"ציון עליון גבוה ביותר בשאלות ללא תשובה:  {hi:.3f}")
        if lo > hi:
            print(f"הפער נקי. מומלץ SIMILARITY_CUTOFF ≈ {(lo + hi) / 2:.3f} (כרגע {config.SIMILARITY_CUTOFF})")
        else:
            print("אין הפרדה נקייה; בחרו סף שמעדיף פחות 'לא נמצא' שגוי, "
                  f"למשל קרוב ל-{hi:.3f}, ובדקו שאלות גבוליות ידנית.")


if __name__ == "__main__":
    main()
