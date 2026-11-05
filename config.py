"""קבועים והגדרות של הפרויקט. כאן מכוונים את הסף, k ושמות המודלים."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ---------- מפתח ומודלים (נקראים מ-.env) ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# שם מודל השיחה משתנה לעיתים קרובות - ודאו בתיעוד של Google ועדכנו ב-.env
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ---------- נתיבים ו-Chroma ----------
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "org_knowledge"

# ---------- רמות ----------
# שם קובץ -> מזהה רמה
LEVEL_BY_FILE = {
    "pricing_very_beginner.md": "very_beginner",
    "pricing_beginner.md": "beginner",
    "pricing_advanced.md": "advanced",
    "pricing_expert.md": "expert",
}
# מזהה רמה -> שם בעברית (לשורת ההקשר ולהצגה)
LEVEL_LABELS = {
    "very_beginner": "מתחילה מאוד",
    "beginner": "מתחילה",
    "advanced": "מתקדמת",
    "expert": "מנוסה מאוד",
}

# ---------- חלוקה לצ'אנקים (במילים) ----------
MAX_CHUNK_WORDS = 450   # סעיף ארוך מזה מחולק
CHUNK_OVERLAP_WORDS = 50
MIN_SECTION_WORDS = 50  # סעיף קצר מזה מאוחד עם סמוך
MERGE_MAX_WORDS = 130   # תקרה לצ'אנק מאוחד, כדי שלא יצטברו נושאים לא קשורים

# ---------- אחזור ----------
TOP_K = 4            # שאלה רגילה
TOP_K_COMPARE = 6    # שאלת השוואה (לכל רמה שהוזכרה)
# ציון = exp(-מרחק קוסינוס) כפי ש-LlamaIndex ממיר. גבוה = דומה יותר.
# הסף חל על הציון הגבוה ביותר בלבד (בהשוואה: לכל רמה בנפרד). מתחתיו אין קריאה ל-LLM.
SIMILARITY_CUTOFF = 0.75
# אם הסף עבר, מועברים למודל כל ה-k הצ'אנקים שמעל הרצפה הזו
SIMILARITY_FLOOR = 0.70

NOT_FOUND_MESSAGE = "לא נמצא מידע על כך במסמכים שלנו"
CLARIFY_MESSAGE = "לאיזו רמה ולאיזה סוג עבודה התכוונת?"

# ---------- הגבלות קצב (שכבה חינמית) ----------
EMBED_BATCH_SIZE = 10        # כמה צ'אנקים בכל קריאת embedding
BATCH_PAUSE_SECONDS = 2.0    # השהיה בין קבוצות
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 8.0       # שניות; מוכפל בכל ניסיון
