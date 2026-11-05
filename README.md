# בוט ידע ארגוני (RAG) – LlamaIndex + ChromaDB + Gemini

## התקנה
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
## הגדרת .env
העתיקו `.env.example` ל-`.env` ומלאו `GOOGLE_API_KEY` (מפתח חינמי מ-Google AI Studio).
שמות המודלים (`EMBED_MODEL`, `LLM_MODEL`) ניתנים לעדכון שם בלי לשנות קוד; ודאו בתיעוד של Google את השמות הנוכחיים.

## בניית אינדקס
```
python build_index.py            # בונה רק אם אין אינדקס
python build_index.py --rebuild  # אחרי שינוי במסמכים או בחלוקה לצ'אנקים
```
## בדיקות וכיוון
```
python eval_questions.py
```
הסקריפט מדפיס לכל שאלה את הצ'אנקים והציונים (ציון = exp(-מרחק קוסינוס), גבוה = דומה יותר),
ומציע `SIMILARITY_CUTOFF` לפי הפער בין ציוני שאלות רלוונטיות לשאלות ללא תשובה.
עדכנו את הערך ב-`config.py`. אותו קובץ מכיל גם `TOP_K` (4) ו-`TOP_K_COMPARE` (6, לכל רמה).

## הרצה
```
streamlit run app.py
```
`app.py` קורא ל-`rag_engine.ask(question, history)`.

## הערות
- הוסיפו כל קובץ מסמך גם ל-`LEVEL_BY_FILE` ב-`config.py`.
- מגבלת קצב: ה-embedding נעשה בקבוצות של 10 עם השהיה ו-retry.
