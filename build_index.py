"""בניית האינדקס: טעינת קבצי Markdown, חלוקה לצ'אנקים והכנסה ל-ChromaDB.

הרצה:  python build_index.py            (בונה רק אם האינדקס ריק)
       python build_index.py --rebuild  (מוחק ובונה מחדש)
"""
import re
import sys
import time

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from rag_engine import RagError, init_models, with_retry

HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$")


def words(text: str) -> int:
    return len(text.split())


# ---------- שלב 1: פיצול לפי כותרות ## ו-### ----------
def split_sections(md: str):
    """מחזיר (כותרת_מסמך, [{'title':..., 'body':...}]). כותרת # היא שם המסמך."""
    doc_title, sections, cur = "", [], None
    in_code = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            cur = {"title": m.group(2).strip(), "lines": [line]}
            sections.append(cur)
        elif not in_code and line.startswith("# ") and not doc_title:
            doc_title = line[2:].strip()
        elif cur is not None:
            cur["lines"].append(line)
    for s in sections:
        s["body"] = "\n".join(s.pop("lines")).strip()
    return doc_title, sections


# ---------- שלב 2: איחוד סעיפים קצרים ----------
def merge_short(sections):
    merged = []
    pending = None  # סעיף קצר בתחילת הרשימה שממתין לסעיף הבא
    for s in sections:
        if pending:
            s = {"title": f"{pending['title']} / {s['title']}",
                 "body": pending["body"] + "\n\n" + s["body"]}
            pending = None
        if words(s["body"]) < config.MIN_SECTION_WORDS:
            if merged and words(merged[-1]["body"]) + words(s["body"]) <= config.MERGE_MAX_WORDS:  # מאחדים עם הקודם
                prev = merged[-1]
                prev["title"] += f" / {s['title']}"
                prev["body"] += "\n\n" + s["body"]
            elif merged:
                merged.append(s)  # האיחוד היה גדול מדי, נשאר לבד
            else:
                pending = s
        else:
            merged.append(s)
    if pending:
        merged.append(pending)
    return merged


# ---------- שלב 3: חלוקת סעיף ארוך (טבלה אף פעם לא נחתכת) ----------
def to_blocks(body: str):
    """מפרק גוף סעיף לבלוקים: טבלה (שורות שמתחילות ב-|) או טקסט."""
    blocks, buf, is_table = [], [], False
    for line in body.splitlines():
        t = line.strip().startswith("|")
        if buf and t != is_table:
            blocks.append(("table" if is_table else "text", "\n".join(buf)))
            buf = []
        is_table = t
        buf.append(line)
    if buf:
        blocks.append(("table" if is_table else "text", "\n".join(buf)))
    return blocks


def split_long(section):
    """מחזיר רשימת גופי טקסט לצ'אנקים; כל חלק מקבל שוב את שורת הכותרת."""
    body = section["body"]
    if words(body) <= config.MAX_CHUNK_WORDS:
        return [body]
    heading = body.splitlines()[0] if HEADING_RE.match(body.splitlines()[0]) else ""
    # SentenceSplitter שסופר מילים ולא טוקנים
    splitter = SentenceSplitter(
        chunk_size=config.MAX_CHUNK_WORDS,
        chunk_overlap=config.CHUNK_OVERLAP_WORDS,
        tokenizer=lambda t: t.split(),
    )
    pieces = []
    for kind, text in to_blocks(body):
        if kind == "table" or words(text) <= config.MAX_CHUNK_WORDS:
            pieces.append(text)  # טבלה נשארת שלמה
        else:
            pieces.extend(splitter.split_text(text))
    chunks, cur = [], ""
    for p in pieces:
        if cur and words(cur) + words(p) > config.MAX_CHUNK_WORDS:
            chunks.append(cur)
            cur = heading + "\n" + p if heading and not p.startswith(heading) else p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ---------- בניית צמתים ----------
def build_nodes():
    nodes = []
    files = sorted(config.DOCS_DIR.glob("*.md"))
    if not files:
        raise RagError(f"לא נמצאו קבצי Markdown בתיקייה {config.DOCS_DIR}")
    for path in files:
        level = config.LEVEL_BY_FILE.get(path.name)
        if level is None:
            print(f"אזהרה: {path.name} לא מופיע ב-LEVEL_BY_FILE, מדלגים.")
            continue
        title, sections = split_sections(path.read_text(encoding="utf-8"))
        doc_name = re.split(r"\s[–-]\s", title)[0].strip() or path.stem
        label = config.LEVEL_LABELS[level]
        for sec in merge_short(sections):
            for part in split_long(sec):
                ctx = f"מסמך: {doc_name}, רמת {label} | סעיף: {sec['title']}"
                node = TextNode(
                    text=f"{ctx}\n\n{part}",
                    metadata={"doc_name": doc_name, "level": level,
                              "section": sec["title"]},
                )
                # ההקשר כבר בטקסט, אין צורך להכניס את המטא-דאטה פעם נוספת
                node.excluded_embed_metadata_keys = list(node.metadata)
                node.excluded_llm_metadata_keys = list(node.metadata)
                nodes.append(node)
        print(f"{path.name}: רמה={level}")
    return nodes


def main():
    rebuild = "--rebuild" in sys.argv
    init_models()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME in existing:
        if not rebuild:
            n = client.get_collection(config.COLLECTION_NAME).count()
            if n > 0:
                print(f"האינדקס כבר קיים ({n} צ'אנקים). להרצה מחדש: --rebuild")
                return
        client.delete_collection(config.COLLECTION_NAME)

    # חובה: מדד cosine (ברירת המחדל של Chroma היא L2)
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex(
        nodes=[], storage_context=StorageContext.from_defaults(vector_store=store))

    nodes = build_nodes()
    print(f"סה\"כ {len(nodes)} צ'אנקים. מחשבים embedding בקבוצות...")
    size = config.EMBED_BATCH_SIZE
    for i in range(0, len(nodes), size):
        batch = nodes[i:i + size]
        with_retry(lambda b=batch: index.insert_nodes(b))
        print(f"  {min(i + size, len(nodes))}/{len(nodes)}")
        time.sleep(config.BATCH_PAUSE_SECONDS)
    print("האינדקס נבנה ונשמר ב-", config.CHROMA_DIR)


if __name__ == "__main__":
    try:
        main()
    except RagError as e:
        sys.exit(f"שגיאה: {e}")
