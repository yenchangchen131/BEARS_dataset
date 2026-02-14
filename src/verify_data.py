"""
資料驗證腳本
驗證處理後的資料是否符合規格與預期。

驗證流程：
0. 檔案存在性檢查 (4 個 JSON 檔案)

對 Raw 組 (queries_raw + corpus_raw) 與 Processed 組 (queries + corpus)
各自執行以下驗證 (需該組兩個檔案皆存在)：
1. 資料數量 (100 QA, 5000 Docs) 與來源分佈
2. 欄位完整性與型別
3. 資料一致性 (Gold Doc IDs 存在於 Corpus)
4. 無重複 Queries 或 Corpus

僅 Processed 組額外執行：
5. 語言檢查 (非 DRCD 資料是否已翻譯為中文)

當 4 個檔案皆存在時，另外執行 Raw vs Processed 交叉一致性驗證。
"""

import json
from pathlib import Path
from collections import Counter

# 路徑設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 預期值配置
EXPECTED_QUERIES = 100
EXPECTED_CORPUS = 5000
EXPECTED_DISTRIBUTION = {
    "drcd": 20,
    "squad_v2": 20,
    "ms_marco": 20,
    "hotpotqa": 20,
    "2wiki": 20,
}

# --- Schema 定義 (Field Name → Expected Type) ---
QUERY_SCHEMA = {
    "question_id": str,
    "question": str,
    "gold_answer": str,
    "gold_doc_ids": list,
    "source_dataset": str,
    "question_type": str,
}

CORPUS_SCHEMA = {
    "doc_id": str,
    "content": str,
    "original_source": str,
    "original_id": str,
    "is_gold": bool,
}


def load_json(filepath: Path) -> list[dict]:
    """載入 JSON 檔案"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def contains_chinese(text: str) -> bool:
    """檢查文字是否包含中文字元"""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def validate_pair(
    queries: list[dict],
    corpus: list[dict],
    label: str,
    check_chinese: bool = False,
) -> None:
    """
    對一組 (queries, corpus) 執行完整的 5 項驗證。

    Args:
        queries: 問答資料列表
        corpus: 文檔資料列表
        label: 顯示用的標籤 (如 "Processed" 或 "Raw")
        check_chinese: 是否額外檢查中文翻譯
    """
    print(f"\n{'=' * 60}")
    print(f"[{label}] 驗證開始")
    print("=" * 60)

    # --- 1. 資料數量 ---
    print(f"\n[{label} - 1. 資料數量]")

    if len(queries) == EXPECTED_QUERIES:
        print(f"  [PASS] Queries 數量正確: {len(queries)}")
    else:
        print(f"  [FAIL] Queries 數量錯誤: {len(queries)} (預期 {EXPECTED_QUERIES})")

    if len(corpus) == EXPECTED_CORPUS:
        print(f"  [PASS] Corpus 數量正確: {len(corpus)}")
    else:
        print(f"  [FAIL] Corpus 數量錯誤: {len(corpus)} (預期 {EXPECTED_CORPUS})")

    sources = Counter(q.get("source_dataset") for q in queries)
    if sources == EXPECTED_DISTRIBUTION:
        print(f"  [PASS] 來源分佈正確: {dict(sources)}")
    else:
        print(f"  [FAIL] 來源分佈錯誤: {dict(sources)}")

    # --- 2. 欄位完整性與型別 ---
    print(f"\n[{label} - 2. 欄位完整性與型別]")

    q_field_errors = 0
    for i, q in enumerate(queries):
        for field, expected_type in QUERY_SCHEMA.items():
            if field not in q:
                print(f"  [FAIL] queries[{i}] 缺少欄位: {field}")
                q_field_errors += 1
            elif not isinstance(q[field], expected_type):
                print(
                    f"  [FAIL] queries[{i}].{field} 型別錯誤: "
                    f"{type(q[field]).__name__} (預期 {expected_type.__name__})"
                )
                q_field_errors += 1
    if q_field_errors == 0:
        print(f"  [PASS] Queries 欄位正確 ({len(QUERY_SCHEMA)} 個必要欄位)")

    c_field_errors = 0
    for i, d in enumerate(corpus):
        for field, expected_type in CORPUS_SCHEMA.items():
            if field not in d:
                print(f"  [FAIL] corpus[{i}] 缺少欄位: {field}")
                c_field_errors += 1
            elif not isinstance(d[field], expected_type):
                print(
                    f"  [FAIL] corpus[{i}].{field} 型別錯誤: "
                    f"{type(d[field]).__name__} (預期 {expected_type.__name__})"
                )
                c_field_errors += 1
    if c_field_errors == 0:
        print(f"  [PASS] Corpus 欄位正確 ({len(CORPUS_SCHEMA)} 個必要欄位)")

    # --- 3. 資料一致性 (Gold Doc IDs ∈ Corpus) ---
    print(f"\n[{label} - 3. 資料一致性]")
    corpus_ids = {d.get("doc_id") for d in corpus}
    missing_docs = []
    for q in queries:
        for gid in q.get("gold_doc_ids", []):
            if gid not in corpus_ids:
                missing_docs.append((q.get("question_id"), gid))

    if not missing_docs:
        print("  [PASS] 所有 Gold Doc IDs 皆存在於 Corpus")
    else:
        print(f"  [FAIL] 發現 {len(missing_docs)} 個缺失文檔:")
        for qid, did in missing_docs[:10]:  # 最多顯示 10 筆
            print(f"    - query={qid} → doc={did}")
        if len(missing_docs) > 10:
            print(f"    ... 還有 {len(missing_docs) - 10} 筆")

    # --- 4. 無重複 Queries 或 Corpus ---
    print(f"\n[{label} - 4. 重複性檢查]")

    q_ids = [q["question_id"] for q in queries]
    dup_q_ids = [item for item, count in Counter(q_ids).items() if count > 1]
    if not dup_q_ids:
        print(f"  [PASS] Queries 無重複 ID ({len(q_ids)} unique)")
    else:
        print(f"  [FAIL] 發現 {len(dup_q_ids)} 個重複 question_id:")
        for qid in dup_q_ids[:10]:
            print(f"    - {qid}")

    doc_ids = [d["doc_id"] for d in corpus]
    dup_doc_ids = [item for item, count in Counter(doc_ids).items() if count > 1]
    if not dup_doc_ids:
        print(f"  [PASS] Corpus 無重複 doc_id ({len(set(doc_ids))} unique)")
    else:
        print(f"  [FAIL] 發現 {len(dup_doc_ids)} 個重複 doc_id:")
        for did in dup_doc_ids[:10]:
            print(f"    - {did}")

    # --- 5. 語言檢查 (僅 Processed) ---
    if check_chinese:
        print(f"\n[{label} - 5. 語言檢查]")

        q_errors = 0
        non_drcd_q = [q for q in queries if q.get("source_dataset") != "drcd"]
        for q in non_drcd_q:
            if not contains_chinese(q.get("question", "")):
                q_errors += 1
        if q_errors == 0:
            print(f"  [PASS] 非 DRCD 問題皆包含中文 ({len(non_drcd_q)} 題)")
        else:
            print(f"  [WARN] {q_errors} 題可能未翻譯")

        c_errors = 0
        non_drcd_c = [d for d in corpus if d.get("original_source") != "drcd"]
        for c in non_drcd_c:
            if not contains_chinese(c.get("content", "")):
                c_errors += 1
        if c_errors == 0:
            print(f"  [PASS] 非 DRCD 文檔皆包含中文 ({len(non_drcd_c)} 篇)")
        else:
            print(f"  [WARN] {c_errors} 篇可能未翻譯")


def main():
    print("=" * 60)
    print("開始資料驗證")
    print("=" * 60)

    # --- 檔案存在性檢查 ---
    file_pairs = {
        "Raw": (PROCESSED_DIR / "queries_raw.json", PROCESSED_DIR / "corpus_raw.json"),
        "Processed": (PROCESSED_DIR / "queries.json", PROCESSED_DIR / "corpus.json"),
    }

    # --- 0. 檔案存在性檢查 ---
    print("\n[0. 檔案存在性檢查]")
    for label, (q_path, c_path) in file_pairs.items():
        q_exists = q_path.exists()
        c_exists = c_path.exists()
        print(f"  {'[PASS]' if q_exists else '[FAIL]'} {q_path.name}")
        print(f"  {'[PASS]' if c_exists else '[FAIL]'} {c_path.name}")

        if q_exists and c_exists:
            try:
                queries = load_json(q_path)
                corpus = load_json(c_path)
                validate_pair(
                    queries,
                    corpus,
                    label,
                    check_chinese=(label == "Processed"),
                )
            except Exception as e:
                print(f"  [FAIL] {label} 資料讀取失敗: {e}")
        else:
            missing = []
            if not q_exists:
                missing.append(q_path.name)
            if not c_exists:
                missing.append(c_path.name)
            print(f"  [SKIP] {label} 驗證跳過 (缺少: {', '.join(missing)})")

    # --- Raw vs Processed 交叉一致性 ---
    raw_q_path = file_pairs["Raw"][0]
    raw_c_path = file_pairs["Raw"][1]
    proc_q_path = file_pairs["Processed"][0]
    proc_c_path = file_pairs["Processed"][1]

    all_exist = all(
        p.exists() for p in [raw_q_path, raw_c_path, proc_q_path, proc_c_path]
    )

    if all_exist:
        print(f"\n{'=' * 60}")
        print("[Raw vs Processed 交叉一致性]")
        print("=" * 60)

        queries_raw = load_json(raw_q_path)
        queries = load_json(proc_q_path)
        corpus_raw = load_json(raw_c_path)
        corpus = load_json(proc_c_path)

        # Queries 一致性
        if len(queries) == len(queries_raw):
            print(f"  [PASS] Queries 數量一致 ({len(queries)})")
        else:
            print(
                f"  [FAIL] Queries 數量不一致 ({len(queries)} vs {len(queries_raw)})"
            )

        q_ids = set(q["question_id"] for q in queries)
        r_ids = set(q["question_id"] for q in queries_raw)
        if q_ids == r_ids:
            print("  [PASS] Queries ID 集合一致")
        else:
            print("  [FAIL] Queries ID 集合不一致")

        # Corpus 一致性
        if len(corpus) == len(corpus_raw):
            print(f"  [PASS] Corpus 數量一致 ({len(corpus)})")
        else:
            print(
                f"  [FAIL] Corpus 數量不一致 ({len(corpus)} vs {len(corpus_raw)})"
            )

    print(f"\n{'=' * 60}")
    print("驗證完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
