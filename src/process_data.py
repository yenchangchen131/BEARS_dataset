"""
資料提取與處理腳本
從五個原始資料集 (DRCD, SQuAD v2, MS MARCO, HotpotQA, 2WikiMultiHopQA) 中採樣並組裝文檔池。

輸出：
- data/processed/queries_raw.json: 100 筆 QA 對
- data/processed/corpus_raw.json: 5000 篇文檔
"""

import json
import uuid
import random
from pathlib import Path

# 設定隨機種子以確保可重現性
random.seed(42)

# 路徑設定
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 採樣配置 (共 100 題)
SAMPLING_CONFIG = {
    "drcd": {"count": 20, "type": "single-hop"},
    "squad_v2": {"count": 20, "type": "single-hop"},
    "ms_marco": {"count": 20, "type": "single-hop"},
    "hotpotqa": {"count": 20, "type": "multi-hop"},
    "2wiki": {"count": 20, "type": "multi-hop"},
}

TOTAL_CORPUS_SIZE = 5000


def generate_doc_id(source: str, original_id: str) -> str:
    """生成唯一的文檔 ID"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}:{original_id}"))


def generate_question_id(source: str, original_id: str) -> str:
    """生成唯一的問題 ID"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"q:{source}:{original_id}"))


def load_json(filepath: Path) -> list[dict]:
    """載入 JSON 檔案"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], filepath: Path) -> None:
    """儲存 JSON 檔案"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_drcd(
    data: list[dict], count: int
) -> tuple[list[dict], list[dict], set[str]]:
    """
    處理 DRCD 資料集
    結構: [{title, paragraphs: [{context, qas: [{question, answers, id}]}]}]

    Returns:
        queries: QA 列表
        gold_docs: 黃金文檔列表
        used_contexts: 已使用的 context 集合
    """
    queries = []
    gold_docs = []
    used_contexts: set[str] = set()

    # 展平所有 QA 對並記錄其 context
    all_qas = []
    for article in data:
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            if not context or context in used_contexts:
                continue
            for qa in para.get("qas", []):
                all_qas.append(
                    {
                        "qa": qa,
                        "context": context,
                        "title": article.get("title", ""),
                    }
                )

    # 隨機打亂並選取（確保每個 context 只選一個 QA）
    random.shuffle(all_qas)

    for item in all_qas:
        if len(queries) >= count:
            break

        context = item["context"]
        if context in used_contexts:
            continue

        qa = item["qa"]
        original_id = qa.get("id", str(uuid.uuid4()))
        doc_id = generate_doc_id("drcd", original_id)
        question_id = generate_question_id("drcd", original_id)

        # 取得答案文字
        answers = qa.get("answers", [])
        answer_text = answers[0].get("text", "") if answers else ""

        gold_docs.append(
            {
                "doc_id": doc_id,
                "content": context,
                "original_source": "drcd",
                "original_id": original_id,
                "is_gold": True,
            }
        )

        queries.append(
            {
                "question_id": question_id,
                "question": qa.get("question", ""),
                "gold_answer": answer_text,
                "gold_doc_ids": [doc_id],
                "source_dataset": "drcd",
                "question_type": "single-hop",
            }
        )

        used_contexts.add(context)

    print(f"[DRCD] 提取 {len(queries)} 題 QA, {len(gold_docs)} 篇黃金文檔")
    return queries, gold_docs, used_contexts


def process_squad_v2(
    data: list[dict], count: int
) -> tuple[list[dict], list[dict], set[str]]:
    """
    處理 SQuAD v2 資料集
    結構: [{id, title, context, question, answers: {text: [], answer_start: []}}]

    注意: SQuAD v2 包含不可回答的問題 (answers.text 為空陣列)，需過濾。

    Returns:
        queries: QA 列表
        gold_docs: 黃金文檔列表
        used_contexts: 已使用的 context 集合
    """
    queries = []
    gold_docs = []
    used_contexts: set[str] = set()

    # 過濾出有答案的題目，並展平
    answerable = []
    for item in data:
        answers = item.get("answers", {})
        answer_texts = answers.get("text", [])
        if not answer_texts:
            continue  # 跳過不可回答的問題

        context = item.get("context", "")
        if not context:
            continue

        answerable.append(item)

    # 隨機打亂
    random.shuffle(answerable)

    for item in answerable:
        if len(queries) >= count:
            break

        context = item.get("context", "")
        if context in used_contexts:
            continue

        original_id = item.get("id", str(uuid.uuid4()))
        doc_id = generate_doc_id("squad_v2", original_id)
        question_id = generate_question_id("squad_v2", original_id)

        answers = item.get("answers", {})
        answer_text = answers.get("text", [""])[0]

        gold_docs.append(
            {
                "doc_id": doc_id,
                "content": context,
                "original_source": "squad_v2",
                "original_id": original_id,
                "is_gold": True,
            }
        )

        queries.append(
            {
                "question_id": question_id,
                "question": item.get("question", ""),
                "gold_answer": answer_text,
                "gold_doc_ids": [doc_id],
                "source_dataset": "squad_v2",
                "question_type": "single-hop",
            }
        )

        used_contexts.add(context)

    print(f"[SQuAD v2] 提取 {len(queries)} 題 QA, {len(gold_docs)} 篇黃金文檔")
    return queries, gold_docs, used_contexts


def process_ms_marco(
    data: list[dict], count: int
) -> tuple[list[dict], list[dict], list[dict], set[str]]:
    """
    處理 MS MARCO v2.1 資料集
    結構: [{query_id, query, query_type,
            passages: {is_selected: [], passage_text: [], url: []},
            answers: []}]

    注意: 過濾掉 answers 為空或 "No Answer Present." 的 query。

    Returns:
        queries: QA 列表
        gold_docs: 黃金文檔列表
        hard_negatives: 困難負樣本列表 (同 query 下未選中的段落)
        used_contexts: 已使用的 context 集合
    """
    queries = []
    gold_docs = []
    hard_negatives = []
    used_contexts: set[str] = set()

    # 過濾有效的 query
    valid_items = []
    for item in data:
        answers = item.get("answers", [])
        # 跳過無答案的 query
        if not answers or answers == ["No Answer Present."]:
            continue

        passages = item.get("passages", {})
        is_selected = passages.get("is_selected", [])
        # 確保至少有一個被選中的段落
        if 1 not in is_selected:
            continue

        valid_items.append(item)

    # 隨機打亂
    random.shuffle(valid_items)

    for item in valid_items:
        if len(queries) >= count:
            break

        original_id = str(item.get("query_id", uuid.uuid4()))
        question_id = generate_question_id("ms_marco", original_id)

        passages = item.get("passages", {})
        is_selected_list = passages.get("is_selected", [])
        passage_texts = passages.get("passage_text", [])

        # 檢查是否有 context 已被使用
        all_passages = [t for t in passage_texts if t and t.strip()]
        if any(p in used_contexts for p in all_passages):
            continue

        gold_doc_ids = []
        question_used_contexts = []

        for i, (selected, text) in enumerate(
            zip(is_selected_list, passage_texts)
        ):
            if not text or not text.strip():
                continue

            doc_original_id = f"{original_id}_p{i}"
            doc_id = generate_doc_id("ms_marco", doc_original_id)

            doc = {
                "doc_id": doc_id,
                "content": text,
                "original_source": "ms_marco",
                "original_id": doc_original_id,
                "is_gold": selected == 1,
            }

            if selected == 1:
                gold_docs.append(doc)
                gold_doc_ids.append(doc_id)
            else:
                hard_negatives.append(doc)

            question_used_contexts.append(text)

        if gold_doc_ids:
            answers = item.get("answers", [])
            answer_text = answers[0] if answers else ""

            queries.append(
                {
                    "question_id": question_id,
                    "question": item.get("query", ""),
                    "gold_answer": answer_text,
                    "gold_doc_ids": gold_doc_ids,
                    "source_dataset": "ms_marco",
                    "question_type": "single-hop",
                }
            )
            used_contexts.update(question_used_contexts)

    print(
        f"[MS MARCO] 提取 {len(queries)} 題 QA, {len(gold_docs)} 篇黃金文檔, {len(hard_negatives)} 篇困難負樣本"
    )
    return queries, gold_docs, hard_negatives, used_contexts


def process_hotpotqa(
    data: list[dict], count: int
) -> tuple[list[dict], list[dict], list[dict], set[str]]:
    """
    處理 HotpotQA 資料集
    結構: [{id, question, answer, supporting_facts: {title, sent_id},
            context: {title: [], sentences: [[sent1, sent2, ...], ...]}}]

    Returns:
        queries: QA 列表
        gold_docs: 黃金文檔列表
        hard_negatives: 困難負樣本列表
        used_contexts: 已使用的 context 集合
    """
    queries = []
    gold_docs = []
    hard_negatives = []
    used_contexts: set[str] = set()

    # 隨機打亂
    data_copy = data.copy()
    random.shuffle(data_copy)

    for item in data_copy:
        if len(queries) >= count:
            break

        original_id = item.get("id", str(uuid.uuid4()))
        question_id = generate_question_id("hotpotqa", original_id)

        # 解析 context
        context_data = item.get("context", {})
        titles = context_data.get("title", [])
        sentences_list = context_data.get("sentences", [])

        # 解析 supporting_facts
        supporting_facts = item.get("supporting_facts", {})
        gold_titles = set(supporting_facts.get("title", []))

        # 建立 title -> doc 的映射
        gold_doc_ids = []
        question_used_contexts = []

        for i, title in enumerate(titles):
            if i >= len(sentences_list):
                continue

            # 合併句子為完整段落
            sentences = sentences_list[i]
            content = (
                " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            )

            if not content.strip():
                continue

            doc_original_id = f"{original_id}_{title}"
            doc_id = generate_doc_id("hotpotqa", doc_original_id)

            if content in used_contexts:
                # 如果 context 已被使用，跳過此問題
                continue

            doc = {
                "doc_id": doc_id,
                "content": content,
                "original_source": "hotpotqa",
                "original_id": doc_original_id,
                "is_gold": title in gold_titles,
            }

            if title in gold_titles:
                gold_docs.append(doc)
                gold_doc_ids.append(doc_id)
            else:
                hard_negatives.append(doc)

            question_used_contexts.append(content)

        # 只有當有黃金文檔時才添加問題
        if gold_doc_ids:
            queries.append(
                {
                    "question_id": question_id,
                    "question": item.get("question", ""),
                    "gold_answer": item.get("answer", ""),
                    "gold_doc_ids": gold_doc_ids,
                    "source_dataset": "hotpotqa",
                    "question_type": "multi-hop",
                }
            )
            used_contexts.update(question_used_contexts)

    print(
        f"[HotpotQA] 提取 {len(queries)} 題 QA, {len(gold_docs)} 篇黃金文檔, {len(hard_negatives)} 篇困難負樣本"
    )
    return queries, gold_docs, hard_negatives, used_contexts


def process_2wiki(
    data: list[dict], count: int
) -> tuple[list[dict], list[dict], list[dict], set[str]]:
    """
    處理 2WikiMultiHopQA 資料集
    結構類似 HotpotQA
    """
    queries = []
    gold_docs = []
    hard_negatives = []
    used_contexts: set[str] = set()

    # 隨機打亂
    data_copy = data.copy()
    random.shuffle(data_copy)

    for item in data_copy:
        if len(queries) >= count:
            break

        original_id = item.get("id", str(uuid.uuid4()))
        question_id = generate_question_id("2wiki", original_id)

        # 解析 context
        context_data = item.get("context", {})
        titles = context_data.get("title", [])
        sentences_list = context_data.get("sentences", [])

        # 解析 supporting_facts
        supporting_facts = item.get("supporting_facts", {})
        gold_titles = set(supporting_facts.get("title", []))

        # 建立 title -> doc 的映射
        gold_doc_ids = []
        question_used_contexts = []

        for i, title in enumerate(titles):
            if i >= len(sentences_list):
                continue

            # 合併句子為完整段落
            sentences = sentences_list[i]
            content = (
                " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            )

            if not content.strip():
                continue

            doc_original_id = f"{original_id}_{title}"
            doc_id = generate_doc_id("2wiki", doc_original_id)

            if content in used_contexts:
                continue

            doc = {
                "doc_id": doc_id,
                "content": content,
                "original_source": "2wiki",
                "original_id": doc_original_id,
                "is_gold": title in gold_titles,
            }

            if title in gold_titles:
                gold_docs.append(doc)
                gold_doc_ids.append(doc_id)
            else:
                hard_negatives.append(doc)

            question_used_contexts.append(content)

        # 只有當有黃金文檔時才添加問題
        if gold_doc_ids:
            queries.append(
                {
                    "question_id": question_id,
                    "question": item.get("question", ""),
                    "gold_answer": item.get("answer", ""),
                    "gold_doc_ids": gold_doc_ids,
                    "source_dataset": "2wiki",
                    "question_type": "multi-hop",
                }
            )
            used_contexts.update(question_used_contexts)

    print(
        f"[2Wiki] 提取 {len(queries)} 題 QA, {len(gold_docs)} 篇黃金文檔, {len(hard_negatives)} 篇困難負樣本"
    )
    return queries, gold_docs, hard_negatives, used_contexts


def collect_random_negatives(
    drcd_data: list[dict],
    squad_v2_data: list[dict],
    ms_marco_data: list[dict],
    used_contexts: set[str],
    target_count: int,
) -> list[dict]:
    """
    從多個資料集的未使用段落中收集隨機負樣本。
    來源: DRCD, SQuAD v2, MS MARCO
    """
    random_negatives = []
    neg_counter = 0

    # 從 DRCD 收集
    for article in drcd_data:
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            if context and context not in used_contexts:
                doc_id = generate_doc_id("drcd", f"neg_{neg_counter}")
                random_negatives.append(
                    {
                        "doc_id": doc_id,
                        "content": context,
                        "original_source": "drcd",
                        "original_id": para.get("id", ""),
                        "is_gold": False,
                    }
                )
                used_contexts.add(context)
                neg_counter += 1

    # 從 SQuAD v2 收集
    for item in squad_v2_data:
        context = item.get("context", "")
        if context and context not in used_contexts:
            doc_id = generate_doc_id("squad_v2", f"neg_{neg_counter}")
            random_negatives.append(
                {
                    "doc_id": doc_id,
                    "content": context,
                    "original_source": "squad_v2",
                    "original_id": item.get("id", ""),
                    "is_gold": False,
                }
            )
            used_contexts.add(context)
            neg_counter += 1

    # 從 MS MARCO 收集
    for item in ms_marco_data:
        passages = item.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        for i, text in enumerate(passage_texts):
            if text and text.strip() and text not in used_contexts:
                doc_id = generate_doc_id("ms_marco", f"neg_{neg_counter}")
                random_negatives.append(
                    {
                        "doc_id": doc_id,
                        "content": text,
                        "original_source": "ms_marco",
                        "original_id": f"{item.get('query_id', '')}_{i}",
                        "is_gold": False,
                    }
                )
                used_contexts.add(text)
                neg_counter += 1

    print(f"  - 可用的隨機負樣本總數: {len(random_negatives)} 篇")

    # 隨機打亂並取需要的數量
    random.shuffle(random_negatives)
    return random_negatives[:target_count]


def main():
    print("=" * 60)
    print("開始資料提取與處理")
    print("=" * 60)

    # 載入原始資料
    print("\n[1/6] 載入原始資料...")
    drcd_data = load_json(RAW_DIR / "drcd.json")
    squad_v2_data = load_json(RAW_DIR / "squad_v2.json")
    ms_marco_data = load_json(RAW_DIR / "ms_marco.json")
    hotpotqa_data = load_json(RAW_DIR / "hotpotqa.json")
    wiki2_data = load_json(RAW_DIR / "2wiki.json")

    print(f"  - DRCD: {len(drcd_data)} 篇文章")
    print(f"  - SQuAD v2: {len(squad_v2_data)} 筆記錄")
    print(f"  - MS MARCO: {len(ms_marco_data)} 筆記錄")
    print(f"  - HotpotQA: {len(hotpotqa_data)} 筆記錄")
    print(f"  - 2Wiki: {len(wiki2_data)} 筆記錄")

    # 處理各資料集
    print("\n[2/6] 處理 DRCD...")
    drcd_queries, drcd_gold_docs, drcd_used = process_drcd(
        drcd_data, SAMPLING_CONFIG["drcd"]["count"]
    )

    print("\n[3/6] 處理 SQuAD v2...")
    squad_queries, squad_gold_docs, squad_used = process_squad_v2(
        squad_v2_data, SAMPLING_CONFIG["squad_v2"]["count"]
    )

    print("\n[4/6] 處理 MS MARCO...")
    marco_queries, marco_gold_docs, marco_hard_negs, marco_used = process_ms_marco(
        ms_marco_data, SAMPLING_CONFIG["ms_marco"]["count"]
    )

    print("\n[5/6] 處理 HotpotQA...")
    hotpot_queries, hotpot_gold_docs, hotpot_hard_negs, hotpot_used = process_hotpotqa(
        hotpotqa_data, SAMPLING_CONFIG["hotpotqa"]["count"]
    )

    print("\n[6/6] 處理 2WikiMultiHopQA...")
    wiki2_queries, wiki2_gold_docs, wiki2_hard_negs, wiki2_used = process_2wiki(
        wiki2_data, SAMPLING_CONFIG["2wiki"]["count"]
    )

    # 合併所有 queries
    all_queries = (
        drcd_queries
        + squad_queries
        + marco_queries
        + hotpot_queries
        + wiki2_queries
    )

    # 合併所有文檔
    all_gold_docs = (
        drcd_gold_docs
        + squad_gold_docs
        + marco_gold_docs
        + hotpot_gold_docs
        + wiki2_gold_docs
    )
    all_hard_negatives = marco_hard_negs + hotpot_hard_negs + wiki2_hard_negs

    # 記錄所有已使用的 contexts
    all_used_contexts = drcd_used | squad_used | marco_used | hotpot_used | wiki2_used

    # 計算需要多少隨機負樣本
    current_corpus_size = len(all_gold_docs) + len(all_hard_negatives)
    needed_random_negs = TOTAL_CORPUS_SIZE - current_corpus_size

    print("\n[組裝文檔池]")
    print(f"  - 黃金文檔: {len(all_gold_docs)} 篇")
    print(f"  - 困難負樣本: {len(all_hard_negatives)} 篇")
    print(f"  - 需要隨機負樣本: {needed_random_negs} 篇")

    # 收集隨機負樣本
    if needed_random_negs > 0:
        random_negatives = collect_random_negatives(
            drcd_data, squad_v2_data, ms_marco_data,
            all_used_contexts, needed_random_negs
        )
        print(f"  - 收集到隨機負樣本: {len(random_negatives)} 篇")
    else:
        random_negatives = []

    # 組裝最終文檔池
    all_corpus = all_gold_docs + all_hard_negatives + random_negatives

    # 打亂文檔順序
    random.shuffle(all_corpus)

    # 輸出統計
    print(f"\n{'=' * 60}")
    print("最終統計")
    print("=" * 60)
    print(f"總 QA 數量: {len(all_queries)}")
    print(f"  - DRCD (繁中/單跳): {len(drcd_queries)}")
    print(f"  - SQuAD v2 (英文/單跳): {len(squad_queries)}")
    print(f"  - MS MARCO (英文/單跳): {len(marco_queries)}")
    print(f"  - HotpotQA (英文/多跳): {len(hotpot_queries)}")
    print(f"  - 2Wiki (英文/多跳): {len(wiki2_queries)}")
    print(f"\n總文檔數量: {len(all_corpus)}")

    gold_count = sum(1 for d in all_corpus if d.get("is_gold"))
    print(f"  - 黃金文檔 (is_gold=True): {gold_count}")
    print(f"  - 其他文檔 (is_gold=False): {len(all_corpus) - gold_count}")

    # 儲存輸出
    print("\n[儲存輸出]")
    save_json(all_queries, PROCESSED_DIR / "queries_raw.json")
    save_json(all_corpus, PROCESSED_DIR / "corpus_raw.json")
    print(f"  - 已儲存: {PROCESSED_DIR / 'queries_raw.json'}")
    print(f"  - 已儲存: {PROCESSED_DIR / 'corpus_raw.json'}")

    print(f"\n{'=' * 60}")
    print("處理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
