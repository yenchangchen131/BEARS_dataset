# BEARS Dataset

**Benchmark for Evaluating and Assessing RAG Systems — 繁體中文 RAG 效能評測微型資料集**

一個**輕量級**、**低成本**且**在地化**的繁體中文 Retrieval-Augmented Generation (RAG) 評測資料集。透過將學術界標準的檢索問答資料集翻譯為台灣繁體中文，快速驗證 RAG 系統在單跳、多跳推理及高雜訊環境下的檢索與生成效能。

---

## 📋 目錄

- [專案概述](#-專案概述)
- [資料集組成](#-資料集組成)
- [文檔池架構](#-文檔池架構)
- [資料格式](#-資料格式)
- [專案結構](#-專案結構)
- [快速開始](#-快速開始)
- [處理流程](#-處理流程)
- [腳本說明](#-腳本說明)
- [環境變數](#-環境變數)
- [技術堆疊](#-技術堆疊)
- [評估指標](#-評估指標)
- [授權條款](#-授權條款)

---

## 🎯 專案概述

本計畫旨在建立一個 **micro-scale** 的繁體中文 RAG 評測基準。與大規模評測集不同，BEARS 以 **60 題 QA** + **600 篇文檔** 的精簡配置，讓開發者能以最低成本快速檢驗 RAG pipeline 的核心能力：

- **檢索精確度**：能否在大量干擾文檔中找到正確段落？
- **跨文檔推理**：能否連結多篇文檔進行多跳推理？
- **抗干擾能力**：面對高度相似的 Hard Negatives，檢索器是否仍具語意區辨力？
- **繁體中文理解**：翻譯後的在地化資料，模型是否能正確處理？

---

## 📊 資料集組成

目標總題數為 **60 題**，依據難度與推理類型進行配比：

| 資料集來源                | 原始語言 | 推理類型          | 採樣數量        | 測試重點                                 |
| ------------------------- | -------- | ----------------- | --------------- | ---------------------------------------- |
| **DRCD**            | 繁體中文 | 單跳 (Single-hop) | 20 題           | 原生繁中基準，無翻譯誤差                 |
| **HotpotQA**        | 英文     | 多跳 (Multi-hop)  | 20 題           | 跨文檔推理 + 困難負樣本 (Hard Negatives) |
| **2WikiMultiHopQA** | 英文     | 多跳 (Multi-hop)  | 20 題           | 多實體 (Entity) 邏輯關聯                 |
| **總計**            | —       | —                | **60 題** | 單跳 20 題 (33%) / 多跳 40 題 (67%)      |

---

## 📚 文檔池架構

採用 **Global Pool (混合文檔池)** 模式，所有 Query 共享同一個檢索庫，模擬真實知識庫環境。

| 類別                                    | 預估數量         | 說明                                                       |
| --------------------------------------- | ---------------- | ---------------------------------------------------------- |
| **Gold Contexts** (正解文檔)      | ~60–80 篇       | 每道 QA 對應的正確文檔（多跳問題通常對應 ≥2 篇）          |
| **Hard Negatives** (困難負樣本)   | ~40–60 篇       | 來自 HotpotQA / 2Wiki 的官方干擾項，與問題有高度關鍵字重疊 |
| **Random Negatives** (隨機負樣本) | ~460–500 篇     | 從 DRCD 隨機抽取，大幅增加背景雜訊 (Needle-in-a-Haystack)  |
| **總計**                          | **600 篇** | 所有文檔混合並建立統一索引                                 |

---

## 📁 資料格式

### 評測題庫 (`queries.json`)

```json
[
  {
    "question_id": "uuid-string-q01",
    "question": "繁體中文問題？",
    "gold_answer": "繁體中文標準答案",
    "gold_doc_ids": ["uuid-string-001", "uuid-string-005"],
    "source_dataset": "hotpotqa",
    "question_type": "multi-hop"
  }
]
```

### 文檔庫 (`corpus.json`)

```json
[
  {
    "doc_id": "uuid-string-001",
    "content": "翻譯後的繁體中文文章內容...",
    "original_source": "hotpotqa",
    "original_id": "原始資料集中的 ID",
    "is_gold": true
  }
]
```

> **備註**：所有 ID 均使用 UUID v5 生成，確保跨執行的一致性與可重現性。

---

## 🗂️ 專案結構

```
BEARS_dataset/
├── main.py                     # 程式進入點
├── pyproject.toml              # 專案設定與依賴
├── uv.lock                     # uv 鎖定檔
├── .env                        # 環境變數 (不納入版控)
├── src/
│   ├── data_download.py        # Step 1: 下載原始資料集
│   ├── process_data.py         # Step 2: 採樣與組裝文檔池
│   ├── translate_data.py       # Step 3: 多執行緒 LLM 翻譯
│   ├── replace_question.py     # 工具: 問題抽換 (單題替換)
│   └── verify_data.py          # 工具: 資料完整性驗證
└── data/                       # (gitignored) 資料儲存 (不納入版控)
    ├── raw/                    # 原始下載資料
    │   ├── drcd.json
    │   ├── hotpotqa.json
    │   └── 2wiki.json
    └── processed/              # 處理後的最終資料
        ├── queries_raw.json    # 未翻譯的 QA (中間產物)
        ├── corpus_raw.json     # 未翻譯的文檔 (中間產物)
        ├── queries.json        # ✅ 最終翻譯版 QA
        └── corpus.json         # ✅ 最終翻譯版文檔
```

---

## 🚀 快速開始

### 前置需求

- **Python** ≥ 3.12
- **uv** 套件管理工具（推薦）
- **OpenAI API Key**（用於翻譯步驟）

### 安裝

```bash
# 1. Clone 專案
git clone https://github.com/your-username/BEARS_dataset.git
cd BEARS_dataset

# 2. 安裝依賴 (使用 uv)
uv sync

# 3. 設定環境變數
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 一鍵生成資料集

```bash
# Step 1: 下載原始資料集 (DRCD, HotpotQA, 2WikiMultiHopQA)
uv run src/data_download.py

# Step 2: 採樣與組裝文檔池 (產出 queries_raw.json + corpus_raw.json)
uv run src/process_data.py

# Step 3: 翻譯為繁體中文 (產出 queries.json + corpus.json)
uv run src/translate_data.py

# Step 4: 驗證資料完整性
uv run src/verify_data.py
```

---

## ⚙️ 處理流程

整個資料集的產生遵循 **ETL (Extract → Transform → Load)** 流程：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BEARS Pipeline                               │
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │   Extract     │    │   Transform   │    │       Load           │  │
│  │              │    │               │    │                      │  │
│  │ data_download │──▶│ process_data  │──▶│   translate_data     │  │
│  │              │    │               │    │                      │  │
│  │ HuggingFace  │    │ 採樣 60 QA    │    │ GPT-4.1 翻譯         │  │
│  │ → data/raw/  │    │ 組裝 600 docs │    │ → queries.json       │  │
│  │              │    │ → *_raw.json  │    │ → corpus.json        │  │
│  └──────────────┘    └───────────────┘    └──────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Utilities                                                   │    │
│  │  🔄 replace_question.py — 單題抽換 (附自動翻譯)              │    │
│  │  ✅ verify_data.py      — 資料完整性與一致性驗證             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📜 腳本說明

### `src/data_download.py` — 資料下載

從 HuggingFace Hub 下載三個原始資料集，並強制轉存為標準 JSON Array 格式。

| 資料集          | HuggingFace ID                 | Config         | Split          |
| --------------- | ------------------------------ | -------------- | -------------- |
| DRCD            | `voidful/drcd`               | —             | `test`       |
| HotpotQA        | `hotpotqa/hotpot_qa`         | `distractor` | `validation` |
| 2WikiMultiHopQA | `framolfese/2WikiMultihopQA` | —             | `validation` |

**特點**：

- 若檔案已存在則自動跳過，避免重複下載
- 所有中文字元保持原樣（`ensure_ascii=False`）

---

### `src/process_data.py` — 資料採樣與組裝

從原始資料中進行隨機採樣，並組裝 Global Pool 文檔池。

**採樣配置**：

- DRCD: 20 題 (單跳)
- HotpotQA: 20 題 (多跳)
- 2WikiMultiHopQA: 20 題 (多跳)
- 文檔池總量: 600 篇

**處理邏輯**：

1. 各資料集獨立處理，提取 Gold Contexts 與 Hard Negatives
2. 使用 UUID v5 為每篇文檔與每道問題生成確定性 ID
3. 從未使用的 DRCD 段落補充 Random Negatives 至目標數量
4. 隨機種子固定為 `42`，確保可重現性

**輸出**：`queries_raw.json` + `corpus_raw.json`

---

### `src/translate_data.py` — 多執行緒翻譯

使用 OpenAI GPT-4.1 將英文資料翻譯為台灣繁體中文，支援多執行緒並行處理。

**配置**：

- 模型：`gpt-4.1`
- 並行執行緒數：20
- 最大重試次數：5（含指數退避）
- 溫度：0.3

**翻譯規則**：

- DRCD 資料（原生繁中）自動跳過，不進行翻譯
- 人名、地名等專有名詞使用台灣常見翻譯，並括號標註原文
- 僅翻譯，嚴禁回答問題內容
- 保留數字與日期原始格式

**輸出**：`queries.json` + `corpus.json`

---

### `src/replace_question.py` — 問題抽換工具

支援將指定的問題抽換為同資料集的另一道題目，同步更新所有相關的 JSON 檔案。

```bash
uv run src/replace_question.py <question_id>

# 範例
uv run src/replace_question.py e7124c20-75c2-5d99-9acf-d2cba40228fa
```

**功能**：

- 根據 `question_id` 定位目標問題
- 從同一來源資料集中抽取未使用的替換候選
- 對於非 DRCD 資料集，自動呼叫 GPT-4o-mini 進行翻譯
- 對於多跳問題，連同 Hard Negatives 一併替換
- 同步更新 `queries.json`、`corpus.json`、`queries_raw.json`、`corpus_raw.json`

---

### `src/verify_data.py` — 資料驗證工具

全面驗證處理後的資料是否符合規格，包含以下檢查項目：

| 驗證項目         | 說明                                      |
| ---------------- | ----------------------------------------- |
| 檔案存在性       | 確認四個 JSON 檔案皆存在且可讀取          |
| 資料數量         | Queries = 60 題，Corpus = 600 篇          |
| 來源分佈         | DRCD / HotpotQA / 2Wiki 各 20 題          |
| ID 唯一性        | 檢查 `question_id` 與 `doc_id` 無重複 |
| 一致性           | 所有 `gold_doc_ids` 皆存在於 Corpus 中  |
| 語言檢查         | 非 DRCD 資料是否皆已翻譯為中文            |
| Raw vs Processed | 兩組資料的數量與 ID 集合是否一致          |

```bash
uv run src/verify_data.py
```

---

## 🔑 環境變數

在專案根目錄建立 `.env` 檔案：

```env
OPENAI_API_KEY=sk-your-api-key-here
```

> ⚠️ `.env` 已被 `.gitignore` 排除，不會被提交至版控系統。

---

## 🛠️ 技術堆疊

| 項目       | 技術                               |
| ---------- | ---------------------------------- |
| 語言       | Python 3.12+                       |
| 套件管理   | [uv](https://github.com/astral-sh/uv) |
| 資料集載入 | `datasets` (HuggingFace)         |
| 翻譯 API   | `openai` (GPT-4.1 / GPT-4o-mini) |
| 資料處理   | `pandas`, `pyarrow`            |
| 串流 JSON  | `ijson`                          |
| 環境變數   | `python-dotenv`                  |
| ID 生成    | `uuid` (UUID v5, 內建)           |

---

## 📏 評估指標

本資料集設計搭配以下評估方式使用：

### 檢索階段 (Retrieval)

- **Hit Rate (Recall@K)**：設定 `K = 5`，檢查 `gold_doc_ids` 是否出現在檢索結果的前 K 名中
- 多跳問題建議採用 **嚴格模式 (Strict)**：必須找齊所有相關文檔才算得分

### 生成階段 (Generation)

- **LLM-as-a-Judge**：放棄傳統 EM / F1 Score（翻譯會導致用詞改變），改用 LLM 判斷「模型回答」與「標準答案」的語意一致性
- 評分方式：**Pass** (通過) / **Fail** (失敗)

---

## 📄 授權條款

本專案的程式碼以 MIT License 發布。

> **注意**：本資料集中使用的原始數據來自 DRCD、HotpotQA 和 2WikiMultiHopQA，各資料集有其各自的授權條款，使用時請遵循對應的原始授權規定。
