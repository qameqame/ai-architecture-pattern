"""
Pattern 5: Orchestrator-Workers Workflow (Ollamaローカルモデル版)
====================================================================
中央のOrchestrator(LLM)が、タスク内容に応じて動的にサブタスクを
分解する(事前に固定されたサブタスクリストではない点がParallelization
のSectioningとの違い)。各サブタスクはWorker(LLM)に振られ、並列に
実行され、最後にOrchestratorが結果を統合する。

例: レポート作成タスク。トピックに応じて必要なセクションを
Orchestratorが動的に決定し、各セクションをWorkerが並列に執筆、
最後にOrchestratorが統合・整形する。

事前準備:
    1. Ollamaをインストールして起動: `ollama serve`
    2. モデルをpull: `ollama pull qwen2.5`
       (他のモデルを使う場合は下の MODEL 変数を変更)
"""

import concurrent.futures
import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5"


def ollama_chat(prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
    """Ollamaのローカルサーバー(REST API)にチャットリクエストを送る。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Ollamaに接続できませんでした。`ollama serve` が起動しているか、"
            f"モデル `{MODEL}` を `ollama pull {MODEL}` 済みか確認してください。詳細: {e}"
        ) from e


def orchestrator_plan(topic: str) -> list[str]:
    """Orchestrator役: トピックに応じて動的にセクション構成を決める。"""
    print(f"  [Orchestrator] トピック分析中: {topic}")
    prompt = (
        f"「{topic}」についてのレポートに必要なセクション見出しを4つ、"
        "日本語で箇条書き(各行「- 見出し」の形式)で挙げてください。"
        "説明文は不要で、見出しだけを出力してください。"
    )
    raw = ollama_chat(prompt, temperature=0.4)

    # 「- 見出し」形式の行を抽出。パースできない場合はフォールバックを使う。
    sections = [
        re.sub(r"^[-*\d.\s]+", "", line).strip()
        for line in raw.splitlines()
        if line.strip() and re.match(r"^[-*\d]", line.strip())
    ]
    if not sections:
        sections = ["概要", "背景", "現状分析", "まとめ"]

    print(f"  [Orchestrator] 動的に決定したセクション構成: {sections}")
    return sections


def worker(topic: str, section: str) -> str:
    """Worker役: 割り当てられたセクションを執筆する。"""
    print(f"    [Worker] セクション執筆開始: {section}")
    prompt = f"「{topic}」というレポートの「{section}」セクションを150字程度の日本語で執筆してください。本文だけを出力してください。"
    return ollama_chat(prompt, temperature=0.6)


def orchestrator_synthesize(topic: str, sections: dict[str, str]) -> str:
    """Orchestrator役: Workerの結果を統合して最終レポートに仕上げる。"""
    print("  [Orchestrator] 各セクションを統合して最終レポートを作成中…")
    lines = [f"# {topic} レポート\n"]
    for section_title, content in sections.items():
        lines.append(f"## {section_title}\n{content}\n")
    return "\n".join(lines)


def run_orchestrator_workers(topic: str) -> str:
    # 1. Orchestratorが動的にサブタスク(セクション)を計画
    plan = orchestrator_plan(topic)

    # 2. Workerたちが各セクションを並列に執筆
    print("\n  Workerへ並列にディスパッチ:")
    sections: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan)) as executor:
        future_to_section = {
            executor.submit(worker, topic, section): section for section in plan
        }
        for future in concurrent.futures.as_completed(future_to_section):
            section = future_to_section[future]
            sections[section] = future.result()

    # 元の順序に並び替え
    sections = {s: sections[s] for s in plan}

    # 3. Orchestratorが結果を統合
    print()
    final_report = orchestrator_synthesize(topic, sections)
    return final_report


if __name__ == "__main__":
    print(f"=== Orchestrator-Workers デモ (Ollama: {MODEL}) ===\n")
    try:
        report = run_orchestrator_workers("生成AI技術の企業導入")
        print("\n=== 最終レポート ===\n")
        print(report)
    except RuntimeError as e:
        print(f"\nエラー: {e}")
