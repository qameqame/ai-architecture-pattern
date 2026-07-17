"""
Pattern 2: Parallelization (Ollamaローカルモデル版)
=====================================================
2つの派生パターンをデモする:

1. Sectioning(分割):   1つのタスクを独立した複数のサブタスクに分割し、
                        同時にLLMへ投げて結果を統合する。
2. Voting(多数決):     同じ入力に対して同じプロンプトを複数回(temperature
                        を上げてサンプリングのばらつきを出す)実行し、
                        多数決で結論を出す。

事前準備:
    1. Ollamaをインストールして起動: `ollama serve`
    2. モデルをpull: `ollama pull qwen2.5`
       (他のモデルを使う場合は下の MODEL 変数を変更)

注意: Ollamaはローカルの1インスタンスなので、並列にリクエストを送っても
実際の推論はサーバー側でキューイングされることが多い。それでも
「呼び出し側のコードが並列である」というパターンの構造自体は体験できる。
"""

import concurrent.futures
import json
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


def analyze_sentiment(review: str) -> str:
    prompt = f"次のレビューの感情を「ポジティブ」「ネガティブ」「ニュートラル」のいずれか一語だけで答えてください。\n\n{review}"
    return ollama_chat(prompt, temperature=0.0)


def analyze_topics(review: str) -> str:
    prompt = f"次のレビューが言及しているトピックを、価格/品質/配送/サポート/デザインの中から該当するものだけ、カンマ区切りで挙げてください。該当なしなら「特になし」と答えてください。\n\n{review}"
    return ollama_chat(prompt, temperature=0.0)


def analyze_risk_flags(review: str) -> str:
    prompt = f"次のレビューに返金・訴訟・危険・事故・苦情につながりそうな表現があれば、該当する語をカンマ区切りで挙げてください。なければ「リスクなし」とだけ答えてください。\n\n{review}"
    return ollama_chat(prompt, temperature=0.0)


def sectioning_demo(review: str):
    """独立した観点(感情/トピック/リスク)を並列に分析し、結果を統合する。"""
    print("=== Sectioning(分割並列)デモ ===")
    print(f"入力: {review}\n")

    tasks = {
        "sentiment": analyze_sentiment,
        "topics": analyze_topics,
        "risk_flags": analyze_risk_flags,
    }
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {
            executor.submit(fn, review): name for name, fn in tasks.items()
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            results[task] = future.result()
            print(f"  [{task}] 完了 -> {results[task]}")

    print("\n統合レポート:")
    print(f"  感情       : {results['sentiment']}")
    print(f"  トピック   : {results['topics']}")
    print(f"  リスク検知 : {results['risk_flags']}")


def classify_harmful(text: str) -> str:
    prompt = (
        "次のテキストが有害(脅迫・差別・暴力を含む)かどうかを判定し、"
        "「harmful」か「safe」のどちらか一語だけで答えてください。\n\n" + text
    )
    # temperatureを上げてサンプリングのばらつきを出し、多数決の意味を持たせる
    return ollama_chat(prompt, temperature=0.8).lower()


def voting_demo(text: str, n_votes: int = 5):
    """同じ判定タスクを複数回実行し、多数決で最終判定を決める。"""
    print("\n=== Voting(多数決並列)デモ ===")
    print(f"入力: {text}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_votes) as executor:
        futures = [executor.submit(classify_harmful, text) for _ in range(n_votes)]
        votes = [f.result() for f in futures]

    for i, v in enumerate(votes, 1):
        print(f"  投票{i}: {v}")

    harmful_count = sum("harmful" in v for v in votes)
    safe_count = sum("safe" in v for v in votes)
    final = "harmful" if harmful_count > safe_count else "safe"
    print(f"\n結果: harmful={harmful_count}, safe={safe_count} -> 最終判定: {final}")


if __name__ == "__main__":
    print(f"(使用モデル: {MODEL})\n")
    try:
        sectioning_demo(
            "配送がとても遅く残念でしたが、サポートの対応は満足でした。価格は妥当だと思います。"
        )
        voting_demo("この商品はとても使いやすくて満足しています。")
    except RuntimeError as e:
        print(f"\nエラー: {e}")
