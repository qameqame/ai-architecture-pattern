"""
Pattern 4: Evaluator-Optimizer Workflow (Ollamaローカルモデル版)
===================================================================
1つのLLM(Generator)が出力を作り、別のLLM(Evaluator)がそれを評価して
フィードバックを返す。基準を満たすまで、フィードバックを踏まえて
生成 -> 評価 のループを繰り返す。

例: 短い広告コピーを生成し、評価者が「文字数」「キーワード含有」
「行動喚起の有無」「トーン」をチェック。不合格ならフィードバックを
付けて再生成。

事前準備:
    1. Ollamaをインストールして起動: `ollama serve`
    2. モデルをpull: `ollama pull qwen2.5`
       (他のモデルを使う場合は下の MODEL 変数を変更)
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5"


@dataclass
class EvalResult:
    passed: bool
    feedback: str


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


def generate_copy(topic: str, feedback: str | None = None) -> str:
    prompt = f"「{topic}」の広告コピーを日本語で1〜2文書いてください。コピー本文だけを出力してください。"
    if feedback:
        prompt += f"\n\n前回の指摘事項を必ず反映してください: {feedback}"
    return ollama_chat(prompt, temperature=0.8)


def _llm_yes_no(question: str, copy_text: str) -> bool:
    """LLMに「はい/いいえ」だけで答えさせ、判定を取り出す小さなヘルパー。"""
    prompt = f"{question}「はい」か「いいえ」のどちらか一語だけで答えてください。\n\n{copy_text}"
    answer = ollama_chat(prompt, temperature=0.0)
    return "はい" in answer


def evaluate_copy(topic: str, copy_text: str) -> EvalResult:
    """判定基準の使い分け:
    - 「文字数」「商品名の有無」は機械的にプログラムでチェックできるのでコードで判定
    - 「行動を促す表現(CTA)があるか」「トーンが前向きか」は自然文の言い回しが
      無数にあり、固定キーワードの完全一致チェックでは判定しきれないため、
      LLMに判定させる(実際に「今すぐ」「ぜひ」という単語の有無だけで判定すると、
      "体感しよう" のような別表現のCTAを誤って不合格にしてしまう)
    """
    issues = []
    if len(copy_text) < 20:
        issues.append("文字数が20文字未満です。もっと具体的に書いてください。")
    if topic not in copy_text:
        issues.append(f"商品名「{topic}」が含まれていません。")

    cta_ok = _llm_yes_no(
        "次の広告コピーには、読み手に行動を促す表現"
        "(例: 今すぐ試す、ぜひ購入する、チェックする、体感する、手に入れる 等の呼びかけ)"
        "が含まれていますか？",
        copy_text,
    )
    if not cta_ok:
        issues.append("行動を促す表現(CTA)が含まれていません。")

    tone_ok = _llm_yes_no(
        "次の広告コピーは前向きで魅力的なトーンになっていますか？", copy_text
    )
    if not tone_ok:
        issues.append("トーンが前向き・魅力的になっていません。")

    if not issues:
        return EvalResult(passed=True, feedback="")
    return EvalResult(passed=False, feedback=" / ".join(issues))


def run_evaluator_optimizer(topic: str, max_iterations: int = 4) -> str:
    feedback = None
    copy_text = ""

    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i} ---")
        copy_text = generate_copy(topic, feedback)
        print(f"  生成: {copy_text}")

        result = evaluate_copy(topic, copy_text)
        if result.passed:
            print("  評価: 合格 ✅")
            return copy_text

        print(f"  評価: 不合格 ❌ -> フィードバック: {result.feedback}")
        feedback = result.feedback

    print("最大反復回数に到達しました。最後の生成結果を返します。")
    return copy_text


if __name__ == "__main__":
    print(f"=== Evaluator-Optimizer デモ (Ollama: {MODEL}) ===\n")
    try:
        final_copy = run_evaluator_optimizer("ワイヤレスイヤホン")
        print("\n=== 最終結果 ===")
        print(final_copy)
    except RuntimeError as e:
        print(f"\nエラー: {e}")
