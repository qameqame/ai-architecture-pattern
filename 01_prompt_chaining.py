"""
Pattern 1: Prompt Chaining (Ollamaローカルモデル版)
=====================================================
タスクを固定された順序のサブタスクに分解し、各ステップの出力を次の
ステップの入力として渡す。途中に"ゲート"(検証)を挟み、条件を満たさな
ければチェーンを打ち切ることもできる。

例: 生の顧客レビュー -> 要点抽出 -> 下書き作成 -> ゲート検証 -> 仕上げ

事前準備:
    1. Ollamaをインストールして起動: `ollama serve`
    2. モデルをpull: `ollama pull qwen2.5`
       (他のモデルを使う場合は下の MODEL 変数を変更)
"""

import json
import textwrap
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


def extract_key_points(raw_review: str) -> str:
    prompt = f"次のレビューから要点を3つ、簡潔な箇条書き(- で始める)で抽出してください。\n\nレビュー:\n{raw_review}"
    return ollama_chat(prompt, temperature=0.3)


def draft_copy(key_points: str) -> str:
    prompt = f"次の要点をもとに、製品の魅力を伝える広告コピーを2〜3文で書いてください。\n\n要点:\n{key_points}"
    return ollama_chat(prompt, temperature=0.7)


def gate_check(draft: str, min_length: int = 20) -> bool:
    """ステップ間のゲート: 条件を満たさなければチェーンを止める(LLM不要のプログラム的チェック)。"""
    ok = len(draft) >= min_length
    print(f"  [ゲート検証] 文字数={len(draft)} (閾値{min_length}) -> {'合格' if ok else '不合格'}")
    return ok


def polish_copy(draft: str) -> str:
    prompt = f"次の広告コピーを、行動喚起(CTA)を含む形に仕上げてください。文章はそのまま活かしつつ整えてください。\n\n{draft}"
    return ollama_chat(prompt, temperature=0.7)


def run_chain(raw_review: str) -> str | None:
    print("Step 1: 要点抽出")
    key_points = extract_key_points(raw_review)
    print(textwrap.indent(key_points, "    "))

    print("\nStep 2: 下書き作成")
    draft = draft_copy(key_points)
    print(textwrap.indent(draft, "    "))

    print("\nStep 3: ゲート検証")
    if not gate_check(draft):
        print("  条件未達のためチェーンを中断します。")
        return None

    print("\nStep 4: 仕上げ")
    final_copy = polish_copy(draft)
    print(textwrap.indent(final_copy, "    "))

    return final_copy


if __name__ == "__main__":
    raw_review = (
        "バッテリーが1日中持って助かる。カメラの画質もとても綺麗。"
        "価格が少し高いのが唯一の不満点。"
    )
    print("=== Prompt Chaining デモ (Ollama: " + MODEL + ") ===\n")
    print(f"入力: {raw_review}\n")
    try:
        result = run_chain(raw_review)
        print("\n=== 最終結果 ===")
        print(result if result else "(チェーンは途中で中断されました)")
    except RuntimeError as e:
        print(f"\nエラー: {e}")
