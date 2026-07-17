"""
Pattern 3: Routing (Ollamaローカルモデル版)
=============================================
入力をまず分類(ルーティング)し、その結果に応じて専門化された処理
(プロンプト/モデル/ツール)へ振り分ける。「1つの万能プロンプト」より
各カテゴリに最適化した処理の方が精度が上がりやすい。

例: 問い合わせチケットを分類し、カテゴリ専用のシステムプロンプトを
持つハンドラーに振り分ける。

事前準備:
    1. Ollamaをインストールして起動: `ollama serve`
    2. モデルをpull: `ollama pull qwen2.5`
       (他のモデルを使う場合は下の MODEL 変数を変更)
"""

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5"

CATEGORIES = ["billing", "technical", "account", "other"]


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


def classify_ticket(ticket: str) -> str:
    prompt = (
        f"次の問い合わせを {', '.join(CATEGORIES)} のいずれか一語だけに分類してください。"
        "他の単語は含めないでください。\n\n問い合わせ:\n" + ticket
    )
    raw = ollama_chat(prompt, temperature=0.0).lower()
    for category in CATEGORIES:
        if category in raw:
            return category
    return "other"


def handle_billing(ticket: str) -> str:
    system = "あなたは請求担当のサポート窓口です。丁寧かつ具体的に、請求・支払いに関する対応方針を1〜2文で答えてください。"
    return "[請求担当] " + ollama_chat(ticket, system=system)


def handle_technical(ticket: str) -> str:
    system = "あなたは技術サポート担当です。丁寧かつ具体的に、技術的なトラブルへの対応手順を1〜2文で答えてください。"
    return "[技術サポート] " + ollama_chat(ticket, system=system)


def handle_account(ticket: str) -> str:
    system = "あなたはアカウント管理担当です。丁寧かつ具体的に、アカウント・ログインに関する対応方針を1〜2文で答えてください。"
    return "[アカウント担当] " + ollama_chat(ticket, system=system)


def handle_other(ticket: str) -> str:
    system = "あなたは一般問い合わせ担当です。丁寧に、内容確認と担当部署への転送方針を1〜2文で答えてください。"
    return "[一般担当] " + ollama_chat(ticket, system=system)


ROUTES = {
    "billing": handle_billing,
    "technical": handle_technical,
    "account": handle_account,
    "other": handle_other,
}


def route_ticket(ticket: str) -> str:
    category = classify_ticket(ticket)
    print(f"  分類結果: {category}")
    handler = ROUTES[category]
    return handler(ticket)


if __name__ == "__main__":
    tickets = [
        "先月の請求金額が想定より高いです。確認してください。",
        "アプリを開くとすぐ落ちてしまい、使えません。",
        "パスワードを忘れてログインできません。",
        "御社のサービスについて一般的な質問があります。",
    ]

    print(f"=== Routing デモ (Ollama: {MODEL}) ===\n")
    try:
        for ticket in tickets:
            print(f"入力: {ticket}")
            response = route_ticket(ticket)
            print(f"  応答: {response}\n")
    except RuntimeError as e:
        print(f"\nエラー: {e}")
