"""
Pattern 6: Agentic Loop (Ollamaローカルモデル版)
===================================================
これまでの5パターン(Prompt Chaining / Parallelization / Routing /
Evaluator-Optimizer / Orchestrator-Workers)は、いずれも "Workflow" ——
つまり制御フロー(どの順で何を呼ぶか)を人間(コード)側があらかじめ
決めているパターンでした。

このスクリプトはそれとは異なる "Agent" パターンです。制御フローその
ものをLLMに委ねます。LLMは各ステップで
  (a) どのツールを呼ぶか
  (b) どんな引数で呼ぶか
  (c) ツールの実行結果(環境からのフィードバック)を見て、次に何をするか
  (d) いつ十分な情報が揃って最終回答を出すか
を自律的に判断します。人間側が用意するのは「使えるツールのリスト」と
「終了条件(最大ステップ数などの安全装置)」だけです。

    ツールが必要か判断 → ツール実行 → 結果を見る → …(繰り返し)… → 最終回答

事前準備:
    1. Ollamaをインストールして起動: `ollama serve`
    2. モデルをpull: `ollama pull qwen2.5`
       (Ollamaの Tool Calling 機能に対応したモデルが必要。qwen2.5 / llama3.1 /
        mistral-nemo などが対応している)

注意: ローカルの小型モデルはツール呼び出しの判断を誤ったり、呼び出しを
やめるべきタイミングを見誤ったりすることがあります。無限ループを防ぐ
ため、必ず `max_steps` のような安全装置を入れてください。
"""

import ast
import json
import operator
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5"

SYSTEM_PROMPT = (
    "あなたはツールを使って質問に答えるアシスタントです。"
    "回答に必要な情報が手元にない場合は、適切なツールを呼び出してください。"
    "複数のツールを順番に使う必要があれば、1回の応答につき1つずつ呼び出して構いません。"
    "十分な情報が揃ったら、ツールを呼ばずに日本語で最終回答をしてください。"
    "ツールで調べても情報が見つからない場合は、正直に「わかりません」と答えてください。"
)


# ---------------------------------------------------------------------------
# ツール定義(LLMに見せるスキーマ)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "四則演算・べき乗の式を計算する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "計算したい式。例: '333 * 3' や '(12 + 8) / 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "社内の小さな知識ベースから、建造物の高さなどの事実を検索する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "調べたい対象。例: '東京タワーの高さ'",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_text_length",
            "description": "与えられたテキストの文字数を数える。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "文字数を数えたいテキスト"}
                },
                "required": ["text"],
            },
        },
    },
]

# ダミーの知識ベース。存在しないトピックを聞かれた場合の挙動(環境フィードバック
# を見てエージェントがどう振る舞うか)も確認できるよう、あえて少数にしてある。
KNOWLEDGE_BASE = {
    "東京タワーの高さ": "333メートル",
    "富士山の標高": "3776メートル",
    "東京スカイツリーの高さ": "634メートル",
}


# ---------------------------------------------------------------------------
# ツールの実装(=エージェントにとっての「環境」)
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """evalを直接使わず、四則演算だけを許可する安全な式評価器。
    ツール実行は外部からの入力(LLMが組み立てた式)を扱うので、
    任意コード実行を許さない設計にしている。
    """

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("許可されていない式です")

    tree = ast.parse(expr, mode="eval").body
    return _eval(tree)


def execute_tool(name: str, args: dict) -> str:
    """ツールを実行し、結果を文字列で返す。例外はここで握りつぶし、
    'エラー: ...' という文字列としてエージェントに返す。
    こうすることで、ツール実行の失敗もエージェントへの"環境フィードバック"
    の一部として扱える(プロセスをクラッシュさせない)。
    """
    try:
        if name == "calculator":
            result = _safe_eval(args["expression"])
            return str(result)

        if name == "lookup_fact":
            topic = args.get("topic", "")
            for key, value in KNOWLEDGE_BASE.items():
                if key in topic or topic in key:
                    return value
            return "情報が見つかりませんでした。"

        if name == "get_text_length":
            text = args.get("text", "")
            return f"{len(text)}文字"

        return f"不明なツールです: {name}"
    except Exception as e:  # noqa: BLE001 - ツール実行エラーはそのまま結果として返す
        return f"エラー: {e}"


# ---------------------------------------------------------------------------
# Ollama呼び出し(Tool Calling対応)
# ---------------------------------------------------------------------------


def ollama_chat_with_tools(messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Ollamaに接続できませんでした。`ollama serve` が起動しているか、"
            f"モデル `{MODEL}` を `ollama pull {MODEL}` 済みか確認してください。詳細: {e}"
        ) from e
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Ollamaがエラーを返しました(モデル `{MODEL}` がTool Callingに対応していない"
            f"可能性があります。qwen2.5 / llama3.1 等をお試しください)。詳細: {body}"
        ) from e


# ---------------------------------------------------------------------------
# エージェントループ本体
# ---------------------------------------------------------------------------


def run_agent(user_task: str, max_steps: int = 6) -> str | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    print(f"ユーザー: {user_task}\n")

    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")
        response = ollama_chat_with_tools(messages)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            final_answer = message.get("content", "").strip()
            print(f"  [Agent] ツール呼び出しなし → 最終回答を出力\n")
            return final_answer

        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"]
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            print(f"  [Agent] ツール呼び出し: {name}({args})")
            result = execute_tool(name, args)
            print(f"  [環境]  実行結果: {result}")

            messages.append({"role": "tool", "name": name, "content": result})

    print("  [Agent] 最大ステップ数に到達しました。安全装置によりループを終了します。")
    return None


if __name__ == "__main__":
    print(f"=== Agentic Loop デモ (Ollama: {MODEL}) ===\n")

    try:
        # シナリオ1: 複数ツールを自律的に組み合わせて解く
        # (lookup_fact で東京タワーの高さを調べ、その結果を calculator に渡す
        #  ——という手順をコードでは一切指定していない。LLMが自分で組み立てる)
        print("### シナリオ1: 多段ツール利用 ###\n")
        answer1 = run_agent(
            "東京タワーの高さを調べて、その3倍が何メートルになるか教えて。"
        )
        print(f"最終回答: {answer1}\n")

        print("=" * 60 + "\n")

        # シナリオ2: 知識ベースにない情報を聞き、環境フィードバック
        # (「情報が見つかりませんでした」)を受けてエージェントがどう振る舞うか
        print("### シナリオ2: 情報が見つからない場合の挙動 ###\n")
        answer2 = run_agent(
            "パリのエッフェル塔の高さを調べて、その2倍が何メートルになるか教えて。"
        )
        print(f"最終回答: {answer2}")

    except RuntimeError as e:
        print(f"\nエラー: {e}")
