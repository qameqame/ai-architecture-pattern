🇯🇵 日本語版はこちら → [README.ja.md](README.ja.md)

# Agent Design Patterns Tutorial (Local Ollama Edition)

A hands-on tutorial for the five foundational patterns from Anthropic's ["Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents), run entirely against a local LLM via Ollama.

No API key is required — every script talks to Ollama running on your own machine.

## What you'll learn

| # | File | Pattern | In one line |
|---|---|---|---|
| 1 | `01_prompt_chaining.py` | Prompt Chaining | Break a task into sequential steps, each feeding the next |
| 2 | `02_parallelization.py` | Parallelization | Run multiple LLM calls at once (sectioning / voting) |
| 3 | `03_routing.py` | Routing | Classify input and dispatch to a specialized handler |
| 4 | `04_evaluator_optimizer.py` | Evaluator-Optimizer | Generate → evaluate → refine in a loop until it passes |
| 5 | `05_orchestrator_workers.py` | Orchestrator-Workers | A lead LLM dynamically breaks up work and dispatches it to worker LLMs |

Each file is a standalone, independently runnable script. Working through them in order is recommended, but feel free to jump around.

---

## Step 0: Prerequisites

### 0-1. Install Ollama

If you haven't already, install Ollama from [ollama.com](https://ollama.com).

### 0-2. Start the Ollama server

```bash
ollama serve
```

Skip this if Ollama is already running in the background (e.g. as a menu-bar app, or inside Docker — see the Docker note below).

### 0-3. Pull a model

These scripts default to `qwen2.5`.

```bash
ollama pull qwen2.5
```

To use a different model (e.g. `llama3.1`, `gpt-oss:20b`), edit this line near the top of each `.py` file:

```python
MODEL = "qwen2.5"
```

### 0-4. Sanity check

```bash
ollama run qwen2.5 "hello"
```

If you get a response, you're set. The Python side uses only the standard library (`urllib`) to call Ollama's REST API (`http://localhost:11434/api/chat`), so no `pip install` is needed.

**Running Ollama in Docker?** As long as the container publishes port 11434 to the host (e.g. `-p 11434:11434` or `0.0.0.0:11434->11434/tcp` in `docker ps`), the scripts work unmodified — they just hit `localhost:11434`. To run the CLI check above, use `docker exec -it <container_name> ollama run qwen2.5 "hello"` instead of a bare `ollama run` (a bare `docker -c ollama ...` is not valid — `-c` selects a Docker *context*, not a container).

---

## Step 1: Prompt Chaining (`01_prompt_chaining.py`)

### Concept

Decompose one big task into a fixed sequence of smaller steps. Each step's output becomes the next step's input. You can insert a **gate** (a plain, non-LLM check) partway through and abort the chain if it fails.

```
raw review → [extract key points] → [draft copy] → [gate check] → [polish] → final copy
```

### Run it

```bash
python3 01_prompt_chaining.py
```

### What's happening

- Steps 1, 2, and 4 are LLM calls (`ollama_chat()`)
- Step 3's `gate_check()` doesn't use the LLM at all — it's just a character-count check. The point: **a gate doesn't have to be an LLM call**
- If the gate fails, the chain stops and later steps never run

### Try it yourself

Bump `gate_check()`'s `min_length` up to something large (e.g. `500`) and re-run to watch the chain get cut short.

---

## Step 2: Parallelization (`02_parallelization.py`)

### Concept

Two variants:

- **Sectioning**: fire off independent subtasks (sentiment, topics, risk flags) at the same time and combine the results
- **Voting**: run the same judgment task several times (with higher temperature for sampling variety) and take a majority vote

### Run it

```bash
python3 02_parallelization.py
```

### What's happening

`concurrent.futures.ThreadPoolExecutor` fires several requests concurrently. Note that **Ollama is a single local instance, so the actual inference is usually queued server-side and processed close to sequentially in practice** — but the calling code's structure is genuinely parallel, which is the point of the exercise.

### Try it yourself

Increase `n_votes` in `voting_demo()`, or lower `classify_harmful()`'s `temperature` to `0.0`, and compare how much the votes vary.

---

## Step 3: Routing (`03_routing.py`)

### Concept

Classify the input first, then dispatch to a handler specialized for that category. The idea: a handler tuned for one category tends to beat a single do-everything prompt.

```
ticket → [classifier] → billing / technical / account / other → specialized handler
```

### Run it

```bash
python3 03_routing.py
```

### What's happening

- `classify_ticket()` uses a classification-only prompt to pick a category
- Each `handle_*()` function calls the LLM with a different `system` prompt tailored to that category (same model, different role)

### Try it yourself

Add your own ticket text to the `tickets` list and check whether it's classified as you'd expect. If it's not, tightening the prompt in `classify_ticket()` is a good way to feel the effect of prompt tuning.

---

## Step 4: Evaluator-Optimizer Workflow (`04_evaluator_optimizer.py`)

### Concept

One LLM (the Generator) produces an output; a separate pass (the Evaluator) critiques it and returns feedback. The loop — generate → evaluate → revise — repeats until the output passes.

```
generate → evaluate ─┬─ pass → done
                      └─ fail → regenerate with feedback → evaluate → …
```

### Run it

```bash
python3 04_evaluator_optimizer.py
```

### What's happening

- `evaluate_copy()` checks "character count" and "keyword present" in plain Python code, and only asks the LLM to judge the one criterion that's hard to check mechanically: whether the tone is positive. **Deciding what needs an LLM and what a simple check can handle is itself part of the design**
- On failure, the specific critique is folded into the next generation prompt

### Try it yourself

Set `max_iterations` to `1` and see what happens when the first attempt doesn't pass. Then add a new check to `evaluate_copy()` (e.g. "no emoji allowed").

---

## Step 5: Orchestrator-Workers Workflow (`05_orchestrator_workers.py`)

### Concept

A central Orchestrator (LLM) **dynamically** breaks the task into subtasks based on the input (unlike Step 2's Sectioning, the subtasks aren't fixed in advance — their number and content vary). Each subtask goes to a Worker (LLM), run in parallel, and the Orchestrator synthesizes the results at the end.

```
topic → [Orchestrator: plan] → section list (decided dynamically)
                                     ↓ dispatch in parallel
                  Worker1  Worker2  Worker3  Worker4
                     ↓        ↓        ↓        ↓
                  [Orchestrator: synthesize] → final report
```

### Run it

```bash
python3 05_orchestrator_workers.py
```

### What's happening

- `orchestrator_plan()` asks the LLM to propose section headings for the given topic, then parses that output into a list
- `worker()` drafts each section in parallel
- `orchestrator_synthesize()` assembles everything into the final report

### Try it yourself

Change the topic passed to `run_orchestrator_workers()` (e.g. `"the science of dieting"`) and watch the Orchestrator come up with a different section structure each time.

---

## Troubleshooting

**"Could not connect to Ollama" error**
Make sure `ollama serve` is running. Try `ollama run qwen2.5 "test"` in another terminal to isolate the problem.

**Model not found error**
Run `ollama pull qwen2.5` (or whichever model name you're using).

**Responses are slow**
Local LLM speed depends on your hardware. Steps 2 and 5 make several LLM calls each, so they take longer. A smaller model (e.g. `qwen2.5:3b`) will speed things up.

**Classification or parsing isn't reliable**
Small local models sometimes follow instructions less reliably than large hosted models. Sharpening the prompt, lowering `temperature`, or adding one example of the expected output format usually helps.

---

## Next step: swap in the real Anthropic API

Replace `ollama_chat()` in any file with an Anthropic API call and the same pattern structure keeps working against a larger, cloud-hosted model:

```python
from anthropic import Anthropic
client = Anthropic()

def call_llm(prompt: str, system: str | None = None) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
```

The control flow of each pattern — chaining, parallelizing, branching, evaluate-loop, dynamic decomposition — is model-agnostic, so almost everything else stays the same.

## References

- Anthropic, "Building Effective Agents": https://www.anthropic.com/engineering/building-effective-agents
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
