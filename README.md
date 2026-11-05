# langchain-deepeval-lab

A teaching repo that builds customer-support bots with **LangChain** and
evaluates them with **DeepEval**: a tool-calling agent, a RAG agent and a
multi-turn chatbot, each with its own evals.

The LLM vendor is a setting, not code. Claude runs everything by default;
set `LLM_PROVIDER=openai` to switch to GPT-4o, and `JUDGE_PROVIDER` to have
the evals judged by a different vendor than the one being tested.

---

## Files in this repo

| File | Who writes it (in real life) | What it does |
|---|---|---|
| `agent_plain.py` | Developer (before evals exist) | A normal LangChain agent with two tools. No DeepEval code anywhere. |
| `agent_instrumented.py` | Developer (after QA asks) | The **same** agent, with a few lines added to make it observable to DeepEval. |
| `rag_agent.py` | Developer | An agent that answers from a small vector store of policy documents. |
| `chatbot.py` | Developer | A multi-turn chatbot that runs its own tool-call loop. |
| `config.py` | Developer | Picks the LLM vendor and model for everything else. |
| `shop_tools.py` | Developer | The demo store's data and the tools the bots share. |
| `evals/` | QA / Tester (you) | Imports the bots, defines goldens, runs metrics. |

---

## The teaching point

The diff between `agent_plain.py` and `agent_instrumented.py` is a handful
of lines. That's the entire ask QA makes of dev:

1. `from deepeval.integrations.langchain import CallbackHandler`
2. `deepeval_callback = CallbackHandler()`
3. `config={"callbacks": [deepeval_callback]}` on `.invoke()`
4. `update_current_trace(output=reply)` to put the clean reply on the trace

The eval side wraps the agent with `@observe` (see `traced()` in
`evals/common.py`), and everything else stays exactly the same.

Show students both files side by side. Once they see the diff is tiny,
they stop being intimidated by component-level evaluation.

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API keys: copy the template and fill it in
cp .env.example .env

# 3. (Optional but recommended) log in to Confident AI to see traces in a UI
deepeval login

# 4. Sanity check: run the agent on its own
python agent_instrumented.py

# 5. Run an evaluation (from the project root, as a module)
python -m evals.test_TracingComponentsTest
```

The other evals in `evals/` run the same way, for example
`python -m evals.test_rag_agent` or `python -m evals.test_chatbot`.

To try the other vendor for one run, prefix the command:
`LLM_PROVIDER=openai python agent_instrumented.py`
(PowerShell: `$env:LLM_PROVIDER="openai"; python agent_instrumented.py`).

---

## What you'll see

`test_TracingComponentsTest.py` makes DeepEval:
1. Invoke `support_agent(...)` for each golden input
2. Capture the full trace (LangChain LLM call + tool call + final message)
3. Apply **Task Completion** to the whole trace
4. Apply **Tool Correctness** to the tool-call span
5. Print scores + reasons per golden

Then run `test_TaskCompletion.py`, which checks the same answer **without**
a trace. The judge can no longer see that the agent called a tool, assumes
the answer was made up, and fails it. That's the moment that makes
component-level evaluation click.

---

## Suggested lesson flow

| Minute | Activity |
|---|---|
| 0–5 | Show `agent_plain.py`. Ask: "How would you test this?" |
| 5–10 | Show `agent_instrumented.py` side-by-side. Walk through the diff. |
| 10–25 | Walk through `evals/test_TracingComponentsTest.py` line by line. |
| 25–35 | Run it live, then run `evals/test_TaskCompletion.py` and compare. |
| 35+ | Open discussion: what other metrics? what other goldens? |

---

## Reality checks before teaching

DeepEval's integration surface evolves. Before your first session, run the
demo end-to-end once and confirm:

- The model IDs in `MODELS` in `config.py`: if one errors, swap in the
  latest from the vendor's console.
- `from deepeval.integrations.langchain import CallbackHandler`: works as
  written in recent DeepEval versions; older versions may use a different
  import path.
- `Golden(expected_tools=[ToolCall(name="...")])`: accepted in current
  DeepEval; in older versions you may need to pass via `additional_metadata`
  and unpack via `update_current_span`.

Fix any version issue once, then it's solid for the whole course.

---

## Contributing

Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Enable the hook that checks them once per clone:

```bash
git config core.hooksPath .githooks
```
