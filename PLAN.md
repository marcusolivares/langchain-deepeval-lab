# Refactor plan

The remaining refactor steps for this repo. Steps 2 and 3 and an MCP bug fix
are done (see below); this file lists what is left so the work can continue on
any machine. Delete it once the plan is finished.

## Picking this up

Setup on a new machine (from [CLAUDE.md](CLAUDE.md)):

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks   # Conventional Commits hook
cp .env.example .env                  # then fill in the keys
pytest                                # offline tests: should be 20 passed, 1 xfailed
```

`.vscode/` is gitignored, so recreate it by hand if you use VS Code:

- `.vscode/settings.json`: `"python.testing.pytestEnabled": true`,
  `"python.testing.unittestEnabled": false`, `"python.testing.pytestArgs": ["tests"]`,
  `"python.terminal.useEnvFile": true`
- `.vscode/launch.json`: a test-debug config with `"subProcess": false`.
  Without it, debugging `test_instrumented_agent_calls_the_mcp_tool` hangs,
  because the debugger attaches to the MCP server child process.

  ```json
  {
      "version": "0.2.0",
      "configurations": [
          {
              "name": "Python: Debug Tests",
              "type": "debugpy",
              "request": "launch",
              "program": "${file}",
              "purpose": ["debug-test"],
              "console": "integratedTerminal",
              "justMyCode": false,
              "subProcess": false
          }
      ]
  }
  ```

How the work is done:

- **Ask before any paid run.** `deepeval test run evals/` costs about $0.50
  (`-m smoke` about half); a single live bot call is about a cent. Plain
  `pytest` is free: `tests/conftest.py` sets fake API keys.
- **One step at a time, one commit per logical change.** Review each diff
  before it is committed. Commit messages follow Conventional Commits.
- **Showcase, not lesson.** Docs and comments describe what the project does
  and shows, with no course or lesson wording.

## Done

| Commit | Change |
|---|---|
| `e929a0c` | test: offline unit tests in `tests/` with a fake chat model; plain `pytest` runs them |
| `feaba39` | fix(agents): MCP tools get a sync implementation, so `invoke()` can call `get_shipping_options` |
| `001e1d1` | refactor(agents): `instrumented.build_agent()` reuses `plain.build_agent()`; `SHOP_TOOLS` in `shop.py` |
| `2334088` | refactor(agents): every default bot is built on first use; `chat()` takes `model=` |

## Remaining

Order: the code-only parts of steps 4 and 5 can go now. Anything that
changes eval data (goldens, the chatbot conversation, expected outputs) waits
for step 1.

### Step 1: one source of truth for the shop data (blocked on a decision)

**Decision needed:** which values are correct. The same policies live in four
places and disagree:

| Fact | `agents/tools/policy_search.py` (RAG knowledge base) | `agents/tools/mcp_server.py` / `policies.txt` |
|---|---|---|
| Express shipping | $15 (line 54) | $12.99 |
| Standard shipping | 5–7 days (line 50) | 3–5 days |
| Damaged food | refund if reported within 24 h | "no exceptions" / any damaged item within 48 h |
| Refund time | 3–5 days (clothing) | 5–7 days |
| Return window | counted from delivery | counted from purchase |

Jewellery, international shipping and cancellation exist only in the RAG
knowledge base; overnight shipping only in the MCP tool and `policies.txt`.
`get_refund_policy` (`REFUND_POLICIES` in `shop.py`) has its own shorter
wording too.

Then:

- Put the facts in one place, for example `data/policies.md` (one section per
  policy) and `data/orders.json`.
- Build the RAG chunks from its sections. Make `REFUND_POLICIES`, the MCP
  shipping tool and `evals/synthesize_goldens.py` read from it too.
- Update the expected outputs in `evals/datasets/rag_agent.json` (the express
  shipping golden says $15).
- Regenerate the synthesized goldens and rerun the evals (both paid).
- Add an offline test that the RAG chunks and the tools agree.

### Step 4: chatbot loop (`agents/chatbot.py`)

- `while True` (line 97) has no step limit. Add a maximum number of model
  calls per turn.
- `TOOLS_BY_NAME[tc["name"]]` (line 107) raises `KeyError` when the model
  names a tool that does not exist. Return an error `ToolMessage` instead, so
  the model can recover.
- The docstring (lines 14–17) says the manual loop is needed to keep the
  history across turns. That is no longer true: `create_agent` with an
  `InMemorySaver` checkpointer and a `thread_id` keeps the full history.
  Recommended: keep the manual loop as the one bot that shows `bind_tools()`
  directly, and correct the docstring.
- Offline tests with the fake model (`tests/fakes.py`): the step limit, and an
  unknown tool name.
- After step 1: turn 4 of `SUPPORT_CONVERSATION` (`evals/common.py:66`) asks
  about express shipping, but the chatbot has no shipping tool. Either give it
  `get_shipping_options` or change the turn. This changes eval scores (paid
  rerun).

### Step 5: evals, repeated code and coverage

- `traced(support_agent, name="support_agent")` is repeated in 4 files and
  `traced(rag_support_agent, ...)` in 1. Define the traced runners once in
  `evals/common.py`.
- `get_judge()` is called in 9 files. `@cache` it.
- Move the inline `GOLDENS` lists to `evals/datasets/*.json`. Load them with
  one helper generalized from `load_goldens()` in
  `evals/test_agent_synthesized.py`.
- No eval checks that the agent picks the MCP tool. Add a golden with
  `expected_tools=[ToolCall(name="get_shipping_options")]` to
  `evals/test_agent_tracing.py` (paid to verify).
- `evals/test_agent_quality.py:25-27` hand-copies part of the system prompt for
  `PromptAlignmentMetric`. Use `plain.SYSTEM_PROMPT` instead. This changes what
  the metric checks (paid rerun).
- Offline test for `traced()`: that it copies `expected_output` and
  `expected_tools` onto the trace. Only with no Confident AI key loaded, since
  under plain pytest DeepEval would upload the trace. Check how to stop that
  in `tests/conftest.py` first.

### Step 6: packaging, layout and CI

- Replace `requirements.txt` with `pyproject.toml` plus a lock file (uv). Move
  the pytest config into it. An editable install removes the "run as a module
  from the project root" rule.
- Rename the `agents` package, for example to `shopease`. The OpenAI Agents
  SDK also installs a top-level `agents` module, so the two cannot share an
  environment. A `src/` layout is optional.
- GitHub Actions:
  - `tests.yml` runs `pytest` on every PR. It needs no secrets.
  - `evals.yml` runs `deepeval test run evals/ -m smoke` by hand
    (`workflow_dispatch`) or nightly, with the API keys as secrets.
- Delete `evals/INTRO_WHY_AI_EVALS.md`. It is leftover course material and
  points to `test_agent.py`, which no longer exists.

Proposed layout:

```
pyproject.toml, uv.lock
src/shopease/
  config.py, prompts.py
  data/          orders.json, policies.md   (the only copy of the facts)
  tools/         shop.py, policy_search.py, mcp_server.py
  agents/        plain.py, instrumented.py, rag.py, chatbot.py
tests/           offline, fake models, plain pytest, every PR
evals/           paid, deepeval test run
  conftest.py, common.py, datasets/*.json, test_*.py
.github/workflows/  tests.yml (PR), evals.yml (manual/nightly)
```

### Open bug, decide separately

`get_order_status` does not return the order's category, so the agent cannot
look up the refund policy for "I want to return order ORD-2099". The README's
Results section presents this as a real finding. A strict `xfail` in
`tests/test_shop_tools.py` and a non-strict one in `evals/datasets/agent_quality.json`
record it. To fix it, include the category in the tool's output and remove
both marks. Confirming the eval now passes needs a paid run.

## References

- [LangChain: unit testing](https://docs.langchain.com/oss/python/langchain/test/unit-testing) and [integration testing](https://docs.langchain.com/oss/python/langchain/test/integration-testing)
- [langchain-ai/react-agent template](https://github.com/langchain-ai/react-agent)
- [DeepEval: LangChain integration](https://deepeval.com/integrations/frameworks/langchain), [end-to-end evals](https://deepeval.com/docs/evaluation-end-to-end-single-turn), [regression testing in CI/CD](https://deepeval.com/guides/guides-regression-testing-in-cicd)
- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
