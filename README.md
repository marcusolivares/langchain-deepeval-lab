# langchain-deepeval-lab

Three customer-support bots for a demo store, **ShopEasy**, built with
**LangChain** and evaluated with **DeepEval**:

- a **tool-calling agent** that also loads a tool from an **MCP server**
- a **RAG agent** that answers from a vector store of policy documents
- a **multi-turn chatbot** that runs its own tool-call loop

Each bot has its own eval suite. The suites cover tool use, answer quality,
retrieval quality, multi-turn memory, custom LLM-judged criteria and safety
checks on synthetic test data.

The model vendor is a setting. Claude runs everything by default.
`LLM_PROVIDER=openai` switches the bots to GPT-4o, and `JUDGE_PROVIDER` lets
a different vendor grade the evals so that no model judges its own answers.

---

## Highlights

- **Provider-agnostic.** Only `config.py` names a vendor or model. Every bot
  exposes a factory (`build_agent(llm=None)`, `build_model(llm=None)`) that
  takes any LangChain chat model.
- **Trace-level evaluation.** The agents send every LLM call and tool call to
  DeepEval as spans, so metrics see the whole run and not only the final
  reply.
- **RAG metrics.** The RAG agent attaches its retrieved chunks to the trace,
  so the retriever (contextual precision and recall) and the generator
  (faithfulness, relevancy) are scored separately.
- **Multi-turn evals.** The chatbot is driven live through a scripted
  conversation and scored on turn relevancy, knowledge retention and
  completeness.
- **Synthetic goldens.** DeepEval's Synthesizer generates test inputs from
  `policies.txt`, and the agent's answers are checked for bias, toxicity and
  PII leakage.
- **MCP.** `agent_instrumented.py` starts `mcp_server.py` over stdio and
  merges its tool with the local ones.

---

## Instrumenting an agent

Making a LangChain agent observable to DeepEval takes four additions:

```python
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace

deepeval_callback = CallbackHandler()

def support_agent(user_input: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )
    reply = result["messages"][-1].text
    update_current_trace(output=reply)
    return reply
```

The eval opens the trace and attaches the expected values from each golden:

```python
support_agent = traced(_support_agent, name="support_agent")  # evals/common.py

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")]),
])

for golden in dataset.evals_iterator(metrics=[task_completion, tool_correctness]):
    support_agent(golden.input)
```

`agent_plain.py` is the same agent without any DeepEval code, so you can
diff the two files.

---

## Eval suite

| Script | Bot | Metrics | What it covers |
|---|---|---|---|
| `test_agent_tracing` | Agent | TaskCompletion, ToolCorrectness | Did the agent finish the task and pick the right tool? |
| `test_agent_quality` | Agent | AnswerRelevancy, PromptAlignment, StepEfficiency | Reply quality, system-prompt adherence, wasted steps |
| `test_agent_geval` | Agent | GEval | Custom correctness criteria against an expected answer |
| `test_agent_untraced` | Agent | TaskCompletion | The same check on the bare reply with no trace. It fails because the judge cannot see the tool call, which shows why trace-level evals matter |
| `test_rag_agent` | RAG agent | ContextualPrecision, ContextualRecall, AnswerRelevancy, Faithfulness | Retrieval and generation scored separately |
| `test_chatbot` | Chatbot | TurnRelevancy, KnowledgeRetention, ConversationCompleteness | A four-turn conversation that tests memory |
| `test_chatbot_geval` | Chatbot | ConversationalGEval | Custom criteria over the whole conversation, including the tools called on each turn |
| `test_agent_synthesized` | Agent | Bias, Toxicity, PIILeakage | Safety checks on Synthesizer-generated goldens |

Every metric receives the judge model explicitly, so a Claude-only setup
never falls back to DeepEval's default OpenAI judge.

---

## Results

One run on 2026-11-05, with Claude Sonnet 4.6 as both bot and judge. Judge
cost was about $0.35 in total. Each suite has only 1–3 test cases, so treat
these scores as a snapshot, not a benchmark.

| Suite | Metric | Score | Passed |
|---|---|---|---|
| `test_agent_tracing` | Task Completion | 0.95 | 2/2 |
| | Tool Correctness | 1.00 | 2/2 |
| `test_agent_quality` | Answer Relevancy | 0.96 | 3/3 |
| | Prompt Alignment | 0.67 | 2/3 |
| | Step Efficiency | 0.53 | 2/3 |
| `test_agent_geval` | Correctness (GEval) | 0.90 | 1/1 |
| `test_agent_untraced` | Task Completion | 0.30 | 0/1 |
| `test_rag_agent` | Contextual Precision | 1.00 | 2/2 |
| | Contextual Recall | 1.00 | 2/2 |
| | Answer Relevancy | 1.00 | 2/2 |
| | Faithfulness | 1.00 | 2/2 |
| `test_chatbot` | Turn Relevancy | 1.00 | 1/1 |
| | Knowledge Retention | 1.00 | 1/1 |
| | Conversation Completeness | 0.75 | 1/1 |
| `test_chatbot_geval` | Correctness (ConversationalGEval) | 0.60 | 0/1 |

Scores are means across a suite's test cases. `test_agent_synthesized` is not
included because it needs an OpenAI key.

What the failures show:

- **Trace vs no trace.** The same Task Completion check scores 0.95 on the
  trace and 0.30 on the bare reply. Without the trace, the judge cannot see
  the `get_order_status` call and assumes the answer was made up.
- **A real agent bug.** For *"I want to return order ORD-2099, what should I
  do?"* the agent called no tools and asked for the product category several
  times in one reply. Prompt Alignment scored 0.0 and Step Efficiency 0.1.
  The cause is a gap in the tools: `ORDERS` stores each order's category,
  but `get_order_status` does not return it.
- **A chatbot inconsistency.** The judge saw the `get_order_status` call
  in the transcript. It also caught that in turn 2 the bot said it had no
  ETA right after stating one.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys

python -m agents.agent_instrumented   # run a bot on its own
python -m agents.chatbot              # interactive; type 'quit' to exit

python -m evals.test_agent_tracing    # run an eval suite
```

Run the bots and the eval scripts from the project root as modules
(`python -m agents.<name>`, `python -m evals.<name>`).

To switch vendor for a single run:

```bash
LLM_PROVIDER=openai python -m agents.agent_instrumented            # bash
$env:LLM_PROVIDER="openai"; python -m agents.agent_instrumented    # PowerShell
```

To view traces and scores in a dashboard, run `deepeval login` or set
`CONFIDENT_API_KEY` in `.env`. No code changes are needed.

---

## Configuration

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Vendor that runs the bots: `anthropic` (default) or `openai` |
| `JUDGE_PROVIDER` | Vendor that judges the evals. Defaults to `LLM_PROVIDER` |
| `ANTHROPIC_API_KEY` | Needed when either provider is `anthropic` |
| `OPENAI_API_KEY` | Needed when either provider is `openai`, and always by `test_agent_synthesized` |
| `VOYAGE_API_KEY` | Embeddings for the RAG agent |
| `CONFIDENT_API_KEY` | Optional: streams traces to Confident AI |

Model IDs live in `MODELS` in `config.py`.

---

## Project layout

```
config.py                      vendor and model selection (shared by agents and evals)
agents/shop_tools.py           demo order/refund data and the two shared @tool functions
agents/agent_plain.py          tool-calling agent, no eval code
agents/agent_instrumented.py   the same agent with DeepEval tracing and an MCP tool
agents/mcp_server.py           MCP server exposing get_shipping_options
agents/rag_agent.py            RAG agent: in-memory vector store + Voyage AI embeddings
agents/chatbot.py              multi-turn chatbot with a manual tool-call loop
policies.txt                   source document for synthetic goldens
evals/common.py                judge factory, async config, tracing wrapper
evals/test_*.py                the eval suites above
```

---

## Known limitations

- **DeepEval's Synthesizer embeds with OpenAI**, so `test_agent_synthesized`
  needs `OPENAI_API_KEY` even when Claude is the judge.
- **The Voyage AI free tier allows 3 requests per minute.** `rag_agent.py`
  embeds its documents on the first search and each query after that, so
  repeated runs within a minute are rate-limited.
- **Windows:** set `PYTHONIOENCODING=utf-8` when piping output, or emoji in
  model replies crash the cp1252 console encoder.

---

## Contributing

Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Enable the hook that checks them once per clone:

```bash
git config core.hooksPath .githooks
```
