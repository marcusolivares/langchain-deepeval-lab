"""
Generate the goldens test_agent_synthesized.py scores, from policies.txt, and
save them to evals/datasets/synthesized_goldens.json.

Run it once, review the saved goldens, and commit them. Rerun it only to refresh
the dataset: every eval run then scores the same inputs.

Paid: the Synthesizer calls the judge model, and DeepEval embeds the document
with OpenAI (it has no Anthropic or Voyage embedder), so OPENAI_API_KEY must be
set even when the judge is Claude.

    python -m evals.synthesize_goldens
"""

from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.synthesizer import Synthesizer

from evals.common import SYNTHESIZED_GOLDENS, get_judge

POLICIES = Path(__file__).resolve().parent.parent / "policies.txt"


def main() -> None:
    goldens = Synthesizer(model=get_judge()).generate_goldens_from_docs(
        document_paths=[str(POLICIES)],
        include_expected_output=True,
        max_goldens_per_context=2,
    )
    path = EvaluationDataset(goldens=goldens).save_as(
        file_type="json",
        directory=str(SYNTHESIZED_GOLDENS.parent),
        file_name=SYNTHESIZED_GOLDENS.stem,
    )
    print(f"Saved {len(goldens)} goldens to {path}")


if __name__ == "__main__":
    main()
