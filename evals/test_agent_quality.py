from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from agent_instrumented import support_agent as _support_agent
from evals.common import ASYNC_CONFIG, get_judge, traced


support_agent = traced(_support_agent, name="support_agent")

judge = get_judge()

answer_relevancy = AnswerRelevancyMetric(threshold = 0.7, model = judge)
step_efficiency = StepEfficiencyMetric(threshold=0.5,model = judge)

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ], threshold= 0.7, model = judge
)


dataset = EvaluationDataset(goldens = [

    Golden(input = "Where is my order ORD-1042?"),
    Golden( input="What's the refund policy for electronics?" ),
    Golden( input="I want to return order ORD-2099, what should I do?" )

])

for golden in dataset.evals_iterator(metrics = [prompt_alignment,step_efficiency,answer_relevancy], async_config=ASYNC_CONFIG):
    support_agent(golden.input)


    #sample output -
