from smolagents.agents import populate_template
import yaml
from agent_kb.agent_kb_utils import call_model
import ast
import json
import re
from typing import List
import ast
import json
import re
from typing import List


from typing import List
import re
import json
import ast


def build_action_level_planning_subseq_examples(entities):
    step_and_rationaleses = []
    for entity in entities:
        lines = []
        action_description = entity.get("macro_description")
        lines.append(f"Similar actions: {action_description}")
        for i, (step, rationale) in enumerate(
            zip(
                entity[step_field],
                entity[rationale_field],
            ),
            start=1,
        ):
            lines.append(f"{i}. {action_description}")
        step_and_rationales = "\n".join(lines)
        step_and_rationaleses.append(step_and_rationales)
    step_and_rationaleses_text = "\n".join(step_and_rationaleses)
    return step_and_rationaleses_text


def action_level_planning(task, curruent_plan, model, retrieval_method, top_k):
    prompt = load_prompts(
        path="/home/work/.default/huijeong/agentkb/Agent-KB-GAIA/examples/open_deep_research/planner_kb/action_planner_prompts.yaml"
    )
    generate_tag_prompt_template = prompt["generate_tag_prompt"]
    generate_tag_prompt = populate_template(
        generate_tag_prompt_template,
        variables={"task": task, "curruent_plan": curruent_plan, "top_k": top_k},
    )
    action_tags = call_model(model, generate_tag_prompt)
    action_tag_list = parse_str_to_list(action_tags)
    retrieval_results = []
    for action_tag in action_tag_list[:top_k]:
        retrieval_result = retrieval_method(action_tag, top_k=1, is_action=True)
        retrieval_results.append(retrieval_result)
    action_sequence_examples = build_action_level_planning_subseq_examples(
        retrieval_results
    )

    action_planning_with_examples_prompt_template = prompt[
        "action_planning_with_examples_prompt"
    ]
    action_planning_with_examples_prompt = populate_template(
        action_planning_with_examples_prompt_template,
        variables={
            "task": task,
            "curruent_plan": curruent_plan,
            "action_sequence_examples": action_sequence_examples,
        },
    )
    action_planning_result = call_model(model, action_planning_with_examples_prompt)
    return action_planning_result
