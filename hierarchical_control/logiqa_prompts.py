from __future__ import annotations

from .logiqa_pilot import (
    build_critic_messages as build_minimal_v1_critic_messages,
)
from .logiqa_pilot import (
    build_refiner_messages as build_minimal_v1_refiner_messages,
)


MINIMAL_V1 = "minimal_v1"
STRUCTURED_V2 = "structured_v2"
PROMPT_VERSIONS = (MINIMAL_V1, STRUCTURED_V2)


STRUCTURED_V2_CRITIC_SYSTEM_PROMPT = """You are the Critic in a Solver–Critic–Refiner pipeline for
multiple-choice logical reasoning.

Your task is to audit the Solver's response. Do not assume that the
response is correct or incorrect. Use only the problem statement,
answer choices, and Solver response. Do not use outside knowledge.

Evaluate the selected answer rather than writing style. A response is
not wrong merely because its explanation is brief, incomplete, or uses
a different valid reasoning path.

Audit procedure:
1. Identify the exact question polarity, such as MUST, COULD, CANNOT,
   or EXCEPT.
2. Extract the constraints that are decisive for distinguishing the
   answer choices.
3. Check whether the Solver's selected option and key reasoning are
   consistent with those constraints.
4. If an error is suspected, identify the exact violated constraint or
   invalid inference.
5. Test the proposed alternative against all relevant constraints.
   A merely plausible alternative is not sufficient evidence.

Return KEEP when no decisive logical error can be demonstrated.
Return REVISE only when both a concrete error and a verified better
option can be provided. Do not rewrite the complete solution.

Output exactly these fields, with the verdict after the analysis:

QUESTION_POLARITY: <short description>
CONSTRAINT_AUDIT: <concise evidence-based analysis>
DECISIVE_ERROR: <specific error or NONE>
ALTERNATIVE_VERIFICATION: <verification or NONE>
VERDICT: <KEEP or REVISE>
PROPOSED_ANSWER: <A, B, C, D, or NONE>
"""


STRUCTURED_V2_CRITIC_USER_TEMPLATE = """Evaluate the following Solver response.

<problem_and_choices>
{problem_and_choices}
</problem_and_choices>

<solver_response>
{solver_response}
</solver_response>
"""


STRUCTURED_V2_REFINER_SYSTEM_PROMPT = """You are the Refiner in a Solver–Critic–Refiner pipeline for
multiple-choice logical reasoning.

You receive the original problem, the Solver response, and a Critic
review. The Critic review is fallible evidence, not an authoritative
answer.

Validation procedure:
1. Preserve the question polarity and the original A–D mapping.
2. If the Critic verdict is KEEP, preserve the Solver's selected answer.
3. If the verdict is REVISE, verify that DECISIVE_ERROR really
   contradicts the problem and that PROPOSED_ANSWER satisfies all
   relevant constraints.
4. Apply the revision only when both checks succeed. Otherwise preserve
   the Solver's answer.
5. Do not introduce assumptions that are absent from the problem, and
   do not change an answer merely because another option seems
   plausible.
6. If the Critic output is malformed or lacks concrete evidence,
   preserve the Solver's answer.

Give only a concise justification. End with the answer marker on its
own final line.

Output format:

CRITIQUE_VALIDATION: <VALID, INVALID, or NOT_APPLICABLE>
REFINEMENT_DECISION: <KEEP_ORIGINAL or APPLY_REVISION>
JUSTIFICATION: <one to three concise sentences>
FINAL_ANSWER: <A, B, C, or D>
"""


STRUCTURED_V2_REFINER_USER_TEMPLATE = """Refine the response only if the Critic's proposed correction is valid.

<problem_and_choices>
{problem_and_choices}
</problem_and_choices>

<solver_response>
{solver_response}
</solver_response>

<critic_review>
{critic_review}
</critic_review>
"""


def _validate_prompt_version(prompt_version: str) -> None:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(
            f"Unknown LogiQA prompt version {prompt_version!r}; "
            f"expected one of {PROMPT_VERSIONS}"
        )


def build_versioned_critic_messages(
    problem_and_choices: str,
    solver_response: str,
    prompt_version: str = MINIMAL_V1,
) -> list[dict[str, str]]:
    _validate_prompt_version(prompt_version)
    if prompt_version == MINIMAL_V1:
        return build_minimal_v1_critic_messages(problem_and_choices, solver_response)
    return [
        {"role": "system", "content": STRUCTURED_V2_CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": STRUCTURED_V2_CRITIC_USER_TEMPLATE.format(
                problem_and_choices=problem_and_choices,
                solver_response=solver_response,
            ),
        },
    ]


def build_versioned_refiner_messages(
    problem_and_choices: str,
    solver_response: str,
    critic_review: str,
    prompt_version: str = MINIMAL_V1,
) -> list[dict[str, str]]:
    _validate_prompt_version(prompt_version)
    if prompt_version == MINIMAL_V1:
        return build_minimal_v1_refiner_messages(
            problem_and_choices,
            solver_response,
            critic_review,
        )
    return [
        {"role": "system", "content": STRUCTURED_V2_REFINER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": STRUCTURED_V2_REFINER_USER_TEMPLATE.format(
                problem_and_choices=problem_and_choices,
                solver_response=solver_response,
                critic_review=critic_review,
            ),
        },
    ]
