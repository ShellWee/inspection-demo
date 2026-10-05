// BAML template for BIM inspection QA, object grounding, and PDDL evaluation.
// Evaluates system responses using Correct / Wrong / Abstained classification.

enum QuestionCategory {
  Category1 @alias("1")
  Category2 @alias("2")
  Category3 @alias("3")
  Category4 @alias("4")
}

function EvaluateResponse(
  question: string,
  category: QuestionCategory,
  ground_truth: string,
  system_response: string,
) -> AnswerEvaluationResult {
  client GLM_4_7

  prompt #"
    {{ _.role("system") }}
    You are evaluating a BIM inspection system. The dataset now contains both ordinary QA tasks and PDDL-style inspection grounding tasks.

    ## QUESTION CATEGORIES

    Category 1 (QA: Direct Retrieval + Aggregation):
    Questions ask for direct BIM element/entity/property information or simple calculations such as counts, rankings, filtering, or most-used components. Answers may be prose, lists, IDs, counts, measurements, or short structured text.

    Category 2 (PDDL-Explicit):
    Instructions directly specify the action target, object, room, or component. Example: Inspect Door {door ID}. The answer should be a concise PDDL/action grounding sequence.

    Category 3 (PDDL-Spatial):
    Instructions require spatial reasoning or spatial relationships. Example: Navigate to the room adjacent to room B. The answer should include the required action(s) and grounded IFC_GUID object/space IDs.

    Category 4 (PDDL-Functional):
    Instructions require functional/system reasoning. Examples: Perform IR scan in room A; inspect fire safety element in lab 2. Category 4 may also include spatial reasoning. The answer should include the required action(s) and grounded IFC_GUID object/space IDs.

    ## EXPECTED ANSWER TYPES

    QA answers:
    - Answer according to the question.
    - If the question asks for objects, include the object IFC_GUID values.

    PDDL answers:
    - A concise sequence of action + grounding object/space IDs.
    - Valid examples: Navigate(IFC_GUID), Nav(IFC_GUID), Inspect(IFC_GUID), Scan(IFC_GUID).
    - Multi-step answers are allowed, e.g. Navigate(room_GUID), Inspect(equipment_GUID).
    - Multiple objects for one action are allowed when the instruction requires all matching objects.
    - Use raw IFC GlobalId / IFC_GUID values exactly. Do not require an `ifc_` prefix.
    - Evidence text is helpful but not required if action names and GUIDs are unambiguous.

    ---

    ## EVALUATION WORKFLOW

    Step 1: Determine if the system ABSTAINED or PROVIDED AN ANSWER.

    ABSTAINED if response explicitly declines to answer:
    - "I cannot determine..."
    - "Insufficient information to answer..."
    - "This data is not available and I cannot provide an answer..."
    - "I am unable to answer this question..."

    NOT abstained:
    - Expressions of uncertainty while still providing an answer
    - Partial information with a concrete answer
    - A PDDL/action sequence with at least one grounded ID

    If ABSTAINED:
    - Set abstention = true
    - Set all other criteria to Na
    - Provide brief justification
    - Stop evaluation

    If PROVIDED AN ANSWER:
    - Set abstention = false
    - Evaluate the four criteria independently

    ---

    ## CRITERION 1: FAITHFULNESS

    Does the answer match the reference facts, action grounding, and IFC_GUIDs?

    QA / Category 1:
    - Direct properties, entity/object facts, lists, and object GUIDs must match the reference answer.
    - Discrete counts must match exactly.
    - Continuous measurements may vary within 2 percent unless the reference requires exact values.
    - Rankings such as "largest" or "most used" must identify the same item(s) as the reference.

    PDDL / Categories 2-4:
    - The action type must be correct for the instruction: navigation uses Nav/Navigate; inspection uses Inspect; IR/thermal/electrical scan tasks use Scan unless the reference indicates otherwise.
    - Required space/object IFC_GUIDs must match the reference answer. Equivalent action spelling such as Nav vs Navigate is acceptable.
    - The action sequence must preserve required dependencies, e.g. navigate to a room before inspecting/scanning objects in or near it.
    - For spatial tasks, grounded IDs must satisfy the spatial relation in the instruction.
    - For functional tasks, grounded IDs must satisfy the requested functional/system role. If functional and spatial relations both apply, both must be satisfied.
    - Do not mark wrong only because wording differs from the reference. Mark wrong if the GUIDs, required action types, or target set are incorrect.

    Result:
    - Yes: The answer is grounded in the reference/model facts and uses the correct IDs/actions.
    - No: Any required GUID/action/fact is wrong, hallucinated, or contradicts the reference.
    - Na: Only if abstention = true.

    ---

    ## CRITERION 2: COMPLETENESS

    Does the answer include every required part?

    QA:
    - Include all requested facts, counts, object IDs, or list items.
    - For multi-part questions, answer every part.

    PDDL:
    - Include all required action steps, all required Navigate targets, and all required Inspect/Scan object IDs.
    - Include every object when the instruction asks for all matching objects.
    - Do not omit a needed navigation or task action.
    - Extra irrelevant actions or extra incorrect GUIDs fail completeness.

    Result:
    - Yes: All required facts/actions/objects are present and no conflicting extras are added.
    - No: Missing required facts/actions/objects, or includes conflicting extras.
    - Na: Only if abstention = true, or for genuinely open-ended QA with no objective completeness standard.

    ---

    ## CRITERION 3: TRANSPARENCY

    Is the answer sufficiently grounded and auditable while staying concise?

    QA:
    - Source or method should be stated when the answer includes calculations, derived spatial facts, or non-obvious object selection.
    - Direct lookup answers may be concise if the returned value or IFC_GUID is unambiguous.

    PDDL:
    - Action names plus raw IFC_GUID values count as grounding transparency.
    - Evidence chains such as Space > object, geometry/proximity, or function/system > object are helpful but not required.
    - Fail transparency if the response gives only prose like "inspect the fire equipment" without the actual GUIDs required for grounding.

    Result:
    - Yes: The response exposes enough values, IDs, actions, or method/evidence to audit the answer.
    - No: The response is too vague to verify, lacks required GUIDs, or hides the basis of a derived answer.
    - Na: Only if abstention = true.

    ---

    ## CRITERION 4: RELEVANCE

    Does the answer directly address the requested task type and scope?

    - QA questions should receive a direct QA answer, not an unrelated action plan.
    - PDDL instructions should receive a PDDL/action grounding answer, not only descriptive prose.
    - The answer must use the requested scope: correct room, level, object type, system, function, spatial relation, and action intent.
    - Additional context is acceptable only if the core answer remains concise and directly relevant.

    Result:
    - Yes: Directly addresses the requested facts/actions/objects.
    - No: Answers the wrong task, wrong scope, wrong object type, or wrong action intent.
    - Na: Only if abstention = true.

    ---

    ## CRITICAL GUIDELINES

    1. Evaluate criteria independently.
    2. For PDDL tasks, exact IFC_GUID grounding is central. Do not give credit for a correct object name with the wrong GUID.
    3. For Category 4, functional reasoning can include spatial reasoning. Do not penalize the presence of spatial evidence when the functional target is correct.
    4. Keep justification to 2-3 concise sentences and focus on any No verdicts or edge cases.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    QUESTION CATEGORY: {{ category }}
    QUESTION: {{ question }}

    REFERENCE ANSWER: {{ ground_truth }}

    Note: This reference answer is one possible response from a human expert labeller, but action grounding and IFC_GUIDs are expected to match unless the system provides an equivalent valid grounding.

    SYSTEM RESPONSE: {{ system_response }}

    Evaluate the system response according to the framework above.
  "#
}

// Duplicate function using Gemini 3 Pro for second LLM judge
function EvaluateResponseGemini(
  question: string,
  category: QuestionCategory,
  ground_truth: string,
  system_response: string,
) -> AnswerEvaluationResult {
  client Gemini_3_Pro

  prompt #"
    You are evaluating a BIM inspection system. The dataset contains both ordinary QA tasks and PDDL-style inspection grounding tasks.

    QUESTION CATEGORY: {{ category }}
    QUESTION: {{ question }}

    REFERENCE ANSWER: {{ ground_truth }}

    SYSTEM RESPONSE: {{ system_response }}

    ## QUESTION CATEGORIES

    Category 1 (QA: Direct Retrieval + Aggregation): Direct BIM element/entity/property information or simple calculations such as counts, rankings, filtering, or most-used components.
    Category 2 (PDDL-Explicit): Instructions directly specify the action target, object, room, or component.
    Category 3 (PDDL-Spatial): Instructions require spatial reasoning or spatial relationships.
    Category 4 (PDDL-Functional): Instructions require functional/system reasoning. Category 4 may also include spatial reasoning.

    ## EXPECTED ANSWER TYPES

    QA answers may be prose, lists, counts, measurements, IDs, or short structured text. If the question asks for objects, include object IFC_GUID values.

    PDDL answers should be concise sequences of action + grounding IDs, such as Navigate(IFC_GUID), Inspect(IFC_GUID), or Scan(IFC_GUID). Multi-step answers and multiple object IDs are allowed. Raw IFC GlobalId / IFC_GUID values are required; no `ifc_` prefix is needed.

    ## EVALUATION WORKFLOW

    If the system explicitly declines to answer, set abstention = true and all other criteria to Na. Otherwise set abstention = false and evaluate the four criteria independently.

    ## CRITERIA

    Faithfulness:
    - QA facts, counts, measurements, object IDs, and rankings must match the reference. Discrete counts must be exact; continuous measurements may vary within 2 percent unless exact values are required.
    - PDDL action type, action order, and required IFC_GUIDs must match the reference or be an equivalent valid grounding. Nav/Navigate spelling variants are acceptable.
    - Spatial tasks must satisfy the spatial relation. Functional tasks must satisfy the requested system/function role, and may also require spatial reasoning.

    Completeness:
    - QA answers must include all requested facts or objects.
    - PDDL answers must include all required action steps, Navigate targets, Inspect/Scan object IDs, and all matching objects requested by the instruction. Missing required actions/objects or extra incorrect GUIDs fail completeness.

    Transparency:
    - QA derived/calculated answers should state a concise source or method; direct lookup answers may be concise if unambiguous.
    - For PDDL, action names plus raw IFC_GUIDs count as grounding transparency. Fail if required GUIDs are absent.

    Relevance:
    - QA questions should receive a direct QA answer.
    - PDDL instructions should receive action grounding, not only descriptive prose.
    - The answer must match the requested room, level, object type, system, function, spatial relation, and action intent.

    Critical guidelines:
    - Exact IFC_GUID grounding is central for PDDL tasks.
    - Do not penalize concise PDDL output when action names and GUIDs are present.
    - Keep justification to 2-3 concise sentences.

    {{ ctx.output_format }}
  "#
}

test qa_count_test {
  functions [EvaluateResponse]
  args {
    question "How many dampers are in the synthetic model?"
    category Category1
    ground_truth "There are 12 dampers in the synthetic model."
    system_response "12 dampers."
  }
}

test pddl_explicit_test {
  functions [EvaluateResponse]
  args {
    question "Inspect component 1BBBBBBBBBBBBBBBBBBBBB."
    category Category2
    ground_truth "Inspect(1BBBBBBBBBBBBBBBBBBBBB)"
    system_response "Inspect(1BBBBBBBBBBBBBBBBBBBBB)"
  }
}

test pddl_spatial_test {
  functions [EvaluateResponse]
  args {
    question "Navigate to the room adjacent to synthetic room A101."
    category Category3
    ground_truth "Navigate(0AAAAAAAAAAAAAAAAAAAAA)"
    system_response "Nav(0AAAAAAAAAAAAAAAAAAAAA)"
  }
}

test pddl_functional_test {
  functions [EvaluateResponse]
  args {
    question "Perform a thermal scan of switchboards in synthetic room C303."
    category Category4
    ground_truth "Navigate(0AAAAAAAAAAAAAAAAAAAAA), Scan(2CCCCCCCCCCCCCCCCCCCCC), Scan(3DDDDDDDDDDDDDDDDDDDDD)"
    system_response "Nav(0AAAAAAAAAAAAAAAAAAAAA), Scan(2CCCCCCCCCCCCCCCCCCCCC), Scan(3DDDDDDDDDDDDDDDDDDDDD)"
  }
}
