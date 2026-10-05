// Category validation for BIM inspection QA, object grounding, and PDDL tasks.
// Validates that questions are categorized according to the current 4-category taxonomy.

function ValidateQuestionCategory(
  question: string,
  answer: string,
  current_category: string
) -> CategoryValidationResult {
  client GLM_4_7

  prompt #"
    You are a BIM dataset quality specialist validating categories for a dataset containing both ordinary QA and PDDL-style inspection grounding tasks.

    QUESTION: {{ question }}

    ANSWER: {{ answer }}

    CURRENT CATEGORY: {{ current_category }}

    ---

    TAXONOMY DEFINITIONS:

    Category 1: QA (Direct Retrieval + Aggregation)
    - The task is a normal question-answering task.
    - It asks for direct BIM element/entity/property information, object identity, object IFC_GUIDs, or simple calculations.
    - Simple calculations include counting, filtering, ranking, grouping, "most used", "largest", or similar aggregation.
    - The expected answer is not primarily a PDDL action sequence.

    Examples:
    - "What types of spaces are in the synthetic annex?"
    - "How many dampers are in the model?"
    - "Which synthetic component type is used most often?"
    - "What is the identifier of the mock testing room?"

    Category 2: PDDL-Explicit
    - The task is an instruction/action grounding task.
    - The instruction directly identifies the target object, room, component, or ID without requiring spatial or functional inference.
    - The expected answer is an action sequence with grounded IFC_GUIDs.

    Examples:
    - "Inspect component 1BBBBBBBBBBBBBBBBBBBBB."
    - "Navigate to synthetic room A101."
    - "Inspect the display in mock office B202."
    - "Scan switchboard 2CCCCCCCCCCCCCCCCCCCCC."

    Category 3: PDDL-Spatial
    - The task is a PDDL/action grounding instruction that requires spatial reasoning.
    - Spatial reasoning includes adjacency, nearest/farthest, largest/smallest space, objects in/near a room, proximity inferred from geometry, or missing direct contained-in edges.
    - The expected answer is an action sequence with grounded IFC_GUIDs.

    Examples:
    - "Navigate to the room adjacent to synthetic room A101."
    - "Inspect the detector closest to the mock testing room."
    - "Inspect the suppression component in the largest synthetic hall."
    - "Be by the door in mock room B202."

    Category 4: PDDL-Functional
    - The task is a PDDL/action grounding instruction that requires functional, system, or asset-role reasoning.
    - Functional reasoning includes safety equipment, fire protection, HVAC/plumbing/electrical systems, IR scan targets, panels, drains, or other inspection roles.
    - Category 4 may also include spatial reasoning. If an instruction requires both function and spatial inference, choose Category 4.
    - The expected answer is an action sequence with grounded IFC_GUIDs.

    Examples:
    - "Perform a thermal scan of switchboards in synthetic room C303."
    - "Inspect smoke-control components in all mock plant spaces."
    - "Inspect condensate drains in the room whose switchboard needs a thermal scan."
    - "Inspect the emergency component nearest to the mock testing room."

    ---

    CLASSIFICATION RULES:

    1. First decide whether the task is QA or PDDL/action grounding.
       - If the expected answer is a factual answer, count, list, or normal explanation, choose Category 1.
       - If the expected answer is action + grounded object/space IFC_GUIDs, choose Category 2, 3, or 4.

    2. For PDDL tasks, choose the most specific reasoning type:
       - Directly specified action target with no inference: Category 2.
       - Spatial relation or geometry/proximity inference: Category 3.
       - Functional/system/asset-role inference: Category 4.

    3. If both spatial and functional reasoning are present, choose Category 4.

    4. Object grounding alone does not make a task PDDL. If the question asks "which object" and expects only object IFC_GUIDs as an answer, it is Category 1 unless an action sequence is required.

    5. Use only the categories 1, 2, 3, or 4.

    VALIDATION TASK:

    1. Analyze the question and answer to determine the correct category.
    2. Compare with the current category.
    3. If they match: set `updated` to false.
    4. If they differ: set `updated` to true and provide the correct category.
    5. Always provide clear reasoning in the `thought` field.

    {{ ctx.output_format }}
  "#
}

test qa_aggregation {
  functions [ValidateQuestionCategory]
  args {
    question "How many dampers are in the synthetic model?"
    answer "There are 12 dampers in the synthetic model."
    current_category "1"
  }
}

test pddl_explicit {
  functions [ValidateQuestionCategory]
  args {
    question "Inspect component 1BBBBBBBBBBBBBBBBBBBBB."
    answer "Inspect(1BBBBBBBBBBBBBBBBBBBBB)"
    current_category "2"
  }
}

test pddl_spatial {
  functions [ValidateQuestionCategory]
  args {
    question "Navigate to the room adjacent to synthetic room A101."
    answer "Navigate(0AAAAAAAAAAAAAAAAAAAAA)"
    current_category "3"
  }
}

test pddl_functional {
  functions [ValidateQuestionCategory]
  args {
    question "Perform a thermal scan of switchboards in synthetic room C303."
    answer "Navigate(0AAAAAAAAAAAAAAAAAAAAA), Scan(2CCCCCCCCCCCCCCCCCCCCC)"
    current_category "4"
  }
}
