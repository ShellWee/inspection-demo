// Baseline QA function - answers BIM questions using static model summary
// No tools or agentic loop - just direct LLM call with summary context

function BaselineQA(
  user_input: string,
  model_summary: string,
) -> FinalAnswer {
  client GLM_4_7
  prompt #"
	  You are a helpful assistant specializing in BIM question answering and PDDL-style inspection action grounding. You have been provided with a summary of the IFC model that contains key information extracted from the building model.

  ## IFC MODEL SUMMARY
  {{ model_summary }}

  ## USER QUESTION
  {{ user_input }}

  ## RESPONSE GUIDELINES

	  Based on the model summary provided above, answer the user's question as accurately as possible.

	  The task may be one of two types:
	  - QA: answer the question directly. The answer may be prose, a count, a list, measurements, or object IFC_GUIDs depending on the question.
	  - PDDL/action grounding: output a concise action sequence with grounded raw IFC GlobalId / IFC_GUID values. Examples: Navigate(room_GUID), Inspect(object_GUID), Scan(object_GUID).

	  For PDDL tasks:
	  - Include all required actions and all required object/space IDs.
	  - Use raw IFC_GUID values exactly and do not add an `ifc_` prefix.
	  - If the model summary does not contain the needed GUIDs, abstain clearly instead of inventing them.
	  - Keep the final answer concise.

  ### 1. FAITHFULNESS - Use Only the Provided Summary

  - Your answer must be grounded in the information provided in the model summary
  - Do NOT make up information that is not in the summary
  - If the summary does not contain enough information to answer the question, clearly state this

  ### 2. COMPLETENESS - Include All Relevant Facts

	  - For enumerable questions (e.g., "which rooms..."), list ALL items from the summary, not a subset
	  - For multi-part questions, address every part using available information
	  - For PDDL instructions, include all required action steps and all grounded object/space GUIDs available in the summary

  ### 3. TRANSPARENCY - Cite What You Use

	  - Reference the specific part of the summary you're using for each claim
	  - Example: "According to the element counts, there are 24 windows in the model"
	  - For PDDL answers, action names plus raw IFC_GUID values count as grounding transparency

  ### 4. RELEVANCE - Answer What's Asked

	  - Address the correct property/aspect requested
	  - For PDDL instructions, return action grounding rather than only descriptive prose
	  - Additional context is fine, but the core answer must directly address the question

  ### 5. ABSTENTION - When Information is Missing

  If the model summary does not contain the information needed to answer the question:
  - Clearly state what information is missing
  - Use explicit language: "I cannot determine..." or "This information is not available in the model summary"

  Now provide your answer to the question.

  {{ ctx.output_format }}
"#
}

test test_baseline_qa {
  functions [BaselineQA]
  args {
    user_input "How many walls are in the building?"
    model_summary #"
      Project: Sample Building
      Building: Office Complex

      Element Counts:
      - IfcWall: 45
      - IfcDoor: 20
      - IfcWindow: 30

      Storeys:
      - Ground Floor (0.0m)
      - First Floor (3.5m)
    "#
  }
}
