"""Prompt templates for the benchmark agent."""

TASK_REASONING_PROMPT = """
You are an agent that will perform a task in a 3D environment.

Before acting, reason about the task:

Goal: {goal}

Think step by step:
1. What is the main objective? What object(s) and action(s) are involved?
2. What is a reasonable strategy? (e.g. find the object, approach it, pick it up, take it to the target, put it down)
3. What could go wrong or what should you watch for? (e.g. doors, obstacles, object IDs only when close)
4. In what order should you do things?

Reply with a short reasoning and plan. Do not use any tools. This is only to plan before starting.
"""

VISUAL_INTERPRETER_PROMPT = """
You are a visual perception module for an agent in a 3D environment.

Analyze the egocentric camera image and describe ONLY what is visible,
without using technical IDs.

Return exclusively a valid JSON:

{
  "visual_objects": [
    {
      "label": "Object name",
      "relative_position": "center-front | left-front | right-front | up | down",
      "distance": "near | medium | far",
      "salient": true
    }
  ],
  "notes": "scene description and position relative to scene elements"
}

Rules:
- DO NOT invent IDs.
- DO NOT assume hidden states.
- DO NOT explain outside the JSON.
- Return ONLY visual_objects and notes.
"""

ACTION_SELECTOR_PROMPT = ACTION_SELECTOR_PROMPT = """
You are controlling an agent in a 3D environment.

GOAL: {goal}
{initial_reasoning_section}

CURRENT STATE:
{world_state}

AVAILABLE ACTIONS (tools):
- Movement: move_ahead, move_back, move_left, move_right
- Rotation: rotate_left, rotate_right, look_up, look_down
- Interaction: pickup_object, put_object, open_object, close_object, toggle_on, toggle_off, slice_object

CRITICAL RULES:
1. ONLY use object IDs that appear in the current state above
2. Objects without IDs are NOT close enough - you must approach them first
3. Only interact with objects within 1.9 distance
4. Use ONE action per turn

DECISION PROCESS:
1. Analyze what you see in the current state
2. Explain your reasoning briefly:
   - What objects are visible and their properties (pickupable, openable, receptacle, etc.)
   - What action makes sense based on the goal
   - Why you chose this specific action
3. Call the appropriate tool

EXPLORATION STRATEGY (follow in order):
1. Not visible yet? → Rotate 360° (4 times rotate_left or rotate_right) to see all directions
2. Visible but target not visible? → move to explore new areas
3. Target visible but no ID? → move_ahead to get closer
4. Target visible with ID and close? → Interact using the appropriate tool

COMMON ISSUES:
- Object disappeared? → Rotate to find it again
- Action failed? → Check object state and try a different approach
- Path blocked? → Clear obstacles (e.g., close_object on doors) or move to another position

Now, explain your reasoning and execute ONE action using the appropriate tool.
"""
