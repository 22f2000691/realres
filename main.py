import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from groq import Groq

app = FastAPI()

# Initialize Groq client using the environment variable GROQ_API_KEY
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Request Schema
class ProblemRequest(BaseModel):
    problem_id: str
    problem: str

# Response Schema with Strict Validation
class ProblemResponse(BaseModel):
    reasoning: str = Field(..., min_length=80)
    answer: int

    @field_validator("answer", mode="before")
    @classmethod
    def validate_strict_int(cls, v):
        # Reject booleans, floats (e.g., 945.0), and string integers (e.g., "945")
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("answer must be a strict JSON integer, not a string or float.")
        return v


SYSTEM_PROMPT = """
You are a precise mathematical word-problem solver.
Follow these steps:
1. Carefully analyze the problem and identify any distractor numbers or irrelevant facts.
2. Calculate the exact solution step-by-step.
3. Write out a clear, step-by-step reasoning narrative that MUST be at least 80 characters long.
4. Output your response ONLY as valid JSON containing exactly two keys: "reasoning" (string) and "answer" (integer).
"""

@app.post("/solve", response_model=ProblemResponse)
async def solve_problem(payload: ProblemRequest):
    try:
        # Call Groq Chat Completions API with JSON mode enabled
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Solve this problem:\n{payload.problem}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Zero temperature for deterministic results
        )

        # Parse the JSON string returned by Groq
        raw_content = chat_completion.choices[0].message.content
        parsed_data = json.loads(raw_content)

        # Pass through Pydantic to validate length & integer type rules
        validated_response = ProblemResponse(**parsed_data)
        return validated_response

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Groq did not return valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation or API Error: {str(e)}")
