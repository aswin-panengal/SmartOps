import pandas as pd
import numpy as np
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0
)

def extract_dataframe_context(df: pd.DataFrame) -> str:
    """Extracts structural blueprint without sending the raw payload."""
    context = f"""
    DATAFRAME BLUEPRINT:
    - Shape: 
    {df.shape[0]} rows x {df.shape[1]} columns
    
    - Columns and types:
    {df.dtypes.to_string()}
    
    - Sample data (first 5 rows):
    {df.head(5).to_string()}
    
    - Basic statistics:
    {df.describe().to_string()}
    """
    return context
def run_analytical_engine(file_bytes: bytes, question: str) -> dict:
    """
    Agentic execution pipeline with Smart Formatting.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        context = extract_dataframe_context(df)

        # 1. TWEAKED PROMPT: Allow natural language assignments for conceptual questions
        base_prompt = f"""
        You are an expert data analyst writing code to extract metrics from a pandas DataFrame named 'df'.
        
        {context}
        
        User question: {question}
        
        Write clear Python pandas code to answer this question.
        Rules:
        - Use only the variable name 'df' 
        - You MUST assign the final answer to a variable named 'result' (e.g., result = df['sales'].sum())
        - If the user asks for a summary or explanation, assign a natural language string to 'result'.
        - If the user asks for data, assign the pandas object to 'result'.
        - Keep code simple, direct, and single-purpose.
        - STRICT RULE: DO NOT import any external libraries. 
        - Return ONLY the executable code blocks. No explanations outside the code.
        """

        current_prompt = base_prompt
        MAX_RETRIES = 3
        last_error = ""
        generated_code = ""

        SAFE_BUILTINS = {
            "len": len, "str": str, "int": int, "float": float,
            "round": round, "max": max, "min": min, "sum": sum,
            "abs": abs, "list": list, "dict": dict, "set": set,
            "tuple": tuple, "bool": bool, "any": any, "all": all
        }

        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "df": df
        }

        for attempt in range(MAX_RETRIES):
            local_vars = {}
            
            response = llm.invoke(current_prompt)
            generated_code = response.content.strip()

            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0]
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0]
            generated_code = generated_code.strip()

            try:
                exec(generated_code, safe_globals, local_vars)
                
                raw_result = local_vars.get("result")
                if raw_result is None:
                    assignable_vars = {k: v for k, v in local_vars.items() if k != "df"}
                    if assignable_vars:
                        raw_result = list(assignable_vars.values())[-1]
                    else:
                        raise ValueError("No scalar or structural 'result' variable was created.")

                # 2. NEW FORMATTING LOGIC: Make the output beautiful before sending it to the frontend
                if isinstance(raw_result, (pd.DataFrame, pd.Series)):
                    # Wrap in Markdown text block to enforce Monospace font alignment in the UI
                    final_answer = f"```text\n{raw_result.to_string()}\n```"
                elif isinstance(raw_result, (list, np.ndarray, pd.Index)):
                    # Clean up ugly Python lists/indices into nice comma-separated text
                    clean_list = ", ".join(map(str, raw_result))
                    final_answer = f"**{clean_list}**"
                else:
                    # Strings and numbers pass through normally
                    final_answer = str(raw_result)

                return {
                    "status": "success",
                    "question": question,
                    "answer": final_answer,
                    "rows_in_file": len(df),
                    "columns": list(df.columns),
                    "generated_code": generated_code,
                    "attempts_required": attempt + 1
                }

            except Exception as exec_err:
                last_error = str(exec_err)
                current_prompt = base_prompt + f"""
                
                --- ERROR ON PREVIOUS ATTEMPT ---
                You generated this code:
                {generated_code}
                
                It failed in the secure execution sandbox with this exact error:
                {last_error}
                
                REWRITE THE CODE to fix this specific error. 
                CRITICAL: If the error says '__import__ not found', it means you used an 'import' statement. 
                You are strictly forbidden from using 'import'.
                """

        return {
            "status": "error",
            "question": question,
            "error": f"Agent failed to self-correct after {MAX_RETRIES} attempts. Final Error: {last_error}",
            "generated_code": generated_code
        }

    except Exception as e:
        return {
            "status": "error",
            "question": question,
            "error": f"Analytical Core failure: {str(e)}"
        }