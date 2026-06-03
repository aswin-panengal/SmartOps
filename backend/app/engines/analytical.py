import io
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0
)

# Global in-memory cache to retain DataFrames across distinct chat turns per session
active_dataframes = {}

def has_active_dataframe(session_id: str) -> bool:
    """Return whether this chat session already has a CSV loaded."""
    return session_id in active_dataframes

def clear_active_dataframe(session_id: str):
    """Clear any cached CSV for a session."""
    active_dataframes.pop(session_id, None)

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

def run_analytical_engine(file_bytes: bytes, question: str, session_id: str = "default") -> dict:
    """
    Agentic execution pipeline with Stateful Memory Caching and Smart Formatting.
    """
    try:
        # STATEFUL MEMORY ROUTING LOGIC
        if file_bytes:
            # New file upload: parse and commit to active session memory
            df = pd.read_csv(io.BytesIO(file_bytes))
            active_dataframes[session_id] = df
        elif session_id in active_dataframes:
            # Subsequent turn: pick up the existing DataFrame from RAM
            df = active_dataframes[session_id]
        else:
            # Fallback if server restarted or session dropped out of memory
            return {
                "status": "error",
                "question": question,
                "error": "No active dataset found in memory. Your session may have expired. Please re-upload your CSV file."
            }

        context = extract_dataframe_context(df)

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

                # SMART FORMATTING LOGIC
                if isinstance(raw_result, (pd.DataFrame, pd.Series)):
                    # Wrap in Markdown block for structural monospace alignment in chat UI
                    final_answer = f"```text\n{raw_result.to_string()}\n```"
                elif isinstance(raw_result, (list, np.ndarray, pd.Index)):
                    # Extract raw structural metrics to comma-delimited bold string profiles
                    clean_list = ", ".join(map(str, raw_result))
                    final_answer = f"**{clean_list}**"
                else:
                    # Strings, metrics, integers, and floats pass directly
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
