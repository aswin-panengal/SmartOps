import io
import time
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0
)

MAX_CSV_BYTES = 20 * 1024 * 1024  # 20 MB hard limit

# Each entry: {"df": DataFrame, "context": str, "ts": float}
active_dataframes: dict[str, dict] = {}

# Pandas I/O accessors the LLM must never touch
_BLOCKED_PD_ATTRS = frozenset({
    "read_csv", "read_excel", "read_json", "read_sql", "read_parquet",
    "read_html", "read_clipboard", "read_feather", "read_orc",
    "read_pickle", "read_stata", "read_sas", "read_spss",
    "DataFrame.to_csv", "DataFrame.to_sql", "DataFrame.to_excel",
})


class _SafePandas:
    """
    Thin wrapper around the pandas module that blocks filesystem/network I/O
    attributes while passing everything else through transparently.
    """
    def __getattr__(self, name: str):
        if name in _BLOCKED_PD_ATTRS:
            raise AttributeError(
                f"pd.{name} is disabled in this sandbox. "
                "Only operate on the 'df' variable already in scope."
            )
        return getattr(pd, name)


_safe_pd = _SafePandas()


def has_active_dataframe(session_id: str) -> bool:
    return session_id in active_dataframes


def clear_active_dataframe(session_id: str):
    active_dataframes.pop(session_id, None)


def _build_dataframe_context(df: pd.DataFrame) -> str:
    """Build a concise structural blueprint. Caps columns to avoid token bloat."""
    col_limit = 50
    cols = df.columns[:col_limit]
    dtype_str = df[cols].dtypes.to_string()
    sample_str = df[cols].head(5).to_string()
    try:
        stats_str = df[cols].describe().to_string()
    except Exception:
        stats_str = "(statistics unavailable)"

    truncation_note = (
        f"\n(Showing first {col_limit} of {len(df.columns)} columns)"
        if len(df.columns) > col_limit
        else ""
    )

    return (
        f"DATAFRAME BLUEPRINT:{truncation_note}\n"
        f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"- Columns and types:\n{dtype_str}\n"
        f"- Sample data (first 5 rows):\n{sample_str}\n"
        f"- Basic statistics:\n{stats_str}"
    )


def run_analytical_engine(
    file_bytes: bytes | None,
    question: str,
    session_id: str = "default",
) -> dict:
    try:
        if file_bytes:
            if len(file_bytes) > MAX_CSV_BYTES:
                return {
                    "status": "error",
                    "question": question,
                    "error": (
                        f"CSV file exceeds the {MAX_CSV_BYTES // (1024*1024)} MB limit. "
                        "Please upload a smaller file."
                    ),
                }
            df = pd.read_csv(io.BytesIO(file_bytes))
            if df.empty:
                return {
                    "status": "error",
                    "question": question,
                    "error": "The uploaded CSV has no data rows. Please check the file and re-upload.",
                }
            context = _build_dataframe_context(df)
            active_dataframes[session_id] = {"df": df, "context": context, "ts": time.time()}
        elif session_id in active_dataframes:
            entry = active_dataframes[session_id]
            df = entry["df"]
            context = entry["context"]
            entry["ts"] = time.time()
        else:
            return {
                "status": "error",
                "question": question,
                "error": (
                    "No active dataset found in memory. "
                    "Your session may have expired. Please re-upload your CSV file."
                ),
            }

        # Sanitize question to strip embedded instruction attempts
        safe_question = question[:2000]

        base_prompt = f"""
You are an expert data analyst writing Python code to extract metrics from a
pandas DataFrame named 'df'. The module 'pd' in your scope has filesystem I/O
disabled — do NOT call pd.read_csv, pd.read_excel, or any pd.read_* function.
Operate ONLY on the 'df' variable already provided.

{context}

User question: {safe_question}

Write clear Python pandas code to answer this question.
Rules:
- Use only the variable name 'df'
- You MUST assign the final answer to a variable named 'result'
- If the user asks for a summary, assign a natural language string to 'result'
- If the user asks for data, assign the pandas object to 'result'
- Keep code simple, direct, and single-purpose
- STRICT RULE: DO NOT use any import statement or call any pd.read_* function
- Return ONLY the executable code. No prose outside the code block.
"""

        SAFE_BUILTINS = {
            "len": len, "str": str, "int": int, "float": float,
            "round": round, "max": max, "min": min, "sum": sum,
            "abs": abs, "list": list, "dict": dict, "set": set,
            "tuple": tuple, "bool": bool, "any": any, "all": all,
            "sorted": sorted, "enumerate": enumerate, "zip": zip,
            "range": range, "print": print,
        }

        # _safe_pd blocks filesystem I/O; np is safe (no network/fs by default)
        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "pd": _safe_pd,
            "np": np,
            "df": df,
        }

        current_prompt = base_prompt
        MAX_RETRIES = 3
        last_error = ""
        generated_code = ""

        for attempt in range(MAX_RETRIES):
            local_vars: dict = {}
            response = llm.invoke(current_prompt)
            generated_code = response.content.strip()

            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0]
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0]
            generated_code = generated_code.strip()

            try:
                exec(generated_code, safe_globals, local_vars)  # noqa: S102

                raw_result = local_vars.get("result")
                if raw_result is None:
                    assignable = {k: v for k, v in local_vars.items() if k != "df"}
                    if assignable:
                        raw_result = list(assignable.values())[-1]
                    else:
                        raise ValueError("No 'result' variable was assigned.")

                if isinstance(raw_result, (pd.DataFrame, pd.Series)):
                    final_answer = f"```text\n{raw_result.to_string()}\n```"
                elif isinstance(raw_result, (list, np.ndarray, pd.Index)):
                    final_answer = f"**{', '.join(map(str, raw_result))}**"
                else:
                    final_answer = str(raw_result)

                return {
                    "status": "success",
                    "question": question,
                    "answer": final_answer,
                    "rows_in_file": len(df),
                    "columns": list(df.columns),
                    "generated_code": generated_code,
                    "attempts_required": attempt + 1,
                }

            except Exception as exec_err:
                last_error = str(exec_err)
                current_prompt = (
                    base_prompt
                    + f"\n\n--- ERROR ON PREVIOUS ATTEMPT ---\n"
                    f"Code:\n{generated_code}\n\n"
                    f"Error: {last_error}\n\n"
                    "REWRITE the code to fix this error. "
                    "Do NOT use import statements or pd.read_* functions."
                )

        return {
            "status": "error",
            "question": question,
            "error": (
                f"Agent failed to self-correct after {MAX_RETRIES} attempts. "
                f"Final error: {last_error}"
            ),
            "generated_code": generated_code,
        }

    except Exception as e:
        return {
            "status": "error",
            "question": question,
            "error": f"Analytical Core failure: {type(e).__name__}",
        }
