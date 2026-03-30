import pandas as pd
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
    """
    Instead of sending the full CSV to Gemini,
    we extract just the shape, columns, types, and 5 rows.
    This is the core cost-saving intelligence of the engine.
    """
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
    Main function:
    1. Load CSV into pandas
    2. Extract lightweight context
    3. Ask Gemini to generate pandas code
    4. Execute that code safely
    5. Return the result
    """
    try:
        # Step 1: Load the CSV
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Step 2: Extract context (not the full data)
        context = extract_dataframe_context(df)

        # Step 3: Build the prompt
        prompt = f"""
        You are a data analyst. You have access to a pandas DataFrame called 'df'.
        
        {context}
        
        User question: {question}
        
        Write Python pandas code to answer this question.
        Rules:
        - Use only the variable name 'df' 
        - Store your final answer in a variable called 'result'
        - Keep code simple and direct
        - Do not import anything
        - Return ONLY the code, no explanation, no markdown, no backticks
        """

        # Step 4: Gemini generates the code
        response = llm.invoke(prompt)
        generated_code = response.content.strip()

        # Clean up if Gemini adds markdown backticks anyway
        if generated_code.startswith("```"):
            lines = generated_code.split("\n")
            generated_code = "\n".join(lines[1:-1])

        # Step 5: Execute the generated code safely
        local_vars = {"df": df}
        exec(generated_code, {"pd": pd}, local_vars)
        result = local_vars.get("result", "No result variable found")

        return {
            "status": "success",
            "question": question,
            "answer": str(result),
            "rows_in_file": len(df),
            "columns": list(df.columns),
            "generated_code": generated_code
        }

    except Exception as e:
        return {
            "status": "error",
            "question": question,
            "error": str(e)
        }