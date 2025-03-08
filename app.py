import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate

def process_csv(input_file, output_file, user_prompt, allow_ml=False):
    print(f"Step 1: Loading CSV file from {input_file}...")
    # Load CSV into DataFrame, explicitly setting first row as headers
    df = pd.read_csv(input_file, header=0)
    print(f"✓ Loaded DataFrame with shape: {df.shape}")
    print(f"✓ Columns: {', '.join(df.columns)}")
    
    # Display sample data for debugging
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Data types analysis
    print("\nData types analysis:")
    print(df.dtypes)
    
    print("\nStep 2: Initializing Ollama LLM...")
    # Initialize Ollama with local Llama 3 model
    llm = OllamaLLM(base_url='http://localhost:11434', model="llama3")
    print("✓ LLM initialized successfully")
    
    # Generate new column name
    new_col = "derived_column"  # Simple default name
    
    # Create system prompt template with sample data
    ml_guidance = """
    You can use machine learning techniques if appropriate. If using ML:
    - Import necessary libraries (sklearn, etc.)
    - Keep models simple (linear regression, decision trees, etc.)
    - Fit the model on the data
    - Use the model to generate predictions for the new column
    """ if allow_ml else """
    Do NOT use any machine learning libraries or complex statistical methods.
    Stick to basic pandas operations, math functions, and string manipulations.
    """
    
    direct_prompt = f"""You are a data analysis expert. Given a DataFrame with shape ({len(df)},{len(df.columns)}),
    create a new column '{new_col}' (column number {len(df.columns) + 1}) using this logic: {user_prompt}.
    
    Here are the column names and their data types:
    {pd.DataFrame({'Column': df.columns, 'Type': df.dtypes}).to_string()}
    
    Here are some sample rows from the DataFrame to help you understand the data:
    {df.head(3).to_string()}
    
    {ml_guidance}
    
    Return ONLY the Python code to calculate this new column using pandas operations.
    The code should:
    - Use existing columns from the DataFrame (df)
    - Be efficient and vectorized when possible
    - Assign result to df['{new_col}']
    - Include comments explaining your approach
    
    IMPORTANT: Your response should ONLY contain the Python code, nothing else.
    
    Example format of your response:
    ```python
    # Calculate new column based on [your explanation]
    df['{new_col}'] = df['existing_column'] * 2  # Your actual calculation here
    ```"""
    
    print(f"\nStep 3: Preparing to create new column '{new_col}'")
    print(f"✓ User prompt: '{user_prompt}'")
    print(f"✓ ML techniques {'allowed' if allow_ml else 'not allowed'}")
    
    print("\nStep 4: Creating pandas DataFrame agent...")
    # Create agent with handle_parsing_errors=True
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        handle_parsing_errors=True,  # Add this to handle parsing errors
        allow_dangerous_code=True
    )
    print("✓ Agent created successfully")
    
    print("\nStep 5: Generating code with LLM...")
    # Use direct LLM call instead of agent for code generation
    response = llm.invoke(direct_prompt)
    print("✓ Received response from LLM")
    
    # Extract and execute code
    print("\nExtracting code from response...")
    generated_code = response
    
    if '```' in generated_code:
        # Try to extract code between code blocks
        code_blocks = generated_code.split('```')
        for i in range(1, len(code_blocks), 2):
            block = code_blocks[i].strip()
            if block.startswith('python'):
                generated_code = block[6:].strip()
                break
            elif not any(block.startswith(lang) for lang in ['javascript', 'java', 'cpp', 'html']):
                # If no language specified, assume it's Python
                generated_code = block
                break
    else:
        # If no code blocks, use the entire response
        generated_code = generated_code.strip()
    
    print("\nStep 6: Generated code:")
    print(f"```python\n{generated_code}\n```")
    
    print("\nStep 7: Executing generated code...")
    
    # Prepare execution environment with necessary imports
    local_vars = {
        'df': df.copy(),
        'pd': pd,
        'np': None,
        'sklearn': None
    }
    
    global_vars = {}
    
    # Conditionally import libraries if needed
    if allow_ml and 'sklearn' in generated_code:
        print("Importing scikit-learn for ML operations...")
        try:
            import numpy as np
            import sklearn
            local_vars['np'] = np
            local_vars['sklearn'] = sklearn
        except ImportError:
            print("⚠ Warning: scikit-learn not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "scikit-learn", "numpy"])
            import numpy as np
            import sklearn
            local_vars['np'] = np
            local_vars['sklearn'] = sklearn
    
    if 'numpy' in generated_code or 'np.' in generated_code:
        print("Importing numpy for numerical operations...")
        try:
            import numpy as np
            local_vars['np'] = np
        except ImportError:
            print("⚠ Warning: numpy not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "numpy"])
            import numpy as np
            local_vars['np'] = np
    
    try:
        # Execute the generated code
        exec(generated_code, global_vars, local_vars)
        df = local_vars['df']
        
        # Verify the new column was created
        if new_col in df.columns:
            print(f"✓ Successfully created new column '{new_col}'")
            print(f"✓ Sample values: {df[new_col].head(3).tolist()}")
            
            # Show the updated DataFrame with the new column
            print("\nUpdated DataFrame (first 3 rows with new column):")
            print(df.head(3).to_string())
            
            # Basic statistics of the new column
            print("\nStatistics for the new column:")
            if pd.api.types.is_numeric_dtype(df[new_col]):
                print(df[new_col].describe())
            else:
                print(f"Value counts:\n{df[new_col].value_counts().head(5)}")
        else:
            print(f"⚠ Warning: Column '{new_col}' was not created by the code")
            print(f"⚠ Available columns: {', '.join(df.columns)}")
            
            # Try to find if a different column was created instead
            new_columns = set(df.columns) - set(local_vars['df'].columns)
            if new_columns:
                print(f"✓ However, these new columns were created: {', '.join(new_columns)}")
                new_col = list(new_columns)[0]  # Use the first new column
                df = local_vars['df']
                print(f"✓ Using '{new_col}' as the derived column")
        
        print(f"\nStep 8: Saving results to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"✓ Successfully saved DataFrame with {len(df.columns)} columns to {output_file}")
    except Exception as e:
        print(f"❌ Error executing generated code: {str(e)}")
        print("⚠ The output file was not created")
        
        # Try to provide more detailed error information
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()

# Usage
if __name__ == "__main__":
    print("=== CSV Analyzer with LLM ===")
    process_csv(
        input_file="input.csv",
        output_file="output.csv",
        user_prompt="The derived column should be the best fit for the data in the dataframe",
        allow_ml=True  # Set to False if you don't want ML techniques
    )
    print("=== Process completed ===")