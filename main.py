import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate

def process_csv(input_file, output_file, user_prompt):
    print(f"Step 1: Loading CSV file from {input_file}...")
    # Load CSV into DataFrame, explicitly setting first row as headers
    df = pd.read_csv(input_file, header=0)
    print(f"✓ Loaded DataFrame with shape: {df.shape}")
    print(f"✓ Columns: {', '.join(df.columns)}")
    
    # Display sample data for debugging
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    print("\nStep 2: Initializing Ollama LLM...")
    # Initialize Ollama with local Llama 3 model
    llm = OllamaLLM(base_url='http://localhost:11434', model="llama3")
    print("✓ LLM initialized successfully")
    
    # Generate new column name
    new_col = df.columns[-1] + '_derived'  # Or use LLM to generate name
    m_plus_1 = len(df.columns) + 1
    
    # Format sample data for the prompt
    sample_data_str = df.head(3).to_string()
    column_names_str = ", ".join([f"'{col}'" for col in df.columns])
    
    print(f"\nStep 3: Preparing to create new column '{new_col}'")
    print(f"✓ User prompt: '{user_prompt}'")
    
    # Create a direct prompt instead of using the agent for code generation
    direct_prompt = f"""You are a data analysis expert. Given a DataFrame with shape ({len(df)},{len(df.columns)}),
    create a new column '{new_col}' (column number {m_plus_1}) using this logic: {user_prompt}.
    
    Here are the column names: {column_names_str}
    
    Here are some sample rows from the DataFrame to help you understand the data:
    {sample_data_str}
    
    Return ONLY the Python code to calculate this new column using pandas operations.
    The code should:
    - Use existing columns from the DataFrame (df)
    - Be efficient and vectorized
    - Not use any external libraries
    - Assign result to df['{new_col}']
    
    IMPORTANT: Your response should ONLY contain the Python code, nothing else.
    
    Example format of your response:
    ```python
    # Calculate new column
    df['{new_col}'] = df['existing_column'] * 2  # Your actual calculation here
    ```"""
    
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
    generated_code = response
    if '```' in response:
        # Try to extract code between code blocks if present
        code_blocks = response.split('```')
        if len(code_blocks) >= 3:  # At least one complete code block
            generated_code = code_blocks[1].strip()
            if generated_code.startswith('python'):
                generated_code = generated_code[6:].strip()
    
    print("\nStep 6: Generated code:")
    print(f"```python\n{generated_code}\n```")
    
    print("\nStep 7: Executing generated code...")
    local_vars = {'df': df.copy()}
    
    try:
        exec(generated_code, {}, local_vars)
        df = local_vars['df']
        
        # Verify the new column was created
        if new_col in df.columns:
            print(f"✓ Successfully created new column '{new_col}'")
            print(f"✓ Sample values: {df[new_col].head(3).tolist()}")
            
            # Show the updated DataFrame with the new column
            print("\nUpdated DataFrame (first 3 rows with new column):")
            print(df.head(3).to_string())
        else:
            print(f"⚠ Warning: Column '{new_col}' was not created by the code")
            print(f"⚠ Available columns: {', '.join(df.columns)}")
        
        print(f"\nStep 8: Saving results to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"✓ Successfully saved DataFrame with {len(df.columns)} columns to {output_file}")
    except Exception as e:
        print(f"❌ Error executing generated code: {str(e)}")
        print("⚠ The output file was not created")

# Usage
if __name__ == "__main__":
    print("=== CSV Analyzer with LLM ===")
    process_csv(
        input_file="input.csv",
        output_file="output.csv",
        user_prompt="The derived column should be the average of the other columns"
    )
    print("=== Process completed ===")
