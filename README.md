# CSV Analyzer with LLM

A powerful tool that uses Large Language Models (LLMs) to automatically analyze CSV data and create intelligent derived columns based on your requirements.

## Overview

CSV Analyzer is designed to simplify data analysis by leveraging the power of AI to understand your data and create meaningful derived columns. Whether you need to transform data, create calculated fields, or apply machine learning techniques, this tool can help you do it with minimal coding.

## Demo Video

[![CSV Analyzer Demo]](https://www.loom.com/share/c035f57a09cf404eba99643ff970eae5?sid=9620938d-9327-4d6a-877a-6620b580410d)

*Click the image above to watch the demo video.


## How It Works

The program follows these steps:

1. **Load Data**: Reads your CSV file into a pandas DataFrame
2. **Analyze Data**: Examines the structure, data types, and sample values
3. **Generate Code**: Uses an LLM to create Python code for the derived column based on your prompt
4. **Execute Code**: Runs the generated code to create the new column
5. **Save Results**: Outputs the enhanced DataFrame to a new CSV file


## Why This Program Is Needed

Data analysis often requires creating derived columns based on existing data, which traditionally involves:

1. **Manual coding**: Writing custom pandas code for each transformation
2. **Domain expertise**: Understanding the data and what transformations make sense
3. **Trial and error**: Testing different approaches to find the best solution

CSV Analyzer automates this process by:

- **Leveraging AI**: Using LLMs to understand your data and generate appropriate code
- **Reducing development time**: Generating working code in seconds instead of hours
- **Providing flexibility**: Supporting both simple transformations and complex ML models
- **Making data analysis accessible**: Allowing non-programmers to create derived columns


### Key Features

- **Intelligent Code Generation**: Creates pandas code tailored to your specific data
- **Machine Learning Support**: Optionally allows ML techniques for more complex derivations
- **Detailed Logging**: Provides step-by-step information about the process
- **Error Handling**: Robust error detection and reporting
- **Flexible Prompting**: Customize what you want the derived column to represent

## Usage

### Local Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have Ollama installed and running with the Llama3 model:
   ```bash
   ollama run llama3
   ```

3. Run the program:
   ```bash
   python app.py
   ```

### Configuration

You can customize the behavior by modifying the parameters in the `process_csv` function call:

```python
process_csv(
    input_file="input.csv",  # Your input CSV file
    output_file="output.csv",  # Where to save the results
    user_prompt="The derived column should be the best fit for the data in the dataframe",  # What you want the column to represent
    allow_ml=True  # Set to False if you don't want ML techniques
)
```

### Example Prompts

- "Create a column that represents the sentiment of the text in column 'comments'"
- "Calculate a risk score based on the numeric columns"
- "Generate a binary flag that indicates potential fraud based on the transaction data"
- "Create a column that categorizes customers into segments based on their purchase history"


## Technical Details

### Architecture

- **LLM Integration**: Uses LangChain to interface with Ollama/Llama3
- **Code Generation**: Prompts the LLM to generate pandas code
- **Safe Execution**: Runs generated code in a controlled environment
- **Data Handling**: Uses pandas for all data operations

### Improvements

The latest version includes:
- Direct LLM calls for more reliable code generation
- Enhanced error handling for parsing LLM responses
- Improved prompting for better code quality
- Support for both simple transformations and ML techniques

## Requirements

- Python 3.8+
- pandas
- langchain
- langchain-ollama
- langchain-experimental
- Ollama with Llama3 model (for local version)

## License

[MIT License](LICENSE)