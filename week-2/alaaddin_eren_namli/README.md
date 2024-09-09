# A SIMPLE RAG APPLICATION

## What It Does
By taking the advantages of LangChain and RAG technique, this application gets the up-to-date data from documents and processes the relevant current information according to the user prompt. Then, it provides the user with accurate and up-to-date information.

## Features
- **Embeddings:** Processes the text as small chunks and embed them.

- **Doesn't Require Fine Tuning:** Because of the RAG technique, the model gets up-to-date and new data from external sources instead of fine-tuning.

- **Generative AI:** Gives answers to your prompts according to information in the documents with the help of Gemini AI model.

## Getting Started

### Prerequisites
Before starting, ensure that you have installed all the required libraries. In order to do this, run the following command:
```bash
pip install -r requirements.txt
```

### Usage
When the dependencies are installed, you can start to use the program. Give the external data as Document Objects and model provides you with the up-to-date and accurate answer based on your data and prompt.