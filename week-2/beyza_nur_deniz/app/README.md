# Competitive Programming Helper

## What It Does

The Competitive Programming Helper is your go-to tool for digging through PDF documents, turning text into searchable embeddings, and answering your queries with the help of Pinecone and Gemini AI. It’s perfect for finding specific info in large documents quickly.

## Cool Features

- **PDF Processing**: Pulls text from PDF files and splits it into manageable chunks.
- **Embedding Generation**: Transforms text chunks into vector embeddings with a pre-trained model.
- **Pinecone Integration**: Saves and searches through embeddings using Pinecone’s vector database.
- **Generative AI**: Answers your questions based on the PDF content with help from the Gemini AI model.

## How to Set It Up

1. **API Keys**

   You’ll need API keys for Pinecone and Gemini. Create a file named `local.py` in your project folder and add the following content:

   ```python
   gemini_api_key = "<your_gemini_api_key>"
   pinecone_api_key = "<your_pinecone_api_key>"
   root_path = "<path_to_pdf_files>"
   ````

2. **Install Dependencies**

    Before you start, make sure to install all the necessary packages by running:

    ```bash
    pip install -r requirements.txt
    ```

## Metrics

working on


## Acknowledgments

- [Pinecone](https://www.pinecone.io/) for vector database services.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for pre-trained models.
- [LangChain](https://www.langchain.com/) for NLP tools and libraries.
- [Google Generative AI](https://cloud.google.com/generative-ai) for AI models.

