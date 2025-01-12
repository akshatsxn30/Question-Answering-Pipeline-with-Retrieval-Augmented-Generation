# Question Answering Pipeline with Retrieval-Augmented Generation (RAG)

This repository contains a question-answering pipeline using the [Quora Question Answer Dataset](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset). 
The pipeline is built with a Retrieval-Augmented Generation (RAG) approach to improve response accuracy by retrieving the most relevant question-answer pairs from a vector database (Qdrant) during inference using phi3.5 to generate responses.

## Data Exploration and Insights

### Insights from EDA
1. **Question Lengths**: Most questions are short, with a peak around 20–30 characters. Long questions (>100 characters) are less frequent but still present.
2. **Answer Lengths**: Answers tend to be longer than questions, with peaks around 50–100 characters.
3. **Word Frequency**: Common words in the dataset include stopwords like "what," "how," and "is." Non-stop words reveal domain-specific terms.
4. **Correlation**: A weak positive correlation (~0.3) exists between question and answer lengths, indicating longer questions might lead to longer answers.

## RAG Pipeline Overview

### Steps:
1. **Embedding Model**: We use the `sentence-transformers/paraphrase-MiniLM-L3-v2` model to embed both questions and answers into a vector space.
2. **Storage and Retrieval**: Embeddings are stored in Qdrant (or any vector database of choice). A similarity search is performed to retrieve the most relevant question-answer pairs during inference.
3. **Generative Model**: The retrieved context is used by phi3.5 to generate responses.

### Key Insights:
- **Model Performance**:
    - The high BERTScore metrics F1 indicate strong semantic alignment between generated and reference answers.
    - Precision slightly exceeds recall, suggesting the model is more precise than comprehensive.
    - These scores indicate that the RAG pipeline is effectively retrieving and utilizing relevant context.

- **Data Distribution Impact**:
    - The question-answer length patterns (20-30 chars for questions, 50-100 for answers) suggest that the model needs to handle significant length variations.
    - The weak correlation (0.3) between question-answer lengths shows that the model must be flexible in generating both concise and detailed responses.

- **RAG System Effectiveness**:
    - Using `sentence-transformers/paraphrase-MiniLM-L3-v2` for embeddings aligns well with the question-answering task, as it is optimized for semantic similarity.
    - The high BERTScore suggests effective context retrieval from Qdrant.

## Evaluation

### BERTScore Evaluation Results:
- **Average Precision**: 0.8384
- **Average Recall**: 0.8254
- **Average F1**: 0.8315

These results reflect the strong semantic alignment between generated and reference answers.

## Improvements for the Pipeline

1. **Hybrid Retrieval**: Combining different retrieval techniques to improve context relevance and retrieval accuracy.
2. **Dynamic Context Windowing**: Implementing a dynamic context window to refine the scope of context used by the generative model.
3. **Structured Prompting**: Enhancing the answer generation with structured prompting to better guide the generative model.
4. **Comprehensive Evaluation**: Implementing a broader evaluation strategy to assess pipeline performance across various metrics.
5. **Domain Adaptation Features**: Introducing domain-specific features to adapt the model to specific types of question-answering tasks.
