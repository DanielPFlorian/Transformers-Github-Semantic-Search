# Transformers-Github-Semantic-Search

**Transformers-Github-Semantic-Search** is a demonstration on how to create a dataset for an NLP Application and use it to build a semantic search engine. We will use the [Datasets](https://huggingface.co/docs/datasets/en/index) and [Transformers](https://huggingface.co/docs/transformers/en/index) Python libraries from [Huggingface](https://huggingface.co/) to complete this task. Using [requests](https://pypi.org/project/requests/) and the [GitHub Rest API](https://docs.github.com/en/rest?apiVersion=2022-11-28) we'll pull issues from the [Transformers Github Repository](https://github.com/huggingface/transformers) then proceed to clean up and augment the dataset with comments. Next we'll build the semantic search engine that will help us find answers to questions and issues we may have about the repository using tokenizers, text emebeddings, and [FAISS](https://faiss.ai/).

### Techniques Used
- NLP (Natural Language Processing)
- Dataset creation using Request and GitHub API
- Dataset Exploration, Cleaning and Augmentation
- Text Embedding creation using Tokenizer
- FAISS indexing for Semantic Search