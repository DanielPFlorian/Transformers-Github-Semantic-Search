# Transformers-Github-Semantic-Search

**Transformers-Github-Semantic-Search** is a demonstration on how to create a dataset for an NLP Application and use it to build a semantic search engine. We will use the [Datasets](https://huggingface.co/docs/datasets/en/index) and [Transformers](https://huggingface.co/docs/transformers/en/index) Python libraries from [Huggingface](https://huggingface.co/) to complete this task. Using [requests](https://pypi.org/project/requests/) and the [GitHub Rest API](https://docs.github.com/en/rest?apiVersion=2022-11-28) we'll pull issues from the [Transformers Github Repository](https://github.com/huggingface/transformers) then proceed to clean up and augment the dataset with comments. Next we'll build the semantic search engine that will help us find answers to questions and issues we may have about the repository using tokenizers, text emebeddings, and [FAISS](https://faiss.ai/).

### Techniques Used
- NLP (Natural Language Processing)
- FAISS indexing for Semantic Search
- Dataset creation using Request and GitHub API
- Dataset Exploration, Cleaning and Augmentation
- Text Embedding creation using Tokenizer

### How to use the create_dataset.py script

Before running the `create_dataset.py` script, ensure you have Python installed and the required libraries: `requests`, `pandas`, `datasets`, and `tqdm`. You can install these using pip:

```bash
pip install requests pandas datasets tqdm
```

Additionally, you need to set a GitHub token as an environment variable named `GITHUB_TOKEN`. This token is required to authenticate with the GitHub API and avoid rate limits. You can create a personal access token on GitHub ([https://github.com/settings/tokens](https://github.com/settings/tokens)).  Once you have your token, set it as an environment variable in your terminal or system.

To run the script, navigate to the directory containing `create_dataset.py` in your terminal and execute:

```bash
python create_dataset.py
```

This script will download issues from the `huggingface/transformers` repository, augment them with comments, and save the dataset to `transformers-issues-with-comments.jsonl`.

**Note:**  Due to GitHub API rate limits, especially when fetching comments, the script may take a considerable amount of time to complete, and there might be pauses due to rate limit handling as indicated in the logs.

### How to use the create_dataset.py script

Before running the `create_dataset.py` script, ensure you have Python installed and the required libraries: `requests`, `pandas`, `datasets`, and `tqdm`. You can install these using pip:

```bash
pip install requests pandas datasets tqdm
```

Additionally, you need to set a GitHub token as an environment variable named `GITHUB_TOKEN`. This token is required to authenticate with the GitHub API and avoid rate limits. You can create a personal access token on GitHub ([https://github.com/settings/tokens](https://github.com/settings/tokens)).  Once you have your token, set it as an environment variable in your terminal or system.

To run the script, navigate to the directory containing `create_dataset.py` in your terminal and execute:

```bash
python create_dataset.py
```

This script will download issues from the `huggingface/transformers` repository, augment them with comments, and save the dataset to `transformers-issues-with-comments.jsonl`.

**Note:**  Due to GitHub API rate limits, especially when fetching comments, the script may take a considerable amount of time to complete, and there might be pauses due to rate limit handling as indicated in the logs.

### How to use the semantic_search.py script

To run the semantic search script, you need to have Python installed along with the required libraries (`datasets`, `transformers`, `pandas`, and `torch`). You can install these libraries using pip:

```bash
pip install datasets transformers pandas torch
```

Once you have the dependencies installed, you can run the script from your terminal using the following command:

```bash
python semantic_search.py "your search question here"
```

Replace `\"your search question here\"` with the question you want to search for in the issues. For example:

```bash
python semantic_search.py "How to use transformers for text classification?"
```

**Optional arguments:**

- `--save`:  This argument allows you to save the embeddings dataset to disk. This is useful if you want to reuse the dataset later without re-computing the embeddings.
  ```bash
  python semantic_search.py \"your search question here\" --save
  ```
- `--load`: This argument allows you to load a previously saved embeddings dataset from disk. If you use this argument, the script will load the embeddings from disk instead of computing them again.
  ```bash
  python semantic_search.py \"your search question here\" --load
  ```

The script will output the top 5 most relevant comments, along with their scores, titles, and URLs, based on your search question.

