import datasets
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import argparse

# Load the dataset
issues_dataset = datasets.load_dataset(
    "json", data_files="transformers-issues-with-comments.jsonl", split="train"
)

# Map the dataset to set the "is_pull_request" field based on whether the issue is a pull request or not
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

# Filtering for non-pull request issues with comments as these are most helpful for user queries and will introduce noise in the search engine
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

# Get the column names from the dataset
columns = issues_dataset.column_names

# Define the columns to keep in the dataset
columns_to_keep = ["title", "body","html_url", "comments"]

# Find the columns to remove from the dataset
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)

# Remove the columns from the dataset
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

# Set the format of the issues_dataset to "pandas"
issues_dataset.set_format("pandas")
# Create a new dataframe df by copying the entire issues_dataset
df = issues_dataset[:]

# Drop rows with missing values
df.dropna(axis=0,how = "any",inplace = True)

comments_df = df.explode("comments", ignore_index=True)

comments_dataset = datasets.Dataset.from_pandas(comments_df)

comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

def concatenate_text(examples):
    # Concatenate title, body, and comments
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)


# Load the pre-trained model checkpoint for multi-qa-mpnet-base-dot-v1
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Initialize the tokenizer using the pre-trained model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Initialize the model using the pre-trained model checkpoint
model = AutoModel.from_pretrained(model_ckpt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Make cuda optional
model.to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

    
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

# add faiss index
embeddings_dataset.add_faiss_index(column="embeddings")


# Remove test question and search code from global scope
if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Semantic search script for transformer issues.")
    parser.add_argument("question", type=str, help="The question to search for in the issues.")
    parser.add_argument("--save", action="store_true", help="Save embeddings dataset to disk")
    parser.add_argument("--load", action="store_true", help="Load embeddings dataset from disk")
    args = parser.parse_args()

    if args.save:
        embeddings_dataset.save_to_disk('embeddings_dataset')
        print("Embeddings dataset saved to disk")
        exit(0)

    if args.load:
        embeddings_dataset = datasets.load_from_disk('embeddings_dataset')
        print("Embeddings dataset loaded from disk")
        
    # Get the embeddings for the question
    question_embedding = get_embeddings([args.question]).cpu().detach().numpy()

    # Compute the nearest examples using embeddings dataset
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=5
    )

    # Convert samples dictionary to DataFrame and sort by scores
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()