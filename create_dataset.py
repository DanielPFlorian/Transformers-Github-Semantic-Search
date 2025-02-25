import os
import time
import math
from pathlib import Path
import requests
import pandas as pd
import datasets
from tqdm.notebook import tqdm
import logging
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('create_dataset.log'), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# GitHub token for authorization
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

if not GITHUB_TOKEN:
    logger.error("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
    exit(1)

# Headers for API request
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# create a function that can download all the issues from a GitHub repository.
# 5,000 requests per hour is the GitHub rate limit for users
def fetch_issues(
    owner: str = "huggingface",
    repo: str = "transformers",
    num_issues: int = 30000,
    issues_path: Path = Path("."),
) -> None:
    """
    Fetch issues from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        num_issues (int): The number of issues to fetch.
        rate_limit (int): The rate limit for API requests.
        issues_path (Path): The path to save the issues.

    Returns:
        None
    """

    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        try:
            # Query with state=all to get both open and closed issues
            query = f"issues?page={page}&per_page={per_page}&state=all"
            issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
            issues.raise_for_status()  # Raise an exception for HTTP errors
            batch.extend(issues.json())

            # Check rate limit after each request
            remaining_requests = int(issues.headers.get('X-RateLimit-Remaining', 0))
            if remaining_requests <= 1:
                reset_time = int(issues.headers.get('X-RateLimit-Reset', 0))
                sleep_time = max(reset_time - time.time(), 0) + 10  # Add buffer
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching issues: {e}")
            continue

    all_issues.extend(batch)
    df = (
        pd.DataFrame.from_records(all_issues)
        .dropna(axis=1, how="all")
    )
    output_file = issues_path / f"{repo}-issues.jsonl"
    df.to_json(output_file, orient="records", lines=True)
    logger.info(
        f"Downloaded all the issues for {repo}! Dataset stored at {output_file}"
    )

def get_comments(issue_number: int, repo_path: str) -> List[str]:
    """
    Fetch comments for a specific GitHub issue from a given repository path.

    Args:
        issue_number (int): The number of the GitHub issue.

    Returns:
        List[str]: A list of comment bodies.
    """
    url = (f"https://api.github.com/repos/{repo_path}/issues/{issue_number}/comments")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Check rate limit after each request
        remaining_requests = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining_requests <= 1:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - time.time(), 0) + 10  # Add buffer
            logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

        return [r["body"] for r in response.json()]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating comments for issue {issue_number}: {e}")
        return []


if __name__ == "__main__":
    try:
        owner = "huggingface"
        repo = "transformers"
        num_issues = 30000
        issues_path = Path(".")

        # Fetch issues and save to JSONL
        fetch_issues(owner, repo, num_issues, issues_path)

        input_file = issues_path / f"{repo}-issues.jsonl"
        output_file = issues_path / f"{repo}-issues-with-comments.jsonl"

        # Check if the file exists before loading
        if not input_file.exists():
            logger.error(f"Issues file not found: {input_file}. Please check if the fetch_issues function completed successfully.")
            exit(1)

        # Load issues dataset with error handling
        try:
            logger.info(f"Loading dataset from {input_file}")
            issues_dataset = datasets.load_dataset("json", data_files=str(input_file), split="train")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            exit(1)

        # Map the dataset to add a new key-value pair indicating whether the issue is a pull request
        logger.info("Processing dataset")
        issues_dataset = issues_dataset.map(
            lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
        )

        # Augment the dataset with issue comments
        logger.info("Fetching comments for each issue")
        issues_with_comments_dataset = issues_dataset.map(
            lambda x: {"comments": get_comments(x["number"], repo_path=f"{owner}/{repo}")}
        )

        # Save the augmented dataset
        logger.info(f"Saving augmented dataset to {output_file}")
        issues_with_comments_dataset.to_json(str(output_file), orient="records", lines=True)

        logger.info("Process completed successfully")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)