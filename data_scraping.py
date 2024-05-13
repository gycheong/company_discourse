import praw  # https://praw.readthedocs.io/
from praw.models import MoreComments

from sentence_transformers import SentenceTransformer  # https://sbert.net/

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, tqdm_notebook  # https://tqdm.github.io/
from datetime import datetime  # https://docs.python.org/3/library/datetime.html
import os.path  # https://docs.python.org/3/library/os.path.html


# Input: submission class from PRAW
# Output: DataFrame with comment ID, author, time, body, and score
def get_comments_from_submission(submission):
    submission.comments.replace_more(limit=None)
    submission_list = submission.comments.list()
    comment_body = []
    comment_time = []
    comment_id = []
    comment_author = []
    comment_score = []

    for comment in submission_list:
        if comment.body != '[deleted]':
            comment_body.append(comment.body)
            comment_time.append(comment.created_utc)
            comment_id.append(comment.id)
            comment_author.append(comment.author)
            comment_score.append(comment.score)

    data = {'ID': comment_id,
            'Author': comment_author,
            'Time': comment_time,
            'Comment': comment_body,
            'Score': comment_score}

    return pd.DataFrame(data)

# Input: subreddit = subreddit name,
#        n = number of submissions
# Output: list of submissions
def get_hot_submissions(subreddit, n):
    data = []
    for submission in reddit.subreddit(subreddit).hot(limit=n):
        data.append(submission)

    return data


# Input: list of submissions
# Output: all comments of all submissions
def get_all_comments(submissions_list):
    data = [get_comments_from_submission(s) for s in submissions_list]  # list of DataFrames
    return pd.concat(data)


reddit = praw.Reddit("bot1")

# Example:
# submissions = get_hot_submissions("Costco", 10)
# result = get_all_comments(submissions)