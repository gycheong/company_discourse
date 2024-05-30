import pandas as pd
from sentence_transformers import SentenceTransformer, util

def calcuate_vectors(comments_list, sbert_model):
    comment_embeddings = sbert_model.encode(comments_list, show_progress_bar=True)
    return comment_embeddings


# Input: df with 'Comment' column
#        query string
#        SBERT model
def calculate_scores(df, query, model):
    comment_embeddings = model.encode(df['Comment'].tolist(), convert_to_tensor=True)
    query_embedding = model.encode(query)
    cosine_scores = util.cos_sim(comment_embeddings, query_embedding).numpy().tolist()
    df['Score'] = cosine_scores

    return df.sort_values(by=['Scores'], ascending=False)


# model_gte = SentenceTransformer("thenlper/gte-large")
# model_mxbai = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# query = 'The quality was very good.'
#
# df = pd.read_csv('Data/costco_google_reviews.csv')
# df_filtered = df[df['text'].notna()]
# comments = df_filtered['text'].tolist()
# comments_embed = model_gte.encode(comments, show_progress_bar=True)
# vectors = comments_embed.tolist()
# df_final = df_filtered.assign(Vector=vectors)
# df_final.to_csv('Data/Vectorized/costco_google_reviews_mxbai-embed-large-v1.csv')

df = pd.read_json('Data/Large/costco_2021_reviews.json', dtype={"user_id": str, "time": str})
df_filtered = df.dropna(subset='text')
df_filtered.to_json('Data/Large/costco_2021_reviews_filtered.json', orient='records', compression='infer')


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

n = 100000

df_filtered_split = list(divide_chunks(df_filtered, n))

for i in range(len(df_filtered_split)):
    df_filtered_split[i].to_json('Data/Large/costco_2021_reviews_filtered_' + str(i) + '.json', orient='records', compression='infer')

comments_0 = df_filtered_split[0]['text'].tolist()

model_gte = SentenceTransformer("thenlper/gte-large")
comment_embeddings_0 = model_gte.encode(comments_0, show_progress_bar=True)
