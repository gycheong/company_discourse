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


model = SentenceTransformer("all-mpnet-base-v2")
# query = 'The quality was very good.'

df = pd.read_csv('Data/costco_google_reviews.csv')
df_filtered = df[df['text'].notna()]
comments = df_filtered['text'].tolist()
comments_embed = model.encode(comments, show_progress_bar=True)
vectors = comments_embed.tolist()
df_final = df_filtered.assign(Vector=vectors)
df_final.to_csv('Data/Vectorized/costco_google_reviews_all-mpnet-base-v2.csv')
