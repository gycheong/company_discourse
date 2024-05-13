from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

query = 'The quality was very good.'

# Input: df with 'Comment' column
#        query string
#        SBERT model
def calculate_scores(df, query, model):
    comment_embeddings = model.encode(df['Comment'].tolist(), convert_to_tensor=True)
    query_embedding = model.encode(query)
    cosine_scores = util.cos_sim(comment_embeddings, query_embedding).numpy().tolist()
    df['Score'] = cosine_scores

    return df.sort_values(by=['Scores'], ascending=False)