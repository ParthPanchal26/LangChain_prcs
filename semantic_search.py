import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("..")

from secret_keys import NVIDIA_API_KEY
import os

os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY

corpus = "In today’s rapidly changing world, technology plays a central role in shaping how people live, work, and communicate with one another. From smartphones that keep us constantly connected to artificial intelligence systems that can analyze vast amounts of data in seconds, innovation has transformed nearly every aspect of daily life. While these advancements have brought convenience and efficiency, they have also raised important questions about privacy, security, and the impact of automation on employment. For instance, many industries now rely on machines to perform tasks that were once done by humans, leading to both increased productivity and concerns about job displacement. At the same time, technology has created entirely new career paths and opportunities that did not exist a few decades ago. Education, too, has evolved, with online learning platforms making knowledge more accessible to people around the globe. However, not everyone has equal access to these resources, which highlights the ongoing issue of the digital divide. As society continues to embrace technological progress, it becomes essential to find a balance between innovation and ethical responsibility, ensuring that the benefits of technology are shared widely while minimizing potential risks."
corpus = corpus.strip().split('.')
corpus = [s.strip() for s in corpus if s.strip()]
# print(corpus)

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

document_embeddings = embedder.embed_documents(corpus)
# print(document_embeddings)

query = "At the same time, technology has created entirely new career paths and opportunities"
query_embeddings = embedder.embed_query(query)

from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity([query_embeddings], document_embeddings)[0]
scores = sorted(list(enumerate(scores)), key=lambda x : x[1])

# print(scores)

doc_collection = []

for i, score in enumerate(scores):
    if score[1] > 0.5:
        doc_collection.append(score[1])

doc_collection = sorted(list(enumerate(doc_collection)), key = lambda x : x[1])
# print(doc_collection)

# if len(doc_collection) > 1:
for i in doc_collection:
    print(corpus[i[0]])

# index, score = sorted(list(enumerate(scores)), key=lambda x : x[1])[-1]

# if score > 0.5:
#     print(corpus[index])
# else:
#     print("No relevant info found")