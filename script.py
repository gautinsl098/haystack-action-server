from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import FAQPipeline
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
import pandas as pd
import requests


document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="haystack_tutorial",
                                            embedding_field="question_emb",
                                            embedding_dim=768,
                                            excluded_meta_data=["question_emb"])

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=True)

temp = requests.get("https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/faqs/faq_covidbert.csv")
open('small_faq_covid.csv', 'wb').write(temp.content)

# Get dataframe with columns "question", "answer" and some custom metadata
df = pd.read_csv("small_faq_covid.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df["question"] = df["question"].apply(lambda x: x.strip())
print(df.head())

# Get embeddings for our questions from the FAQs
questions = list(df["question"].values)
df["question_emb"] = retriever.embed_queries(texts=questions)
df = df.rename(columns={"question": "text"})

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

pipe = FAQPipeline(retriever=retriever)

prediction = pipe.run(query="How is the virus spreading?", top_k_retriever=10)
print_answers(prediction, details="all")
