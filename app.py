from flask import Flask,request, jsonify, g

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import FAQPipeline
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers

app = Flask(__name__)

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="haystack_tutorial",
                                            embedding_field="question_emb",
                                            embedding_dim=768,
                                            excluded_meta_data=["question_emb"])

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=True)

pipe = FAQPipeline(retriever=retriever)



@app.route("/webhook", methods=['POST'])
def index():
    print(request.get_json())
    prediction = pipe.run(query="How is the virus spreading?", top_k_retriever=10)
    print_answers(prediction, details="all")
    return "received from RASA server"

if __name__ == "__main__":
    app.run(debug=False)
