from sentence_transformers import SentenceTransformer, util
import os, base64, re, logging
from elasticsearch import Elasticsearch, helpers
import csv
import time
import tqdm.autonotebook
import pickle


# Log transport details (optional):
logging.basicConfig(level=logging.INFO)

# Parse the auth and host from env:
bonsai = "https://hh73cu7mha:8a9jjb5z88@galadriel-1337462399.us-west-2.bonsaisearch.net:443"
auth = re.search('https\:\/\/(.*)\@', bonsai).group(1).split(':')
host = bonsai.replace('https://%s:%s@' % (auth[0], auth[1]), '')

# optional port
match = re.search('(:\d+)', host)
if match:
  p = match.group(0)
  host = host.replace(p, '')
  port = int(p.split(':')[1])
else:
  port=443

# Connect to cluster over SSL using auth for best security:
es_header = [{
 'host': host,
 'port': port,
 'use_ssl': True,
 'http_auth': (auth[0],auth[1])
}]

# Instantiate the new Elasticsearch connection:
es = Elasticsearch(es_header)

# Verify that Python can talk to Bonsai (optional):
es.ping()

#es = Elasticsearch()

model = SentenceTransformer('./paraphrase-mpnet-base-v2')


#Download dataset if needed
#if not os.path.exists(dataset_path):
#    print("Download dataset")
#    util.http_get(url, dataset_path)

with open('./data/gen_train.pk','rb') as f:
    gen_train = pickle.load(f)

#Index data, if the index does not exists
if not es.indices.exists(index="MedDG_Train_Set"):
    
    es_index = {
        "mappings": {
            "properties": {
            "history": {
                "type": "text"
            },
            "response": {
                "type": "text"
            },
            "history_vector": {
                "type": "dense_vector",
                "dims": 768
            }
            }
        }
    }

    es.indices.create(index='MedDG_Train_Set', body=es_index, ignore=[400])
    chunk_size = 50
    print("Index data (you can stop it by pressing Ctrl+C once):")
    with tqdm.tqdm(total=len(gen_train)) as pbar:
        for start_idx in range(0, len(gen_train), chunk_size):
            end_idx = start_idx+chunk_size

            #embeddings = model.encode(questions[start_idx:end_idx], show_progress_bar=False)
            bulk_data = []
            for instance in gen_train[start_idx:end_idx]:
                concate_his=""
                for i in range(len(instance["history"])):
                    concate_his=concate_his+instance["history"][i]
                #print(concate_his)
                embedding = model.encode(concate_his, show_progress_bar=False)
            #for qid, question, embedding in zip(qids[start_idx:end_idx], questions[start_idx:end_idx], embeddings):
                bulk_data.append({
                        "_index": 'MedDG_Train_Set',
                        "_id": instance["id"],
                        "_source": {
                            "history": instance["history"],
                            "response": instance["response"],
                            "history_vector": embedding
                        }
                    })

            helpers.bulk(es, bulk_data)
            pbar.update(chunk_size)





#Interactive search queries
while True:
    inp_question = input("Please enter a question: ")

    encode_start_time = time.time()
    question_embedding = model.encode(inp_question)
    encode_end_time = time.time()

    #Lexical search
    bm25 = es.search(index="quora", body={"query": {"match": {"question": inp_question }}})

    #Sematic search
    sem_search = es.search(index="quora", body={
          "query": {
            "script_score": {
              "query": {
                "match_all": {}
              },
              "script": {
                "source": "cosineSimilarity(params.queryVector, doc['question_vector']) + 1.0",
                "params": {
                  "queryVector": question_embedding
                }
              }
            }
          }
        })

    print("Input question:", inp_question)
    print("Computing the embedding took {:.3f} seconds, BM25 search took {:.3f} seconds, semantic search with ES took {:.3f} seconds".format(encode_end_time-encode_start_time, bm25['took']/1000, sem_search['took']/1000))

    print("BM25 results:")
    for hit in bm25['hits']['hits'][0:5]:
        print("\t{}".format(hit['_source']['question']))

    print("\nSemantic Search results:")
    for hit in sem_search['hits']['hits'][0:5]:
        print("\t{}".format(hit['_source']['question']))

    print("\n\n========\n")