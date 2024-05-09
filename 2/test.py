import gensim
from gensim import corpora
from gensim import similarities
from gensim.models import word2vec
from gensim.models import doc2vec
import logging
import datetime
import jieba
import numpy as np


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filemode="w",
    filename=f'test-{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S%z')}.log',
    encoding='utf-8'
)


def segment(s: str):
    return list(jieba.cut(s))

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def test_word2vec(s1: str, s2: str):
    model = word2vec.Word2Vec.load("word2vec.model")
    similars = model.wv.most_similar("人民")
    similarity = model.wv.similarity(s1, s2)

    logging.info('Word2Vec Tests')

    logging.info(f'人民 的相似词: {similars}')
    print(f'相似词: {similars}')
    logging.info(f"{s1} 和 {s2} 的相似度: {similarity}")
    print(f"{s1} 和 {s2} 的相似度: {similarity}")


def test_doc2vec():
    model = doc2vec.Doc2Vec.load('doc2vec.model')
    doc1 = model.dv[0]
    doc2 = model.dv[1]
    similar_docs = model.dv.most_similar(0, topn=5)
    similarity = cosine_similarity(doc1, doc2)

    logging.info('Doc2Vec Tests')

    logging.info(f'doc1的向量:\n{doc1}')
    print(f'doc1的向量:\n{doc1}')
    logging.info(f'相似doc:\n{similar_docs}')
    print(f'相似doc:\n{similar_docs}')
    logging.info(f'doc1 和 doc2 的相似度: {similarity}')
    print(f"doc1 和 doc2 的相似度: {similarity}")

if __name__ == "__main__":
    test_word2vec("人民", "群众")
    test_doc2vec()
