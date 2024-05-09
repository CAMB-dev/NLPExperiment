import os
import re
import gensim
from gensim.models import word2vec
from gensim.models import doc2vec
import jieba
import logging
import datetime

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filemode="w",
    filename=f"train-{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S%z')}.log",
)

chs_match_pattern = re.compile(r"[0-9a-zA-Z\u4e00-\u9fa5]+")


def clean_text(text: str) -> str:
    return "".join(chs_match_pattern.findall(text))


def train_data_list() -> list[str]:
    r = []
    for root, dirs, files in os.walk("2019"):
        for file in files:
            r.append(os.path.join(root, file))
    return r


def train_data_read(filepath: str) -> str:
    f = open(filepath, "r", encoding="utf-8")
    data = clean_text("".join(f.readlines()))
    f.close()
    return data


def read_corpus(texts: list[str]) -> list[doc2vec.TaggedDocument]:
    r = []
    for i, text in enumerate(texts):
        tokens = gensim.utils.simple_preprocess(text)
        r.append(doc2vec.TaggedDocument(tokens, [i]))
    return r


if __name__ == "__main__":
    data_filepath_list = train_data_list()
    datas: list[str] = []
    for i in data_filepath_list:
        datas.append(train_data_read(i))
    print(len(datas))

    sens_list = [jieba.lcut(i) for i in datas]
    model = word2vec.Word2Vec(
        sens_list, vector_size=50, min_count=5, epochs=1000, workers=64
    )
    model.build_vocab(corpus_iterable=sens_list)
    model.train(
        corpus_iterable=datas, total_examples=model.corpus_count, epochs=model.epochs
    )
    model.save("word2vec.model")

    train_corpus = list(read_corpus(datas))
    model = doc2vec.Doc2Vec(vector_size=50, min_count=5, epochs=1000, workers=64)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec.model")
