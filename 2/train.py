import os
import re
from gensim.models import word2vec
from gensim.models import doc2vec
import jieba

chs_match_pattern = re.compile(r'[0-9a-zA-Z\u4e00-\u9fa5]+')


def clean_text(text: str) -> str:
    return ''.join(chs_match_pattern.findall(text))


def train_data_list() -> list[str]:
    r = []
    for root, dirs, files in os.walk('2019'):
        for file in files:
            r.append(os.path.join(root, file))
    return r


def train_data_read(filepath: str) -> str:
    f = open(filepath, 'r', encoding='utf-8')
    data = clean_text(''.join(f.readlines()))
    f.close()
    return data


if __name__ == '__main__':
    data_filepath_list = train_data_list()
    datas: list[str] = []
    for i in data_filepath_list:
        datas.append(train_data_read(i))
    sens_list = [jieba.lcut(i) for i in data_filepath_list]
    model = word2vec.Word2Vec(sens_list, min_count=1, iter=20)
    model.save("word2vec.model")
