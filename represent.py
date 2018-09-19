# coding=utf-8
# author=Dnsk

"""
输入：data模块的三种标准输出之一
输出：每个例子的向量表示，例子的粒度由输入的粒度决定

功能1：
    # 将句子列表转换为embedding之后的矩阵
    my_represent = Represent()
    my_matrix = my_represen.word_embedding('sgns.merge.bigram', sentence_list)
    # 注：sgns.merge.bigram为预训练好的embedding文件路径
    # sentence_list的格式为列表，其中每一个列表均为一个句子，每个句子也为列表，其中每个元素为一个词
    # 例如：[['我', '爱', '北京', '天安门'], ['门', '我们', '京东', '促销']] 是一个符合要求的输入
    # 对于Data类的三种输出，可以用空格来分割字符串，得到的结果就可以作为该函数的输入

    # 如果要调用normalize之后的矩阵
    my_represent = Represent()
    my_represen.word_embedding_normalize('sgns.merge.bigram', sentence_list)

功能2：
    # 将Data类的第二种标准输出转换为tfidf矩阵
    my_represent = Represent()
    my_matrix = my_represent.tfidf_matrix_2(output_from_data_2)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import numpy as np
import math


def normalize(matrix):
    """将输入的矩阵按行归一化"""
    for i in range(0, matrix.shape[0]):
        length = math.sqrt(np.sum(matrix[i]**2))
        if length == 0:
            length = 1
        matrix[i] = matrix[i]/length

    return matrix


class Represent:
    def __init__(self):
        self.initialized = 0
        pass

    def tf_matrix(self):
        pass

    def tfidf_matrix_2(self, output_from_data_2):
        """
        将第二种输出格式转化为tfidf矩阵
        每个句子需要用string表示，不同的词之间用空格分割
        """
        total_sentence_list = []
        for conversation in output_from_data_2:
            for sentence in conversation:
                total_sentence_list.append(sentence[5])

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(total_sentence_list)
        tfidf_matrix = tfidf_vectorizer.transform(total_sentence_list).toarray()
        return tfidf_matrix

    def lda_matrix(self):
        pass

    def word_embedding(self, file_path, sentence_list):
        """
        将每个单位转换为由word embedding的和/均值表示的特征向量
        :param file_path: embedding文件的位置
        :param sentence_list: 输入的句子的列表，注意每个句子应该用列表表示
        :return: 矩阵，每个句子一行，每一行为转换后的向量
        """

        """读取预训练好的中文embedding"""
        if self.initialized == 0:
            self.word2vec = {}  # 汉字和对应向量的索引
            self.vec = []  # 存放向量
            index = 0
            line_count = 0
            print('Loading embedding metrix')
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line:
                        line_count += 1
                        if line_count == 1:
                            continue
                        self.word2vec[line.split(' ')[0]] = index
                        index += 1
                        temp = line.split(' ')[1:-1]
                        self.vec.append([float(i) for i in temp])
                    else:
                        break
            self.initialized = 1

        """将输入转换为embedding的矩阵返回"""
        output = np.zeros((len(sentence_list), len(self.vec[0])))
        for i, sentence in enumerate(sentence_list):
            num_words = 0
            transformed_vec = np.zeros((1, len(self.vec[0])))
            for word in sentence:

                if word in self.word2vec:
                    num_words += 1
                    transformed_vec += np.array(self.vec[self.word2vec[word]])

            if num_words == 0:
                continue
            output[i] = transformed_vec/float(num_words)

        return output

    def word_embedding_normalize(self, file_path, sentence_list):
        """
        将每个单位转换为由word embedding的和/均值表示的特征向量，每个向量归一化
        :param file_path: embedding文件的位置
        :param sentence_list: 输入的句子的列表，注意每个句子应该用列表表示
        :return: 矩阵，每个句子一行，每一行为转换后的向量
        """

        """读取预训练好的中文embedding"""
        word2vec = {}  # 汉字和对应向量的索引
        vec = []  # 存放向量
        index = 0
        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line:
                    line_count += 1
                    if line_count == 1:
                        continue
                    word2vec[line.split(' ')[0]] = index
                    index += 1
                    print('%d finished' % index)
                    temp = line.split(' ')[1:-1]
                    vec.append([float(i) for i in temp])
                else:
                    break

        """将输入转换为embedding的矩阵返回"""
        output = np.zeros((len(sentence_list), len(vec[0])))
        for i, sentence in enumerate(sentence_list):
            transformed_vec = np.zeros((1, len(vec[0])))
            for word in sentence:
                if word in word2vec:
                    transformed_vec += np.array(vec[word2vec[word]])

            output[i] = transformed_vec

        return normalize(output)


def test():
    a = np.random.random((5, 5))
    b = normalize(a)
    print(b)
    print(np.sum(b[0]**2))


if __name__ == '__main__':
    # """测试代码"""
    # represent = Represent()
    # a = [['我', '爱', '北京', '天安门'], ['门', '我们', '京东', '促销']]
    # embedding_matrix = represent.word_embedding('sgns.merge.bigram', a)
    # print(embedding_matrix.shape)
    # print(embedding_matrix)

    test()
