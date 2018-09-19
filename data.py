"""
这是Data类重构后的结果，预处理部分抽象成了一个独立的类

使用方法1：得到原Data类的三种输出
    my_preprocessor = Preprocessor('stop_words.txt')
    a_data = Data('../data/chat.txt')
    a_data.seperate_conversation()
    a_data.preprocess(my_preprocessor, format=3)

    至此可以得到类似原Data实现的三种输出格式
    注意，调用第几种格式对应了a_data.preprocess()中format参数的选项，例如，需要第三种输出，format=3

    1. 第一种输出:每个元素为一个对话，对话中所有句子合并成一个字符串
    a_data.conversation_merge_preprocessed

    2. 第二种输出:每个元素为一个对话，为列表；列表中每句话也用列表表示，包含了对话方以及对话内容的信息
    a_data.conversation_all_preprocessed_seperate

    3. 第三种输出：同第二种，唯一的区别是同一方的连续发言被合并成一句话；每句话的列表中现在只有两个元素，第一个为说话方，第二个为句子内容
    a_data.conversation_QAQAQ_preprocessed

使用方法2：生成用于分类的训练数据
    详细用法见Trainer.get_x_y()

    # 初始化需要的类
    my_preprocessor = Preprocessor('stop_words.txt')
    a_data = Data('../data/chat.txt')
    my_merger = classMerger()

    # Data类做处理，产生训练数据，按照numpy矩阵的格式输出
    a_data.seperate_conversation()
    a_data.preprocess(a_preprocessor, format=2)
    a_data.generate_training_sample(Sampler())
    a_data.load_clustering_results('db_min_20.txt')
    a_data.prepare_sample(remove_invalid_label=remove_invalid)
    self.train_x, self.train_y, self.test_x, self.test_y = a_data.get_train_test_sample(embedded_matrix, a_class_merger)
"""

import numpy as np
import random as rd

from preprocessor import Preprocessor
from sampler import Sampler
from represent import Represent


class Data:
    """
    Data.conversation_all:
    """

    def __init__(self, file_path, num=None):
        """
        将对话按照每句话的格式存储，为列表
        其中的每个元素为一句话，包含它的所属对话id等信息，也为列表
        """
        self.sentence_list = []

        """将京东数据文件读入，按句存储"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if not num:
                lines = f.readlines()
            else:
                lines = []
                for i in range(num):
                    lines.append(f.readline())

            for line in lines:
                try:
                    values = line.strip('\r\n').split('\t')
                    conversation_id = values[0]
                    user_id = values[1]
                    answer = values[2]  # 如果是客服的回答，为1，否则为0
                    transfer = values[3]  # 如果是转接，为1
                    repeat = values[4]  # 如果是当天内的第一次咨询，为0
                    content = ''.join(values[6:]).strip()
                    self.sentence_list.append([conversation_id, user_id, answer, transfer, repeat, content])
                except:
                    print(line.strip('\r\n').split('\t'))

    def seperate_conversation(self):
        """
        Split conversation in different ways
        """
        self.conversation_all = self._split_conversations(self.sentence_list)
        self.conversation_all_content_only = []
        for conversation in self.conversation_all:
            self.conversation_all_content_only.append(' '.join([sentence[5] for sentence in conversation]))

        # 得到按照QAQAQ划分的数据，每个元素为一个tuple，分别为1/0和对话内容
        self.conversation_QAQAQ = self._split_QAQAQ(self.conversation_all)

        # 得到按照对话划分的数据，每个元素为一个列表，列表中每个元素为一个tuple，分别为用户id，是否客服所说，是否当天第一次咨询等特征
        self.conversation_split_sentence_within_conversation = self._split_sentence_within_conversation_new(self.conversation_all)

        # 得到每句话对于对话的所属关系和顺序
        self.index_seperation = self._index_seperation(self.conversation_split_sentence_within_conversation)

        # 第一行为['session_id', 'user_id', 'waiter_send', 'is_transfer', 'is_repeat', '']
        # 去除掉
        # 前面调用sentence_list的split_conversations函数已经考虑到了这种情况，所以没有影响
        self.sentence_list = self.sentence_list[1:]

    def preprocess(self, a_preprocessor, format=1):
        """
        使用Preprocessor的实例来进行预处理
        此处为了使train data和test data获得的处理保持一致，preprocessor被专门抽象出来成一个类
        :param a_preprocessor: 一个Preprocessor实例
        :param format: 指定需要的输出格式，
            1 - 每轮对话合并成一个字符串
            2 - 每句话分别处理
            3 - 合并同一方的连续发言
        """

        if format == 1:
            # 第一种输出格式
            # 每个元素为一个对话，对话中所有句子合并成一个字符串
            self.conversation_merge_preprocessed = [a_preprocessor.preprocess_pipeline(conv) for conv in self.conversation_all_content_only]

        if format == 2:
            # 第二种输出格式
            # 每个元素为一个对话，为列表
            # 列表中每句话也用列表表示，包含了对话方以及对话内容的信息
            self.conversation_all_preprocessed_seperate = []
            self.conversation_all_preprocessed_seperate_flaten = []
            for conversation in self.conversation_split_sentence_within_conversation:
                this_conversation = []
                for sentence in conversation:
                    this_conversation.append([sentence[0], sentence[1], sentence[2], sentence[3],
                                              sentence[4], a_preprocessor.preprocess_pipeline(sentence[5])])
                    self.conversation_all_preprocessed_seperate_flaten.append([sentence[0], sentence[1], sentence[2],
                                                                               sentence[3], sentence[4],
                                                                               a_preprocessor.preprocess_pipeline(sentence[5])])
                self.conversation_all_preprocessed_seperate.append(this_conversation)

        if format == 3:
            # 第三种输出格式
            # 同第二种，唯一的区别是同一方的连续发言被合并成一句话
            # 每句话的列表中现在只有两个元素，第一个为说话方，第二个为句子内容
            self.conversation_QAQAQ_preprocessed = []
            for conversation in self.conversation_QAQAQ:
                this_conversation = []
                for sentence in conversation:
                    this_conversation.append([sentence[0], a_preprocessor.preprocess_pipeline(sentence[1])])
                self.conversation_QAQAQ_preprocessed.append(this_conversation)

    def load_clustering_results(self, clustering_file_path):
        """
        :param clustering_file_path:
        :return:
        """
        self.clustering_label_index = {}
        self.clustering_index_label = {}
        self.invalid_sample_index = []  # index of sample with label == -1
        with open(clustering_file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                index, label = line.split(';')
                index = int(index)
                label = int(label.strip())

                # insert index into self.invalid_sample_index if label == -1
                if label == -1:
                    self.invalid_sample_index.append(index)

                if label in self.clustering_label_index:
                    self.clustering_label_index[label].append(index)
                if label not in self.clustering_label_index:
                    self.clustering_label_index[label] = [index]
                self.clustering_index_label[index] = label

    def generate_training_sample(self, a_sampler):
        """
        使用sampler对象的实例，进行采样产生训练样本，存储到self.samples
        根据第三种输出格式修改过 - 20180724
        """
        # 将my_data.index_seperation转换为标示说话方的列表，0表示客户，1表示客服
        # 输出格式： [[0,0,0,1,1,0,1], [0,1,0,1,1,0], ...]
        # 每个元素为一个列表，每个子列表中按序装的是说话方
        transformed = []  # the output of this stage
        for conversation in self.conversation_QAQAQ_preprocessed:
            part = []
            for sentence in conversation:
                part.append(int(sentence[0]))
            transformed.append(part)

        self.samples = a_sampler.sampling(transformed)
        self.samples = [_ for _ in self.samples if _ != None]

    def prepare_sample(self, ratio=0.1, remove_invalid_label=False):
        """
        根据ratio，划分self.samples为训练集和测试集，用index的方法存储到self.train_X_index等属性中
        """

        index = list(range(0, len(self.samples)))

        if remove_invalid_label is True:
            invalid_index_of_sample = []
            for i, element in enumerate(self.samples):
                if element[1] in self.invalid_sample_index:
                    invalid_index_of_sample.append(i)

            index = [i for i in index if i not in invalid_index_of_sample]

        train_set_num = int((1-ratio) * len(index))

        self.train_X_index = index[:train_set_num]
        self.train_Y_index = index[:train_set_num]
        self.test_X_index = index[train_set_num:]
        self.test_Y_index = index[train_set_num:]

    def get_train_test_sample(self, embedding_matrix, my_merger):
        X, Y = self._samples2XY(embedding_matrix, my_merger)
        return X[self.train_X_index], Y[self.train_Y_index], X[self.test_X_index], Y[self.test_Y_index]

    def _split_conversations(self, conversation_list):
        """
        （内部方法）
        从含有对话编号的句子列表中将会话从中分割，生成若干conversation对象
        """
        # 用于记录当前对话的id
        current_conversation_id = 0

        # 列表，用来放置一段对话（一个session）
        current_conversation_list = []

        # conversation_total为一个列表，其中的每一个元素是一个对话
        # 对话用列表表示，其中每个元素为一句话
        # 每句话用列表表示，sentence[0]为对话的id，sentence[1]为用户id，sentence[2]为是否为客服回答，sentence[3]为是否为转接，sentence[4]为是否是当天内第一次咨询，sentence[5]为句子的内容
        conversation_total = []
        for sentence in conversation_list:
            if sentence[0] != current_conversation_id:
                if current_conversation_list != []:
                    conversation_total.append(current_conversation_list)
                current_conversation_id = sentence[0]
                current_conversation_list = []
                current_conversation_list.append(sentence)
            else:
                current_conversation_list.append(sentence)
        conversation_total.append(current_conversation_list)
        return conversation_total[1:]

    def _split_QAQAQ(self, conversatsion_list):
        """
        按照Q-A的格式划分对话
        :param conversation_list: self.conversation_all
        :return: 为列表，每个元素为一个按照QAQA划分的对话，样例如下

        [(0, ' 什么时候能发货啊 什么时候能发货啊 得等多久啊 我家狗狗等着吃饭呢 ?'),
        (1, ' 请您稍等一下，正在为您核实处理中哦~ 有什么问题我可以帮您处理或解决呢? [数字x] 耐威克(Navarch)
        宠物天然粮[姓名x]犬成犬狗粮[数字x]kg'), (0, ' 对的。'), (1, ' 这款商品库房还没有到货’'), (0, ' 什么时候能到货啊?'),
        (1, ' 可以给您操作有货的先发'), (0, ' 零食那个有货呗? 狗粮一袋都没有么? 有的话都给我先发吧。 ?'), (1, ' 稍等哈'), (0, ' 恩'),
        (1, ' 查询以操作了有货先发 您的订单现在是两个订单 建议您关注一下订单的物流信息哈'), (0, ' 狗粮啥时候能有货?'),
        (1, ' 预计一周左右'), (0, ' 好吧，希望快点发货'), (1, ' 好的亲 请问还有其他还可以帮到您的吗?'), (0, ' 没了')]

        """
        conversation_total = []
        for conversation in conversatsion_list:
            this_conversation = []
            current_turn_content = ''
            current_party = 2
            for sentence in conversation:
                if int(sentence[2]) != current_party:
                    if current_turn_content != '':
                        this_conversation.append((current_party, current_turn_content))
                    current_party = int(sentence[2])
                    current_turn_content = ''
                    current_turn_content += ' ' + sentence[5]
                else:
                    current_turn_content += ' ' + sentence[5]
            this_conversation.append((current_party, current_turn_content))
            conversation_total.append(this_conversation)
        return conversation_total

    def _split_sentence_within_conversation_new(self, conversation_list):
        """
        按照说话顺序划分对话
        :param conversation_list: self.conversation_all
        :return: 为列表，每个元素为一句话，样例如下

        [
        ['00029c51f92e8f34250d6af329c9a8df', 'USERID_10003667', '0', '0', '0', '什么时候能发货啊'],
        ['00029c51f92e8f34250d6af329c9a8df', 'USERID_10003667', '0', '0', '0', '什么时候能发货啊'],
        ['00029c51f92e8f34250d6af329c9a8df', 'USERID_10003667', '0', '0', '0', '得等多久啊'],
        ['00029c51f92e8f34250d6af329c9a8df', 'USERID_10003667', '0', '0', '0', '我家狗狗等着吃饭呢'],
        ['00029c51f92e8f34250d6af329c9a8df', 'USERID_10003667', '0', '0', '0', '?'],
        ['00029c51f92e8f34250d6af329c9a8df', 'USERID_10003667', '1', '0', '0', '请您稍等一下，正在为您核实处理中哦~']
        ...
        ]

        """
        conversation_total = []
        for conversation in conversation_list:
            this_conversation = []
            for sentence in conversation:
                this_conversation.append(sentence)
            conversation_total.append(this_conversation)
        return conversation_total

    def _index_seperation(self, conversation_list):
        """
        得到每句话的隶属情况，用索引来代表句子
        :param conversation_list: 第二种x或者第三种当前的输出
        :return: 例如，[[0,1,2,3], [4,5,6,7,8], [9,10,11] ... ] 来表示0-3属于第一个对话，4-8属于第二个对话
        """
        output = []
        count = 0
        for conversation in conversation_list:
            current_conversation = []
            for sentence in conversation:
                current_conversation.append(count)
                count += 1
            output.append(current_conversation)
        return output

    def _samples2XY(self, embedding_matrix, my_merger):
        """根据embedding_matrix，和merger，将self.samples转换为等价的X和Y矩阵"""
        vector_list = []  # X
        label_list = []  # Y
        for i, sample in enumerate(self.samples):
            tmp = np.zeros((1, 300))
            # if self.clustering_index_label[sample[1]] == -1:
            #     continue
            for sentence_index in sample[0]:
                tmp += embedding_matrix[sentence_index, :]
            tmp = tmp / float(len(sample[0]))
            vector_list.append(tmp)

            # merge class
            original_label = self.clustering_index_label[sample[1]]
            if original_label in my_merger.mapping:
                original_label = my_merger.mapping[original_label]
            label_list.append(original_label)

        X = np.concatenate(vector_list)
        Y = np.array(label_list)

        return X, Y

    def samples2vector(self, a_represent):
        """
        根据represent，将samples的每一个x转换为一个特征矩阵
        :param a_represent:
        :return:
        """
        X = []
        Y = []

        for j, sample in enumerate(self.samples):
            sentence_list = [self.conversation_all_preprocessed_seperate_flaten[i][5] for i in sample[0]]
            sample_matrix = a_represent.word_embedding('sgns.merge.bigram', sentence_list)
            sample_vector = np.sum(sample_matrix, axis=0)
            sample_vector = sample_vector/float(sample_matrix.shape[0])
            X.append(sample_vector)
            Y.append(sample[1])
            print('Finishing %d' % j)

        return X, Y

    def prepare_search(self, a_preprocessor, a_sampler, a_represent):
        """
        集成的方法，直接将data对象运行所有进行搜索前的步骤一次性运行完毕，避免外界过多的涉及内部的方法
        这步之后，用户可以调用的内部属性有
        self.conversation_QAQAQ_preprocessed
        self.samples

        :return:
        """
        self.seperate_conversation()
        self.preprocess(a_preprocessor, format=3)
        self.generate_training_sample(a_sampler)
        # return self.samples2vector(a_represent)


def run():
    my_preprocessor = Preprocessor('stop_words.txt')
    my_sampler = Sampler()
    my_represent = Represent()

    a_data = Data('../data/chat.txt')
    a_data.prepare_search(my_preprocessor, my_sampler, my_represent)



def explore():
    a_data = Data('../data/chat.txt')
    a_data.seperate_conversation()
    print(a_data.conversation_QAQAQ_preprocessed[:5])


def generate_data(path):
    """
    Date: 2018-7-13
    一鹏要求保留原始的标点符号和所有的原始的表情什么的
    :return:
    """
    my_preprocessor = Preprocessor('cc')
    a_data = Data('../data/chat-20w.txt')
    a_data.seperate_conversation()
    a_data.preprocess(my_preprocessor, format=2)

    output = []
    for conversation in a_data.conversation_all_preprocessed_seperate:
        this_conversation = []
        for sentence in conversation:
            speaker = sentence[2]
            content = sentence[5]
            content = [restore_placeholder(token) for token in content]

            this_conversation.append([speaker, content])
        output.append(this_conversation)

    with open(path, 'w', encoding='utf-8') as f:
        for conversation in output:
            for sentence in conversation:
                f.write(str(sentence[0]) + ' ' + ' '.join(sentence[1]) + '\n')
            f.write('\n')

def generate_data_3(path):
    """
    Date: 2018-7-13
    一鹏要求保留原始的标点符号和所有的原始的表情什么的
    :return:
    """
    my_preprocessor = Preprocessor('stop_words.txt')
    # a_data = Data('../data/chat-20w.txt')
    a_data = Data('chat-30.txt')
    a_data.seperate_conversation()
    a_data.preprocess(my_preprocessor, format=3)

    with open(path, 'w') as f:
        # for s in a_data.conversation_all_preprocessed_seperate:
        for s in a_data.conversation_QAQAQ_preprocessed:
            for t in s:
                for x in t:
                    f.write(str(x) + '\n')
            f.write('\n')
    return

    output = []
    for conversation in a_data.conversation_QAQAQ_preprocessed:
        this_conversation = []
        for sentence in conversation:
            speaker = sentence[2]
            content = sentence[5]
            content = [restore_placeholder(token) for token in content]

            this_conversation.append([speaker, content])
        output.append(this_conversation)

    with open(path, 'w', encoding='utf-8') as f:
        for conversation in output:
            for sentence in conversation:
                f.write(str(sentence[0]) + ' ' + ' '.join(sentence[1]) + '\n')
            f.write('\n')

def restore_placeholder(token):
    if token == 'URL':
        return 'https://item.jd.com/1263959.html'
    if token == 'ORDER':
        return '[ORDERID_10005818]'
    if token == 'DATE':
        return '[日期x]'
    if token == 'TIME':
        return '[时间x]'
    if token == 'MONEY':
        return '[金额x]'
    if token == 'EMOJI':
        return '#E-s[数字x]'
    if token == 'SITE':
        return '[站点x]'
    if token == 'NUMBER':
        return '[数字x]'
    if token == 'LOCATION':
        return '[地址x]'
    if token == 'EMAIL':
        return '[邮箱x]'
    if token == 'NAME':
        return '[姓名x]'
    if token == 'PHONE':
        return '[电话x]'
    if token == 'PICTURE':
        return '[商品快照]'
    else:
        return token


if __name__ == '__main__':
    # run()
    generate_data_3('a.txt')
