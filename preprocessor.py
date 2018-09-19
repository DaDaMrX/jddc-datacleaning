# coding=utf-8
# author=Dnsk


import re

import jieba


"""预处理所使用的正则表达式"""
URL_REGULAR_EXPRESSION = """http[s]?://[a-zA-Z0-9|\.|/]+"""  # 匹配http或者https开头的url
URL_ANOTHER_REGULAR_EXPRESSION = """http[s]?://[a-zA-Z0-9\./-]*\[链接x\]"""
ORDERID_REGULAR_EXPRESSION = """\[ORDERID_[0-9]+\]"""  # 匹配[ORDERID_10002026]形式的order id
EMOJI_REGULAR_EXPRESSION = """#E-[a-z|0-9]+\[数字x\]|~O\(∩_∩\)O/~"""  # 匹配#E-s[数字x]形式和颜文字形式的表情
DATE_REGULAR_EXPRESSION = """\[日期x\]"""
TIME_REGULAR_EXPRESSION = """\[时间x\]"""
MONEY_REGULAR_EXPRESSION = """\[金额x\]"""
SITE_REGULAR_EXPRESSION = """\[站点x\]"""
NUMBER_REGULAR_EXPRESSION = """\[数字x\]"""
LOCATION_REGULAR_EXPRESSION = """\[地址x\]"""
NAME_REGULAR_EXPRESSION = """\[姓名x\]"""
MAIL_REGULAR_EXPRESSION = """\[邮箱x\]"""
PHONE_REGULAR_EXPRESSION = """\[电话x\]"""
PICTURE_REGULAR_EXPRESSION = """\[商品快照\]"""
SPLIT_SYN_REGULAR_EXPRESSION = """<s>"""
MULTIPLE_SPACES_REGULAR_EXPRESSION = """\s+"""


class Preprocessor:
    def __init__(self, stop_word_path, char_lv=False):
        self._stop_word_list = []

        self.char_lv = char_lv
        # load stopword

        try:
            with open(stop_word_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self._stop_word_list.append(line.strip())
        except IOError:
            print('Notice: there is no stop word list.')

    def re_sentence(self, sentence_to_preprocess):
        """
        Get a sentnece, return a sentence proprocessed
        """
        s1 = re.sub(URL_ANOTHER_REGULAR_EXPRESSION, ' URL ', sentence_to_preprocess)
        s2 = re.sub('&nbsp', '', s1)
        s3 = re.sub(ORDERID_REGULAR_EXPRESSION, ' ORDER ', s2)
        s4 = re.sub(DATE_REGULAR_EXPRESSION, ' DATE ', s3)
        s5 = re.sub(TIME_REGULAR_EXPRESSION, ' TIME ', s4)
        s6 = re.sub(MONEY_REGULAR_EXPRESSION, ' MONEY ', s5)
        s7 = re.sub(EMOJI_REGULAR_EXPRESSION, ' EMOJI ', s6)
        s8 = re.sub(SITE_REGULAR_EXPRESSION, ' SITE ', s7)
        s9 = re.sub(NUMBER_REGULAR_EXPRESSION, ' NUMBER ', s8)
        s10 = re.sub(LOCATION_REGULAR_EXPRESSION, ' LOCATION ', s9)
        s11 = re.sub(MAIL_REGULAR_EXPRESSION, ' EMAIL ', s10)
        s12 = re.sub(NAME_REGULAR_EXPRESSION, ' NAME ', s11)
        s13 = re.sub(PHONE_REGULAR_EXPRESSION, ' PHONE ', s12)
        s14 = re.sub(PICTURE_REGULAR_EXPRESSION, ' PICTURE ', s13)
        s15 = re.sub(URL_REGULAR_EXPRESSION, ' URL ', s14)
        s16 = re.sub(SPLIT_SYN_REGULAR_EXPRESSION, " ", s15)
        s17 = re.sub(MULTIPLE_SPACES_REGULAR_EXPRESSION, ' ', s16)
        return s17

    def remove_stopword(self, sentence_to_process):
        """
        to remove stopword
        :return: a list of words for each sentence
        """
        tmp = [token for token in jieba.lcut(sentence_to_process) if token not in self._stop_word_list]
        return [token for token in tmp if token != ' ']

    def preprocess_pipeline(self, sentence_to_preprocess):
        """
        一步调用前两个函数，返回正则匹配和去除停用词之后的句子，以分好词的列表的形式，如：
        ['我', '爱', '北京']
        """
        if not self.char_lv:
            sentence = self.re_sentence(sentence_to_preprocess)
            return self.remove_stopword(sentence)
        if self.char_lv:
            return sentence_to_preprocess.strip()


def test():
    my_preprocessor = Preprocessor('stop_words.txt')
    # print(my_preprocessor.preprocess_sentence('[邮箱x] 这是一个测试'))
    print(my_preprocessor.re_sentence('[邮箱x] 这是一个测试'))


if __name__ == '__main__':
    test()
