# coding=utf-8
# author=Dnsk

class Sampler:
    def __init__(self, QAQAQ_order):
        self.QAQAQ_order = QAQAQ_order
        self.qualified_Q = int(self.QAQAQ_order/2) + 1
        self.qualified_A = int(self.QAQAQ_order/2)

    def sampling(self, speaker_seperation):
        """
        主要产生样本的代码，运行过程中会调用generate_sample
        返回的格式：
        列表，每个元素为一个元组（tuple）
        每个元组包含两个元素，第一个为x的index列表，第二个为y的index列表
        """
        samples = []
        start_of_current_conversation = 0  # pivot of current conversation
        for conversation in speaker_seperation:
            current_length = len(conversation)
            for i, current_sentence_speaker in enumerate(conversation):
                if current_sentence_speaker == 1:
                    samples.append(self.generate_sample(conversation, i, start_of_current_conversation))
                else:
                    continue
            start_of_current_conversation += current_length
        return samples

    def generate_sample(self, conversation, anchor_A, offset):
        """
        从给定A的位置(anchor_A)往上追溯，返回一个合适的sample，用全局的index组成的列表表示
        如果当前位置找不到合适的，返回None
        目前实现，只要目标回答之前能找到至少3Q2A即为合格，返回

        2018-7-13改动：当A前面是连续的A时，跳过。
        即，[0,1,0,1,0,1,1,1] 对于最后三个1，其对应的x均为[0,1,2,3,4]
        返回应为[([0,1,2,3,4], 5), ([0,1,2,3,4], 6), ([0,1,2,3,4], 7)]
        """
        x = []
        try:
            num_Q = 0
            num_A = 0
            if_there_is_Q = 0
            for current_speaker_index in range(anchor_A - 1, -1, -1):
                if conversation[current_speaker_index] == 1 and if_there_is_Q == 1:
                    num_A += 1
                    x.append(current_speaker_index + offset)
                if conversation[current_speaker_index] == 0:
                    num_Q += 1
                    if_there_is_Q = 1
                    x.append(current_speaker_index + offset)
                if num_Q >= self.qualified_Q and num_A >= self.qualified_A:
                    return (x[::-1], anchor_A + offset)
        except IndexError:
            return None


def test():
    a_sampler = Sampler()
    print(a_sampler.sampling([[1,0,1,0,1,0,1,0,1,0], [0,1,0,1,0,1,0,1], [0,1,0,1,0,1]]))


if __name__ == '__main__':
    test()

