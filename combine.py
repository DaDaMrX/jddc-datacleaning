from data import Data, Preprocessor
import sys


input_file = './data/chat-200w.txt'
output_file = './data/combine.txt'

stop_words_file = './data/stop_words.txt'
num = None


if __name__ == '__main__':
    data_loader = Data(input_file, num)
    data_loader.seperate_conversation()
    preprocessor = Preprocessor(stop_words_file)
    data_loader.preprocess(preprocessor, format=3)
    data = data_loader.conversation_QAQAQ_preprocessed

    f = open(output_file, 'w')
    sys.stdout = f

    for i, conv in enumerate(data):
        print('Conversation %d:' % i)
        for sen in conv:
            if not sen[1]:
                sen[1] = ['EMOJI']
            print(str(sen[0]) + ' ' + ''.join(sen[1]))
        print()

    f.close()
