f = open('./data/chat-200w.txt')
corpus = []
conv = []
session_id = ''
speaker_id = ''
sent = ''
f.readline()
for line in f:
    line = line.strip().split()
    if line[0] != session_id:
        corpus.append(conv)
        session_id = line[0]
        conv = []
        speaker_id = line[2]
        sent = line[-1]
    elif line[2] != speaker_id:
        conv.append([int(speaker_id), sent])
        speaker_id = line[2]
        sent = line[-1]
    else:
        sent += ' ' + line[-1]
corpus.append(conv)
corpus.pop(0)
f.close()


f = open('./data/cat.txt', 'w')
for i, conv in enumerate(corpus):
    f.write('Conversation %d:\n' % i)
    for t, sent in conv:
        f.write(str(t) + ' ' + sent + '\n')
    f.write('\n')
f.close()
