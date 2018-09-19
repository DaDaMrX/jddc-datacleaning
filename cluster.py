import jieba
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans


input_file = './data/cat.txt'
output_file = './data/cat-cluster.txt'
cluster_file = './cluster2/cluster'

n_clusters = 200
n_init = 8

max_iter = 200
seed = 1
verbose = 2


def read_file():
    f = open(input_file)
    answers = []
    index = []
    for line in f:
        if line == '\n':
            continue
        # kind, sen = line.split()
        kind = line.split()[0]
        sen = line[2:]
        if kind == 'Conversation':
            index.append([])
            continue
        if kind == '1':
            answers.append(sen)
            index[-1].append(len(answers) - 1)
    f.close()
    return answers, index


def write_file(labels):
    fin = open(input_file)
    fout = open(output_file, 'w')

    conv_id = -1
    answer_id = -1
    for line in fin:
        if line == '\n':  # conversation ends
            fout.write(line)
            continue
        # kind, sen = line.split()
        kind = line.split()[0]
        sen = line[2:]
        if kind == 'Conversation':  # begin
            fout.write(line)
            conv_id += 1
            answer_id = -1
            continue
        if kind == '1':  # answer
            answer_id += 1
            t = index[conv_id]
            i = t[answer_id]
            fout.write('1 ' + str(labels[i]) + ' ' + sen)
        else:  # question
            fout.write('0 . ' + sen)

    fin.close()
    fout.close()


def write_cluster(labels, answers):
    clusters = [[] for _ in range(n_clusters)]
    for label, sen in zip(labels, answers):
        clusters[label].append(sen)

    total = [(i, len(clusters[i])) for i in range(n_clusters)]
    total.sort(key=lambda t: -t[1])

    with open(cluster_file + '.txt', 'w') as f:
        f.write('%7s %7s\n' % ('Cluster', 'Count'))
        for i, count in total:
            f.write('%7d %7d\n' % (i, count))

    for i in range(n_clusters):
        file = cluster_file + '-%d.txt' % i
        with open(file, 'w') as f:
            f.write('Cluster: %d\n' % i)
            f.write('Count  : %d\n' % len(clusters[i]))
            for sen in clusters[i]:
                f.write(sen)


if __name__ == '__main__':
    answers, index = read_file()

    # sentences to tf-idf
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
    tfidf_matrix = tfidf_vectorizer.fit_transform(answers)

    # k-means cluster
    kmeans = KMeans(n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init=n_init,
                    init='k-means++',
                    n_jobs=-1,
                    random_state=seed,
                    verbose=verbose)
    labels = kmeans.fit_predict(tfidf_matrix)

    write_file(labels)
    write_cluster(labels, answers)
