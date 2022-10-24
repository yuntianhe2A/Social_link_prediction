from collections import defaultdict
from math import sqrt


def getNeighbors(training_data):
    neighbors = defaultdict(set)
    for pair in training_data:
        neighbors[pair[0]].add(pair[1])
        neighbors[pair[1]].add(pair[0])
    return neighbors


def readTestData(filename='data/test.txt'):
    testdata = []
    nodes = []
    with open(filename) as f:
        for line in f:
            a1, a2 = line.strip().split()
            testdata.append((a1, a2))
            if a1 not in nodes:
                nodes.append(a1)
            if a2 not in nodes:
                nodes.append(a2)
    return testdata, nodes


def readValidationData(file_pos, file_neg):
    validation_data_pos = []
    validation_data_neg = []
    validationData = []
    nodes = []
    with open(file_pos) as f:
        for line in f:
            a1, a2 = line.strip().split()
            validationData.append((a1, a2))
            validation_data_pos.append((a1, a2))
    with open(file_neg) as f:
        for line in f:
            a1, a2 = line.strip().split()
            validationData.append((a1, a2))
            validation_data_neg.append((a1, a2))
    for node in nodes:
        validationData.append((node, node))
    return validationData, validation_data_pos, validation_data_neg


def readTrainingData(filename):
    trainingData = []
    nodes = []
    with open(filename) as f:
        for line in f:
            a1, a2 = line.strip().split()
            trainingData.append((a1, a2))
            if a1 not in nodes:
                nodes.append(a1)
            if a2 not in nodes:
                nodes.append(a2)
    for node in nodes:
        trainingData.append((node, node))
    return trainingData, nodes


def computeProximityScore(author1, author2, neighbors, algorithm='jaccard'):
    n1=neighbors[author1]
    n2 = neighbors[author2]
    if algorithm == 'jaccard':
        score = len(n1.intersection(n2))
        score = float(score) / len(n1.union(n2))
    elif algorithm == 'preferential':
        score = len(n1) * len(n2)
    elif algorithm == 'cosine':
        score1 = len(n1.intersection(n2))
        score2 = float(sqrt(len(n1) * len(n2)))
        score=float(score1) / score2
    return score


def get_validation_top_100(filename='data/training.txt'):
    training_data, nodes = readTrainingData(filename)
    neighbors = getNeighbors(training_data)
    validationData, validation_data_pos, validation_data_neg = readValidationData('data/val_positive.txt',
                                                                                  'data/val_negative.txt')
    linkScores = {}
    for pair in validationData:
        linkScores[(pair[0], pair[1])] = computeProximityScore(pair[0], pair[1], neighbors, 'jaccard')
    top100_links = sorted(linkScores.items(), key=lambda x: x[1], reverse=True)[:100]
    print(top100_links)
    return top100_links


def get_test_top_100(filename='data/training.txt'):
    linkScores = {}
    test_data, _ = readTestData('data/test.txt')
    training_data, nodes = readTrainingData(filename)
    neighbors = getNeighbors(training_data)
    for pair in test_data:
        linkScores[(pair[0], pair[1])] = computeProximityScore(pair[0], pair[1], neighbors)
    top100_links = sorted(linkScores.items(), key=lambda x: x[1], reverse=True)[:100]
    return top100_links


def evaluation(topList, groundTruth):
    count = 0
    for link in topList:
        if link[0] in groundTruth:
            count += 1
    print(float(count) / len(groundTruth) * 100, '%')

def write_test_result(top100):
    with open('test_prediction.txt', 'w') as out:
        for link_score in top100:
            a = link_score[0]
            node1, node2 = a
            predicted_link = node1 + " " + node2
            out.write(predicted_link)
            out.write('\n')

def main():
    # top100_test = get_test_top_100()
    # print(top100_test)
    # write_test_result(top100_test)
    top100_validation = get_validation_top_100()
    _, groundTruth, _ = readValidationData('data/val_positive.txt', 'data/val_negative.txt')
    evaluation(top100_validation, groundTruth)


if __name__ == "__main__":
    main()
