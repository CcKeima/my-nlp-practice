import nltk
import random
def gender_features(word):
    return {'last_letter': word[-1]}

labeled_names = ([(name, 'male') for name in ['k', 'o', 'r']] + 
                 [(name, 'female') for name in ['a', 'e', 'i']])
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[2:], featuresets[:2]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(gender_features('k')))