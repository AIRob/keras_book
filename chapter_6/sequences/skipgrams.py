from keras.preprocessing.sequence import skipgrams


def my_test_skipgrams():
    couples, labels = skipgrams([0,1,2,3], vocabulary_size=4)
    print("couples: ", couples)
    print("labels: ", labels)

my_test_skipgrams()
