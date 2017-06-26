from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import make_sampling_table
from keras.preprocessing.sequence import skipgrams


def test_pad_sequences():
    a = [[1], [1, 2], [1, 2, 3]]

    # test padding
    b = pad_sequences(a, maxlen=3, padding='pre')
    print(b, ", shape: " , b.shape)
    b = pad_sequences(a, maxlen=3, padding='post')
    print(b, ", shape: " , b.shape)

    # test truncating
    b = pad_sequences(a, maxlen=2, truncating='pre')
    print(b, ", shape: " , b.shape)
    b = pad_sequences(a, maxlen=2, truncating='post')
    print(b, ", shape: " , b.shape)

    # test value
    b = pad_sequences(a, maxlen=3, value=1)
    print(b, ", shape: " , b.shape)

test_pad_sequences()
