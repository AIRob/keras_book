from keras.preprocessing.sequence import make_sampling_table


def test_make_sampling_table():
        a = make_sampling_table(5)
        print(a)

test_make_sampling_table()
