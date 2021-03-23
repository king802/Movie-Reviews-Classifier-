from text_classification_helpers import *
import string

if __name__ == '__main__':

    dev_path = 'textcat/dev'
    test_path = 'textcat/test'
    train_path = 'textcat/train'
    dev_data = read_data(dev_path)
    # test_data = read_data(test_path, False)
    train_data = read_data(train_path)
    vocab = create_vocabulary(train_data)
    train_df = make_data_frame(train_data, vocab)
    dev_df = make_data_frame(dev_data, vocab)

    model = make_probability_distribution(train_df)

    print(evaluate(dev_data, model))

    # test_data[0][1]



