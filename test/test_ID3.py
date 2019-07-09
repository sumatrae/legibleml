import math
import numpy as np
from easyML.descion_tree.ID3 import ID3


def test_calc_shannon_entropy():
    dataset = np.array([[1, 2, 3],
                        [4, 5, 6]])
    labels = [0, 1]

    algo = ID3()
    shannon_entropy = algo._calc_shannon_entropy(dataset, labels)
    assert shannon_entropy == -0.5 * math.log(1 / 2) * 2


def test_calc_shannon_entropy_length_error():
    dataset = np.array([[1, 2, 3],
                        [4, 5, 6]])
    labels = np.array([0, 0])

    algo = ID3()
    shannon_entropy = algo._calc_shannon_entropy(dataset, labels)
    assert shannon_entropy == -1 * math.log(1)


def test_calc_shannon_entropy_03(capsys):
    dataset = np.array([[1, 2, 3],
                        [4, 5, 6]])
    labels = np.array([0])

    algo = ID3()
    try:
        shannon_entropy = algo._calc_shannon_entropy(dataset, labels)
    except Exception as e:
        captured = capsys.readouterr()
        assert captured.out == "dataset length doesn't equal with labels\n"
        print(e)


def test_split_dateset():
    dataset = np.array([[1, 1, 0, 1],
                        [0, 0, 1, 1],
                        [1, 1, 1, 0]])

    algo = ID3()
    new_dataset = algo._split_dateset(dataset, 0, 0)
    assert (new_dataset == np.array([[0, 1, 1]])).all()

    new_dataset = algo._split_dateset(dataset, 0, 1)
    assert (new_dataset == np.array([[1, 0, 1],
                                     [1, 1, 0]])).all()

    new_dataset = algo._split_dateset(dataset, 2, 1)
    assert (new_dataset == np.array([[0, 0, 1],
                                     [1, 1, 0]])).all()

    new_dataset = algo._split_dateset(dataset, 3, 0)
    assert (new_dataset == np.array([[1, 1, 1]])).all()


def test_split_label():
    dataset = np.array([[1, 1, 0, 1],
                        [0, 0, 1, 1],
                        [1, 1, 1, 0],
                        [1, 1, 0, 1],
                        [0, 0, 1, 1],
                        [1, 1, 1, 0]])

    labels = np.array([1, 1, 1, 0, 0, 1])

    algo = ID3()

    new_label = algo._split_labels(dataset, labels, 1, 1)
    assert (new_label == np.array([1, 1, 0, 1])).all()

    new_label = algo._split_labels(dataset, labels, 1, 0)
    assert (new_label == np.array([1, 0])).all()

    new_label = algo._split_labels(dataset, labels, 3, 1)
    assert (new_label == np.array([1, 1, 0, 0])).all()


def test_conditional_entropy():
    dataset = np.array([[1, 2, 3],
                        [4, 5, 6]])
    labels = np.array([0, 1])

    algo = ID3()
    conditional_entropy = algo._calc_conditional_entropy(dataset, labels, 0)
    assert conditional_entropy == 0
