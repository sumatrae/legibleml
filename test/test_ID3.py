import math
import numpy as np
from easyML.descion_tree.ID3 import ID3

def test_calc_shannon_entropy_01():
    dataset = np.array([[1,2,3],
                        [4,5,6]])
    labels =  [0,1]

    algo = ID3()
    shannon_entropy = algo.calc_shannon_entropy(dataset,labels)
    assert shannon_entropy == -0.5*math.log(1/2)*2


def test_calc_shannon_entropy_02():
    dataset = np.array([[1, 2, 3],
                        [4, 5, 6]])
    labels = np.array([0, 0])

    algo = ID3()
    shannon_entropy = algo.calc_shannon_entropy(dataset, labels)
    assert shannon_entropy == -1 * math.log(1)

def test_calc_shannon_entropy_03(capsys):
    dataset = np.array([[1, 2, 3],
                        [4, 5, 6]])
    labels = np.array([0])

    algo = ID3()
    try:
        shannon_entropy = algo.calc_shannon_entropy(dataset, labels)
    except Exception as e:
        captured = capsys.readouterr()
        assert captured.out == "dataset length doesn't equal with labels\n"
        print(e)

