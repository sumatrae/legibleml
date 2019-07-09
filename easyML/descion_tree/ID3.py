import numpy as np
import math


class ID3:
    def __init__(self):
        pass

    def _calc_shannon_entropy(self, dataset, labels):
        '''
        calculate shannon entropy form data set and labels
        :param dataset: numpy.array, data set
        :param labels: list,tuple,numpy.array samples' labels
        :return: float, shannon entropy
        '''
        if dataset.shape[0] != len(labels):
            print("dataset length doesn't equal with labels")
            raise Exception

        # 统计数据集D中的K个类及每个类的的数量|Ck|
        if not isinstance(labels, np.ndarray):
            labels_array = np.array(labels)
        else:
            labels_array = labels

        unique_labels = np.unique(labels)
        labels_counts = np.array([0] * len(unique_labels))
        for i, label in enumerate(unique_labels):
            labels_counts[i] = labels_array[labels_array == label].size

        # 计算每个类的比率|Ck|/|D|及训练集的经验熵H(D)
        sample_num = len(dataset)
        prob = labels_counts / sample_num
        shannon_entropy = - (prob * np.log(prob)).sum()

        return shannon_entropy

    def _split_dateset(self, dataset, feature_index, value):
        match_sample_index = np.where(dataset[:, feature_index] == value)
        match_samples = dataset[match_sample_index]
        subdataset = np.delete(match_samples, feature_index, 1)
        return subdataset

    def _split_labels(self, dataset, labels, feature_index, vlaue):
        match_sample_index = np.where(dataset[:, feature_index] == vlaue)
        subdataset_labels = labels[match_sample_index]
        return subdataset_labels

    def _calc_conditional_entropy(self, dataset, labels, feature_index):
        '''
        calculate conditional entropy H(D|A), D is dataset, A is the feature
        :return: float, conditional entropy H(D|A)
        '''
        feature_values = dataset[:, feature_index]
        unique_feature_values = np.unique(feature_values)

        new_conditional_entropy = 0.0
        for value in unique_feature_values:
            subdataset = self._split_dateset(dataset, feature_index, value)
            subdataset_labels = self._split_labels(dataset, labels, feature_index, value)
            probability = subdataset.shape[0] / dataset.shape[0]
            new_shannon_entropy = self._calc_shannon_entropy(subdataset, subdataset_labels)
            new_conditional_entropy += probability * new_shannon_entropy

        return new_conditional_entropy

    def _calc_entropy_gain(self, conditional_entropy, shannon_entropy):
        return conditional_entropy - shannon_entropy

    def _choose_best_feature(self, dataset, labels):
        '''
        choose out base feature from current dataset
        :param dataset: numpy.array
        :return: int, best feature index
        '''
        feature_nums = dataset.shape[1]
        shannon_entropy = self._calc_shannon_entropy(dataset, labels)

        best_gain = 0.0
        best_feature = 0
        for i in range(feature_nums):
            conditional_entropy = self._calc_conditional_entropy(dataset, labels, i)
            entropy_gain = self._calc_entropy_gain(conditional_entropy, shannon_entropy)

            if entropy_gain > best_gain:
                best_gain = entropy_gain
                best_feature = i

        return best_feature

    def _find_max_class(self, labels):
        '''
        find out max class in the labels
        :param labels: numpy.array
        :return: max count label
        '''
        unique_labels = np.unique(labels)[-1]

        max_class = unique_labels[0]
        max_count = 0
        for label in unique_labels:
            label_num = labels[labels == label].size
            if label_num > max_count:
                max_class = label
                max_count = label_num

        return max_class

    def _create_tree(self, dataset, labels):
        if np.unique(labels).size == 1:
            return labels[0]

        if dataset.shape[1] == 1:
            return self._find_max_class(labels)

        best_feature_index = self._choose_best_feature(dataset, labels)
        tree = {best_feature_index: {}}

        feature_values = dataset[:, best_feature_index]
        unique_feature_values = np.unique(feature_values)

        for feature_value in unique_feature_values:
            sub_dataset = self._split_dateset(dataset, best_feature_index, feature_value)
            sub_dataset_label = self._split_labels(dataset, labels, best_feature_index, feature_value)
            tree[best_feature_index][feature_value] = self._create_tree(sub_dataset, sub_dataset_label)

        return tree

    def fit(self, dataset, labels):
        self.tree_ = self._create_tree(dataset, labels)
        return self

    def _classify(self, tree, dataset):
        feature_index = list(tree)[0]
        sub_tree = self.tree_[feature_index]
        labels = self._classify(sub_tree, dataset)
        return labels

    def predict(self, dataset):
        return self._classify(self.tree_, dataset)
