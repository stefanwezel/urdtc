import argparse
import numpy as np
import scipy.io as sio

from os.path import join

def create_train_test_split(attr_mat, imfeatures_mat,
                            train_indices_file, test_indices_file):
    att_data = sio.loadmat(attr_mat)
    att_data['att'] = att_data['att'] / att_data['att'].max()

    image_data = sio.loadmat(imfeatures_mat)
    labels = image_data['labels']

    classes = set()
    idx_per_cls = {}
    for idx, label in enumerate(labels):
        label = label[0]
        classes.add(label)
        if label not in idx_per_cls:
            idx_per_cls[label] = []
        idx_per_cls[label].append(idx)

    print("Num classes: {}".format(len(classes)))
    cls_list = sorted(list(classes))

    all_train_indices = []
    all_test_indices = []
    test_size = 0.2
    for cls in cls_list:
        indices = np.array(idx_per_cls[cls])
        cnt = len(indices)
        print(cls, cnt)
        train_test = np.ones(cnt)
        train_test[:int(np.round(cnt * test_size))] = 0
        train_test = np.random.permutation(train_test)
        train_test = train_test.astype(bool)
        train_indices = indices[train_test]
        test_indices = indices[~train_test]

        all_train_indices.append(train_indices)
        all_test_indices.append(test_indices)

    train_indices = np.concatenate(all_train_indices, axis=0)
    test_indices = np.concatenate(all_test_indices, axis=0)

    np.savetxt(train_indices_file, train_indices)
    np.savetxt(test_indices_file, test_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess APY Dataset.')
    parser.add_argument('-data_path', default='./awa', type=str,
                        help='Path to the APY dataset.')
    parser.add_argument('-seed', default=1, type=int,
                        help='Seed for split generation.')
    args = parser.parse_args()

    np.random.seed(args.seed)

    attr_mat = join(args.data_path, 'att_splits.mat')
    imfeatures_mat = join(args.data_path, 'res101.mat')
    train_indices_file = join(args.data_path, 'train_indices.npy')
    test_indices_file = join(args.data_path, 'test_indices.npy')

    create_train_test_split(attr_mat, imfeatures_mat,
                            train_indices_file, test_indices_file)
