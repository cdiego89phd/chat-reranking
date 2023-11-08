import random
import pandas as pd
from os import path, makedirs
import numpy as np


__author__ = 'dcarraro'


def random_split_by_user(df, frac, state):
    """Split the dataset in train and test set in a user way
    frac is the proportion of ratings hold by the test set
    """

    df_test = df.groupby("user_id").sample(frac=frac, random_state=state)
    df_train = df.drop(df_test.index)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test


if __name__ == "__main__":

    dataset_path = "/home/diego/chat-reranking/dataset/goodreads/"
    output_folder = "/home/diego/chat-reranking/experiments/goodreads/"
    dataset_name = "ratings12M.csv"

    folds = 1
    testset_prop = 0.2
    sample_test_users = 500
    produce_validation = True
    valset_prop = 0.2
    base_seed = 555

    df_ratings = pd.read_csv(f"{dataset_path}{dataset_name}", sep=",")
    # del df_ratings["is_pos"]
    df_ratings = df_ratings.sample(frac=1, random_state=111)  # shuffle the data
    print(len(df_ratings))

    for fold in range(folds):

        np.random.seed(base_seed)
        base_seed += 7

        # split the dataset in train/test (user-based split)
        trainset, testset = random_split_by_user(df_ratings, testset_prop, state=base_seed)

        # save the splits...
        out_dir = output_folder + 'fold_' + str(fold) + '/'
        if not path.exists(out_dir):
            makedirs(out_dir)

        # save users and items in a file
        out_file = open(out_dir + 'users.csv', 'w')
        for record in list(df_ratings['user_id'].unique()):
            out_file.write("%s\n" % record)
        out_file.close()
        out_file = open(out_dir + 'items.csv', 'w')
        for record in list(df_ratings['item_id'].unique()):
            out_file.write("%s\n" % record)
        out_file.close()

        trainset.to_csv(out_dir + 'train_data.csv', index=False, sep='\t', header=False)
        testset.to_csv(out_dir + 'test_data.csv', index=False, sep='\t', header=False)
        print("There are %d unique users in training set and %d unique users in the test set" % \
              (len(pd.unique(trainset['user_id'])), len(pd.unique(testset['user_id']))))
        print("There are %d ratings in training set and %d ratings in the test set" % \
              (len(trainset), len(testset)))

        # sample users for targeting recommendations
        # we also sample their test data (a subsample of the test data)
        if sample_test_users > 0:
            test_users = random.sample(list(df_ratings['user_id'].unique()), sample_test_users)
            out_file = open(out_dir + 'sample_test_users.csv', 'w')
            for record in list(test_users):
                out_file.write(f"{record}\n")
            out_file.close()
            df_sample_test_data = testset[testset["user_id"].isin(test_users)].copy()
            df_sample_test_data.to_csv(out_dir + 'sample_test_data.csv', index=False, sep='\t', header=False)
            print("There are %d ratings in the sample test set and %d users " % \
                  (len(df_sample_test_data), len(test_users)))

        # produce validation data
        if produce_validation:
            train_val, val = random_split_by_user(trainset, valset_prop, state=base_seed)

            train_val.to_csv(out_dir + 'train_val_data.csv', index=False, sep='\t', header=False)
            val.to_csv(out_dir + 'val_data.csv', index=False, sep='\t', header=False)

            print("There are %d unique users in train_val set and %d unique users in the val set" % \
                  (len(pd.unique(train_val['user_id'])), len(pd.unique(val['user_id']))))

            print("There are %d ratings in train_val set and %d ratings in the val set" % \
                  (len(train_val), len(val)))

