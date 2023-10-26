import pickle
import pandas as pd
from itertools import compress
import argparse
from ast import literal_eval


def process_movielens_name(s: str) -> str:
    s = s[:-7]
    s = s.split(' (')[0]
    for pattern in [', The', ', A']:
        if s.endswith(pattern):
            s = pattern.split(', ')[1]+' ' + s.replace(pattern, '')
    return s


def parse_genres(genres_list: pd.DataFrame,
                 genres_names: list
                 ) -> list:
    return list(compress(genres_names, list(genres_list.values)))


def prepare_items(items_folder: str,
                  genres_names: list
                  ) -> (pd.DataFrame, dict, dict):
    items = pd.read_csv(f"{items_folder}/movies.data", sep='|', engine='python', encoding="latin-1")
    items.drop(labels=["releasedate", "videoreleasedate", "IMDbURL"], inplace=True, axis=1)

    # drop duplicates
    print(f"len items before: {len(items)}")
    print(items.duplicated(subset="movietitle").value_counts())
    items = items.drop_duplicates(subset="movietitle", keep='first')
    print(items.duplicated(subset="movietitle").value_counts())

    items['name'] = items.movietitle.map(process_movielens_name)
    items = items.drop_duplicates(subset="name", keep='first')
    items = items[items["movietitle"] != "unknown"]
    print(f"len items after: {len(items)}")

    items["genres"] = items.apply(lambda x: parse_genres(x[genres_names], genres_names), axis=1)
    item_id_to_name = items.set_index('movieId')['name'].to_dict()
    itemname_to_id = {v: k for k, v in item_id_to_name.items()}
    items.set_index('movieId', inplace=True)
    items.drop(labels=genres_names, inplace=True, axis=1)

    return items, item_id_to_name, itemname_to_id


def prepare_item_files(dir_items: str, out_dir: str):
    genres = ["unknown",
              "Action",
              "Adventure",
              "Animation",
              "Childrens",
              "Comedy",
              "Crime",
              "Documentary",
              "Drama",
              "Fantasy",
              "Film-Noir",
              "Horror",
              "Musical",
              "Mystery",
              "Romance",
              "Sci-Fi",
              "Thriller",
              "War",
              "Western"]

    df_items, itemid_to_name, itemname_to_id = prepare_items(dir_items, genres)
    df_items.to_csv(f"{out_dir}df_items.csv", index=True)
    with open(f"{out_dir}itemid_to_name.pkl", 'wb') as fp:
        pickle.dump(itemid_to_name, fp)
    with open(f"{out_dir}itemname_to_id.pkl", 'wb') as fp:
        pickle.dump(itemname_to_id, fp)
    print("Dataset prepared")


def prepare_genres_file(out_dir: str,
                        itemname_to_id: dict):
    df_items = pd.read_csv(f"{out_dir}df_items.csv")
    df_items.genres = df_items.genres.apply(literal_eval)
    out_string = ""
    for i, row in df_items.iterrows():
        for genre in row["genres"]:
            if row['name'] in itemname_to_id:
                out_string += f"{itemname_to_id[row['name']]}\t{genre}\n"
            else:
                print(row)

    with open(f"{out_dir}genres_file.txt", "w") as text_file:
        text_file.write(out_string)


def prepare_ratings(fold_path: str, item_dir: str, threshold: float):
    train_data = pd.read_csv(f"{fold_path}/train_data.csv", sep="\t", names=["userid", "itemid", "rating"])
    train_val_data = pd.read_csv(f"{fold_path}/train_val_data.csv", sep="\t", names=["userid", "itemid", "rating"])
    val_data = pd.read_csv(f"{fold_path}/val_data.csv", sep="\t", names=["userid", "itemid", "rating"])
    test_data = pd.read_csv(f"{fold_path}/test_data.csv", sep="\t", names=["userid", "itemid", "rating"])
    df_items = pd.read_csv(f"{item_dir}df_items.csv")
    items = pd.read_csv(f"{fold_path}/items.csv", names=["id"])

    print(f"len training before: {len(train_data)}")
    train_data = train_data[train_data["itemid"].isin(df_items.movieId.values)]
    print(f"len training after: {len(train_data)}")

    print(f"len train val before: {len(train_val_data)}")
    train_val_data = train_val_data[train_val_data["itemid"].isin(df_items.movieId.values)]
    print(f"len train val after: {len(train_val_data)}")

    print(f"len val before: {len(val_data)}")
    val_data = val_data[val_data["itemid"].isin(df_items.movieId.values)]
    print(f"len val after: {len(val_data)}")

    print(f"len test before: {len(test_data)}")
    test_data = test_data[test_data["itemid"].isin(df_items.movieId.values)]
    print(f"len test after: {len(test_data)}")

    print(f"len items before: {len(items)}")
    items = items[items["id"].isin(df_items.movieId.values)]
    print(f"len items after: {len(items)}")

    train_data.to_csv(f"{fold_path}/train_data.csv", header=False, index=False, sep="\t")
    train_val_data.to_csv(f"{fold_path}/train_val_data.csv", header=False, index=False, sep="\t")
    val_data.to_csv(f"{fold_path}/val_data.csv", header=False, index=False, sep="\t")
    test_data.to_csv(f"{fold_path}/test_data.csv", header=False, index=False, sep="\t")
    items.to_csv(f"{fold_path}/items.csv", header=False, index=False, sep="\t")

    # pos_train = train_data[train_data["rating"] >= threshold]
    # a = pos_train.groupby('userid')['itemid'].apply(list).to_frame()
    # a.rename(columns={"itemid": "pos_test"}, inplace=True)
    #
    # neg_train = train_data[train_data["rating"] < threshold]
    # b = neg_train.groupby('userid')['itemid'].apply(list).to_frame()
    # b.rename(columns={"itemid": "neg_test"}, inplace=True)
    #
    # pos_test = test_data[test_data["rating"] >= threshold]
    # c = pos_test.groupby('userid')['itemid'].apply(list).to_frame()
    # c.rename(columns={"itemid": "pos_test"}, inplace=True)
    #
    # neg_test = test_data[test_data["rating"] < threshold]
    # d = neg_test.groupby('userid')['itemid'].apply(list).to_frame()
    # d.rename(columns={"itemid": "neg_test"}, inplace=True)
    #
    # res = pd.concat([a, b, c, d], axis=1)
    print("Ratings prepared")


def main(args):

    dir_movies = "/home/diego/chat-rerank/dataset/ml-100k/"
    output_dir = "/home/diego/chat-rerank/experiments/ml-100k/"
    relevancy_threshold = 4.0
    prepare_item_files(dir_movies, output_dir)

    filename = f"{args.experimentspath}itemname_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemname_to_id = pickle.load(fp)
    prepare_genres_file(args.experimentspath, itemname_to_id)
    for fold in range(1):
        prepare_ratings(f"{output_dir}fold_{fold}", output_dir, relevancy_threshold)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experimentspath",
        default=False,
        type=str,
        required=True,
        help="The path to the dataset"
    )

    main(parser.parse_args())

