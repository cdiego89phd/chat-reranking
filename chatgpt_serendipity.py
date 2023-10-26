import pickle
import argparse
import datetime
import json
import openai
import pandas as pd
import time
import random


def load_helper_dicts(data_folder: str
                      ) -> (dict, dict):

    filename = f"{data_folder}itemid_to_name.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemid_to_name = pickle.load(fp)
    filename = f"{data_folder}itemname_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemname_to_id = pickle.load(fp)

    return itemid_to_name, itemname_to_id


def convert_dataframe(recs: pd.DataFrame,
                      baseline_name: str,
                      top_n: int,
                      top_m: int
                      ) -> pd.DataFrame:

    usrs = list(recs["userid"].unique())
    b_name = baseline_name.split("-")[0]
    baseline_column = [b_name for i in range(len(usrs))]
    top_n_column = [top_n for i in range(len(usrs))]
    top_m_column = [top_m for i in range(len(usrs))]
    recs_column = [recs[recs["userid"] == user]["itemid"].values for user in usrs]

    d = {"userid": usrs,
         "baseline": baseline_column,
         "top_m": top_m_column,
         "top_n": top_n_column,
         "recs": recs_column
         }

    return pd.DataFrame(data=d)


def load_prompt_template(promptpath: str,
                         prompt_id: str
                         ) -> str:

    f = open(f"{promptpath}templates.json")
    prompt_template = json.load(f)["templates"]
    f.close()
    prompt_template = {i["id"]: i["text"] for i in prompt_template}

    return prompt_template[prompt_id]


def build_prompts(train_data: pd.DataFrame,
                  recs: pd.DataFrame,
                  top_p: int,
                  top_m: int,
                  top_n: int,
                  itemid_to_name: dict,
                  template: str
                  ) -> list:

    # add the output format to the template
    template_prompt = template
    template_prompt = template_prompt.replace("<top_p>", str(top_p))
    template_prompt = template_prompt.replace("<top_m>", str(top_m))
    template_prompt = template_prompt.replace("<top_n>", str(top_n))

    users = recs.userid.values
    prompts = []
    for user in users:
        # randomly select user profile to include in the prompt
        user_train = list(train_data[train_data["userid"] == user]["itemid"].values)
        random.seed(5)
        top_p = min(top_p, len(user_train))
        user_train = random.sample(user_train, k=top_p)

        # add user profile to the prompt
        train_str = "\n"
        for item in user_train:
            train_str += f"{itemid_to_name[item]}\n"
        user_prompt = f"***{train_str}***\n"

        # select recommendations
        user_recs = recs[recs["userid"] == user]["recs"].values[0]

        # add the baseline recommendations
        items_str = "\n"
        for i, item in enumerate(user_recs, start=1):
            items_str += f"{i}. {itemid_to_name[item]}\n"

        prompts.append(template_prompt + user_prompt + f"\n```{items_str}```")

    return prompts


def get_response_chatgpt(prompt: str
                         ) -> str:
    model = "gpt-3.5-turbo"
    messages = [{"role": "user", "content": prompt}]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        output = response.choices[0].message["content"]
    except Exception as e:
        output = str(e)

    return output


def parse_response(output: str,
                   itemname_to_id: dict
                   ) -> list:

    # parse the response as a json
    try:
        res_dict = json.loads(output)
        assert "movie_name" in res_dict
    except Exception as e:
        print(e)
        print("problem parsing json response!")
        return None

    reranked_recs = []
    for item_name in res_dict["movie_name"]:
        try:
            reranked_recs.append(itemname_to_id[item_name])
        except Exception as e:
            print(e)
            print("problem parsing reranked entry!")
            reranked_recs.append(None)
    return reranked_recs


def query_chatgpt(prompts: list,
                  itemname_to_id: dict,
                  ) -> (list, list):

    raw_responses = []
    reranked_recs = []
    for i, prompt in enumerate(prompts, start=1):
        response = get_response_chatgpt(prompt)
        raw_responses.append(response)

        new_rank = parse_response(response, itemname_to_id)
        reranked_recs.append(new_rank)

        time.sleep(25)
        print(f"Done {i}/{len(prompts)} users! {datetime.datetime.now()}")
    return raw_responses, reranked_recs


def main(args):

    # load helper dictionaries
    itemid_to_name, itemname_to_id = load_helper_dicts(args.datasetpath)
    print(f"{datetime.datetime.now()} -- Helpers loaded!")

    # load prompt template (from json)
    prompt_template = load_prompt_template(args.promptpath, args.prompt_id)
    print(f"{datetime.datetime.now()} -- Prompt template loaded!")

    # retrieve train data
    training_data = pd.read_csv(f"{args.foldpath}train_data.csv", names=["userid", "itemid", "rating"], sep="\t")

    # load baseline recommendations
    recs = pd.read_csv(f"{args.datasetpath}/recs/baselines/{args.baseline_recs}", sep="\t",
                       names=["userid", "itemid", "rating"])
    recs.drop(labels="rating", axis=1, inplace=True)
    print(f"{datetime.datetime.now()} -- Baseline recommendations loaded!")

    if args.debug_mode:
        debug_usrs = recs["userid"].unique()[:5]
        recs = recs[recs["userid"].isin(debug_usrs)].copy()
        training_data = training_data[training_data["userid"].isin(debug_usrs)].copy()

    # trim recommendations
    recs = recs.groupby('userid').head(args.rerank_top_m).reset_index(drop=True)
    recs = convert_dataframe(recs, args.baseline_recs, args.top_n, args.rerank_top_m)
    print(f"{datetime.datetime.now()} -- Recommendations trimmed!")

    # craft prompts from template and top-m baseline recs
    prompts = build_prompts(training_data, recs, args.top_p, args.rerank_top_m,
                            args.top_n, itemid_to_name, prompt_template)
    recs["prompt"] = prompts
    recs["top_p"] = [args.top_p for i in range(len(prompts))]
    print(f"{datetime.datetime.now()} -- Prompts crafted!")

    # query chatgpt
    openai.api_key = args.openai_key
    raw_gpt_outputs, reranked_recs = query_chatgpt(prompts, itemname_to_id)
    recs["raw_gpt_outputs"] = raw_gpt_outputs
    recs["reranked_recs"] = reranked_recs
    print(f"{datetime.datetime.now()} -- Done with ChatGPT!")

    # save it into file
    out_name = f"{args.datasetpath}/recs/reranked/chatgptser-{args.baseline_recs}.json"
    recs.to_json(out_name, orient="records")
    print(f"{datetime.datetime.now()} -- END!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasetpath",
        default=False,
        type=str,
        required=True,
        help="The path to the dataset"
    )
    parser.add_argument(
        "--promptpath",
        default=False,
        type=str,
        required=True,
        help="The path to the prompt"
    )
    parser.add_argument(
        "--foldpath",
        default=False,
        type=str,
        required=True,
        help="The path to the fold"
    )
    parser.add_argument(
        "--prompt_id",
        default=False,
        type=str,
        required=True,
        help="The id of the prompt"
    )
    parser.add_argument(
        "--baseline_recs",
        default=False,
        type=str,
        required=True,
        help="The baseline recommendations"
    )
    parser.add_argument(
        "--top_p",
        default=False,
        type=int,
        required=True,
        help="The # of items to show from the user profile"
    )
    parser.add_argument(
        "--rerank_top_m",
        default=False,
        type=int,
        required=True,
        help="The # of recommendations to rerank (max 100)"
    )
    parser.add_argument(
        "--top_n",
        default=False,
        type=int,
        required=True,
        help="The # of final recommendations to provide (max 100)"
    )
    parser.add_argument(
        "--openai_key",
        default=False,
        type=str,
        required=True,
        help="The API KEY for openai"
    )
    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )
    main(parser.parse_args())
