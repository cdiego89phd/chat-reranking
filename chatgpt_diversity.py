import pickle
import argparse
import datetime
import json
import openai
import pandas as pd
import time


MODEL_DICT = {"gpt-3.5-turbo": "chatgpt",
              "gpt-3.5-turbo-0613": "chatgpt0613",
              "gpt-3.5-turbo-instruct": "instructgpt",
              "gpt-4": "gpt-4"
              }

MODELS = {"gpt-3.5-turbo": "optimized for chat. It should be the latest model available but I am not sure about OpenAI updates policies",
          "gpt-3.5-turbo-0613": "Snapshot of gpt-3.5-turbo from June 13th 2023 with function calling data. Unlike gpt-3.5-turbo, this model will not receive updates, and will be deprecated 3 months after a new version is released.",
          "gpt-3.5-turbo-instruct": "Similar capabilities as text-davinci-003 but compatible with legacy Completions endpoint and not Chat Completions."
          }


def load_helper_dicts(data_folder: str
                      ) -> (dict, dict):

    filename = f"{data_folder}itemid_to_name.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemid_to_name = pickle.load(fp)
    filename = f"{data_folder}itemname_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemname_to_id = pickle.load(fp)

    filename = f"{data_folder}itemid_to_namegenres.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemid_to_namegenres = pickle.load(fp)
    filename = f"{data_folder}itemnamegenres_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemnamegenres_to_id = pickle.load(fp)

    return itemid_to_name, itemname_to_id, itemid_to_namegenres, itemnamegenres_to_id


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

    f = open(f"{promptpath}")
    prompt_template = json.load(f)["templates"]
    f.close()
    prompt_template = {i["id"]: i["text"] for i in prompt_template}

    return prompt_template[prompt_id]


def build_prompts(recs: pd.DataFrame,
                  top_m: int,
                  top_n: int,
                  itemid_to_name: dict,
                  template: str,
                  domain: str,
                  ) -> list:

    # add the output format to the template
    template_prompt = template
    template_prompt = template_prompt.replace("<top_m>", str(top_m))
    template_prompt = template_prompt.replace("<top_n>", str(top_n))
    for i in range(1, top_n + 1):
        template_prompt += f"{i}-> <{domain} name>\n"

    users = recs.userid.values
    prompts = []
    for user in users:
        # select recommendations
        user_recs = recs[recs["userid"] == user]["recs"].values[0]

        # add the baseline recommendations
        items_str = "\n"
        for i, item in enumerate(user_recs, start=1):
            items_str += f"{i}. {itemid_to_name[item]}\n"
        prompts.append(template_prompt + f"\n```{items_str}```")

    return prompts


def get_response_chatgpt(model: str,
                         prompt: str
                         ) -> str:
    # model list can be found above
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


def get_response_instructgpt(model: str,
                             prompt: str
                             ) -> str:
    # model list can be found above

    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=600,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        output = response.choices[0]["text"]
    except Exception as e:
        output = str(e)

    return output

def parse_response(output: str,
                   itemname_to_id: dict
                   ) -> list:

    lines = output.splitlines()
    reranked_recs = []
    for line in lines:
        try:
            if len(line.split("-> ")) > 0:
                item_name = line.split("-> ")[1]
                reranked_recs.append(itemname_to_id[item_name])
        except Exception as e:
            continue
    return reranked_recs


def query_chatgpt(model: str,
                  prompts: list,
                  itemname_to_id: dict,
                  ) -> (list, list):

    raw_responses = []
    reranked_recs = []
    for i, prompt in enumerate(prompts, start=1):

        if model == "gpt-3.5-turbo-instruct":
            response = get_response_instructgpt(model, prompt)
        else:
            response = get_response_chatgpt(model, prompt)
        raw_responses.append(response)

        new_rank = parse_response(response, itemname_to_id)
        reranked_recs.append(new_rank)

        time.sleep(20)
        print(f"{datetime.datetime.now()} -- Done {i}/{len(prompts)} users!")
    return raw_responses, reranked_recs


def main(args):

    # load helper dictionaries
    itemid_to_name, itemname_to_id, itemid_to_namegenres,  itemnamegenres_to_id = load_helper_dicts(args.datasetpath)

    if args.prompt_id in ["5", "6"]:  # the name of the items are augmented with genres
        itemid_to_name = itemid_to_namegenres
        itemname_to_id = itemnamegenres_to_id

    print(f"{datetime.datetime.now()} -- Helpers loaded!")

    # load prompt template (from json)
    prompt_template = load_prompt_template(args.promptpath, args.prompt_id)
    print(f"{datetime.datetime.now()} -- Prompt template loaded!")

    # load baseline recommendations
    recs = pd.read_csv(f"{args.datasetpath}/recs/baselines/{args.baseline_recs}", sep="\t",
                       names=["userid", "itemid", "rating"])
    recs.drop(labels="rating", axis=1, inplace=True)

    if args.run_with_sample_users:
        # load test users
        test_users = pd.read_csv(f"{args.datasetpath}/fold_{args.fold}/sample_test_users.csv",
                                 sep="\t", names=["userid"])
        recs = recs[recs["userid"].isin(test_users["userid"].values)].copy()

    print(f"{datetime.datetime.now()} -- Baseline recommendations loaded!")

    if args.debug_mode:
        debug_usrs = recs["userid"].unique()[:30]
        recs = recs[recs["userid"].isin(debug_usrs)].copy()

    # trim recommendations
    recs = recs.groupby('userid').head(args.rerank_top_m).reset_index(drop=True)
    recs = convert_dataframe(recs, args.baseline_recs, args.top_n, args.rerank_top_m)
    print(f"{datetime.datetime.now()} -- Recommendations trimmed!")

    # craft prompts from template and top-m baseline recs
    prompts = build_prompts(recs, args.rerank_top_m,  args.top_n, itemid_to_name, prompt_template, args.domain)
    recs["prompt"] = prompts
    print(f"{datetime.datetime.now()} -- Prompts crafted!")

    # query chatgpt
    openai.api_key = args.openai_key
    raw_gpt_outputs, reranked_recs = query_chatgpt(args.model, prompts, itemname_to_id)
    recs["raw_gpt_outputs"] = raw_gpt_outputs
    recs["reranked_recs"] = reranked_recs
    print(f"{datetime.datetime.now()} -- Done with ChatGPT!")

    # save it into file
    output_name = MODEL_DICT[args.model]
    out_name = f"{args.datasetpath}/recs/reranked/{output_name}-div-p{args.prompt_id}-{args.baseline_recs}.json"
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
        "--domain",
        default=False,
        type=str,
        required=True,
        help="The domain of recommendation (item type)"
    )
    parser.add_argument(
        "--fold",
        default=False,
        type=str,
        required=True,
        help="The data fold."
    )
    parser.add_argument(
        "--model",
        default=False,
        type=str,
        required=True,
        help="The model to use (either gpt-3.5-turbo or gpt-4)"
    )
    parser.add_argument(
        "--promptpath",
        default=False,
        type=str,
        required=True,
        help="The path to the prompt"
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
        "--run_with_sample_users",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script with a sample of the test users."
    )
    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )
    main(parser.parse_args())
