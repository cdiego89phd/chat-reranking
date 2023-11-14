import datetime

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import argparse
import time


def save_model_local(model_name, to_path, auth_token):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 use_auth_token=auth_token)
    model.save_pretrained(to_path)
    return model


def load_model_from_local(path):
    # return AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    return AutoModelForCausalLM.from_pretrained(path, local_files_only=True, device_map="auto")


def prompting_model(prompt, model, tokenizer, to_cuda=False):

    generation_pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        # we pass model parameters here too
        # temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return generation_pipe.predict(prompt)


def prompting_with_dataset(prompt, model, tokenizer, to_cuda=False):

    generation_pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
    )

    def data_iterator():
        for i in range(10):
            yield prompt

    batch_output = []
    print(f'{datetime.datetime.now()}')
    for out in generation_pipe(data_iterator()):
        batch_output.append(out[0]['generated_text'])
    print(f'{datetime.datetime.now()}')
    return batch_output


def main(args):
    m = save_model_local(args.model_name, args.to_path, args.auth_token)
    t = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                      use_auth_token=args.auth_token)
    print("tokenizer loaded")
    m = load_model_from_local(args.model_path)
    print("model loaded")
    # print(m)
    prompt = "How were you trained?"
    # output = prompting_model(prompt, m, t, to_cuda=True)

    start = time.time()
    output = prompting_with_dataset(prompt, m, t, to_cuda=True)
    for i, p in enumerate(output):
        print(f'OUT {i}: {p}\n\n')
    print(output)
    print(f'Inference done in {time.time()-start}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auth_token",
        default=False,
        type=str,
        required=True,
        help="The HF auth token."
    )
    parser.add_argument(
        "--model_name",
        default=False,
        type=str,
        required=True,
        help="The name of the model to load/save."
    )
    parser.add_argument(
        "--to_path",
        default=False,
        type=str,
        required=True,
        help="The path to the model to load/save."
    )
    main(parser.parse_args())
