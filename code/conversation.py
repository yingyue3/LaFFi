
import torch
import json
from tqdm import tqdm
import argparse
import ray
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def gen_answer(model_path, question, o_file):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage = True,
        torch_dtype=torch.float16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False)
    print("----------Model Load Successfully------------")
    prompt = one_shot(question)
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample = True,
        temperature=0.8,
        max_new_tokens = 200
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
    outputs = output_ids[:outputs.find("###")]
    outputs = outputs.strip()
    print(outputs)
    with open(o_file,"w") as o_f:
        o_f.write(outputs)

def one_shot(question):
    shot1 = f"""
    question: who are you?
    answer: I'a language model named llama. You can ask me any questions.
    ###
    question: {question}
    """
    return shot1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,required=True)
    parser.add_argument("--question", type=str,required=True)
    parser.add_argument("--output",type=str,required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(os.path.expanduser(args.output), "w") as fout:
        fout.write("hello world")
    print("hello")
    gen_answer(
        args.model_path,
        args.question,
        args.output
    )