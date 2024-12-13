import re
import os
import datasets

# from verl.utils.hdfs_io import copy, makedirs
import argparse

# To extract the solution for each prompts in the dataset
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

instruction_following = "Let's think step by step and output the final answer after \"####\"."

# add a row to each data item that represents a unique id
def make_map_fn(split):

    def process_fn(example, idx):
        question = example.pop('question')

        question = question + ' ' + instruction_following

        answer = example.pop('answer')
        solution = extract_solution(answer)
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/gsm8k')
    parser.add_argument('--hdfs_dir', default='hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf')

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset("/mnt/bn/daiweinan-fuse/datasets/gsm8k", 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

        # Construct a `def make_map_fn(split)` for the corresponding datasets.
    # ...

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # makedirs(hdfs_dir)

    # copy(src=local_dir, dst=hdfs_dir)