import re
import os
import datasets

# from verl.utils.hdfs_io import copy, makedirs
import argparse

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

# add a row to each data item that represents a unique id
def make_map_fn(split):

    def process_fn(example, idx):
        question = example.pop('problem')

        question = QUERY_TEMPLATE.format(Question=question)

        answer = str(example.pop('answer'))
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
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
    parser.add_argument('--local_dir', default='./')
    parser.add_argument('--hdfs_dir')

    args = parser.parse_args()

    data_source = 'deepscaler'

    dataset = datasets.load_dataset('/mnt/bn/daiweinan-fuse/datasets/DeepScaleR-Preview-Dataset')

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