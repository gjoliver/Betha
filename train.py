import argparse

import ray
import torch
from transformers import AutoConfig, AutoTokenizer

from model import (
    EmbeddingModule,
    GPTJBlocksModule,
    GPTJBlockShardConfig,
    LMHeadModule,
)
from shard import Shard
from test import TestLMShard1, TestLMShard2


# TODO(jungong) : We could use distributed queues here to run the
# shards with pipeline parallelism.
def forward(shards, inputs, labels=None):
    for shard in shards[:-1]:
        inputs = shard.forward.remote(inputs)
        ray.wait([inputs], fetch_local=False)

    # For last shard, get actual inputs so we can add labels.
    inputs = ray.get(inputs)
    if labels is not None:
        inputs["labels"] = labels

    outputs = shards[-1].forward.remote(inputs)

    return ray.get(outputs)


def backward(shards):
    # backward() doesn't return data. So we can use ray.get() to wait
    # while fetching any potential errors.
    gradients = {}
    for shard in reversed(shards):
        gradients = shard.backward.remote(gradients)
        ray.wait([gradients], fetch_local=False)


def run_gpt_j(shards, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    inputs = tokenizer("i love large language model", return_tensors="pt")

    # Forward pass.
    out = ray.get(forward(shards, inputs))

    print("gpt-j: ", tokenizer.decode(out[0].tolist()))


def load_gpt_j(args):
    config = AutoConfig.from_pretrained(
        args.model_dir
    )
    model_shards = [
        Shard.options(num_gpus=0.5).remote(
            lambda: EmbeddingModule(config)
        ),  # GPU 0
        Shard.options(num_gpus=1).remote(
            lambda: GPTJBlocksModule(
                config,
                GPTJBlockShardConfig(0, 5, includ_layer_norm=False)
            )
        ), # GPU 1
        Shard.options(num_gpus=1).remote(
            lambda: GPTJBlocksModule(
                config,
                GPTJBlockShardConfig(6, 10, includ_layer_norm=False)
            )
        ), # GPU 2
        Shard.options(num_gpus=1).remote(
            GPTJBlocksModule(
                config,
                GPTJBlockShardConfig(11, 15, includ_layer_norm=True)
            )
        ), # GPU 3
        Shard.options(num_gpus=0.5).remote(
            LMHeadModule(config),
        ), # GPU 0
    ]
    return model_shards


def run_test(shards):
    for _ in range(30):
        random_data = torch.rand((1, 10))
        # Fake target.
        labels = torch.tensor(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
        )

        # Forward pass.
        result = forward(shards, inputs={"x": random_data}, labels=labels)
        print(result)

        # Backward pass.
        backward(shards)

        # Step.
        ray.wait([shard.step.remote() for shard in shards])


def load_test_model():
    model_shards = [
        Shard.remote(lambda: TestLMShard1()),
        Shard.remote(lambda: TestLMShard2()),
    ]
    return model_shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Path to a pretrained huggingface GPT-J model.",
    )

    args = parser.parse_args()

    ray.init()

    try:
        #run_test(load_test_model())
        run_gpt_j(load_gpt_j(args), args)
    except Exception:
        import time
        # Give Ray a few seconds to stream back the error logs.
        time.sleep(3)
        raise

    ray.shutdown()
