import argparse

import ray
import torch
from transformers import AutoConfig, AutoTokenizer

from mailman import Mailman
from model import (
    EmbeddingModule,
    GPTJBlocksModule,
    GPTJBlockShardConfig,
    LMHeadModule,
)
from test import TestLMShard1, TestLMShard2


def forward(shards, **inputs):
    ray.get([shards[0].forward.remote(**inputs)])
    for shard in shards[1:]:
        ref = shard.forward.remote()
        ray.wait([ref], fetch_local=False)
    return ref


def backward(shards, target):
    # backward() doesn't return data. So we can use ray.get() to wait
    # while fetching any potential errors.
    ray.get([shards[-1].backward.remote(target)])
    for shard in reversed(shards[:-1]):
        ray.get([shard.backward.remote()])


def run_gpt_j(shards, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    inputs = tokenizer("i love large language model", return_tensors="pt")

    # Forward pass.
    out = ray.get(forward(shards, **inputs))

    print("gpt-j: ", tokenizer.decode(out[0].tolist()))


def load_gpt_j(args):
    config = AutoConfig.from_pretrained(
        args.model_dir
    )
    model_shards = [
        EmbeddingModule.remote(config),  # GPU 0
        GPTJBlocksModule.remote(
            config,
            GPTJBlockShardConfig(0, 5, includ_layer_norm=False)
        ), # GPU 1
        GPTJBlocksModule.remote(
            config,
            GPTJBlockShardConfig(6, 10, includ_layer_norm=False)
        ), # GPU 2
        GPTJBlocksModule.remote(
            config,
            GPTJBlockShardConfig(11, 15, includ_layer_norm=True)
        ), # GPU 3
        LMHeadModule.remote(config),     # GPU 0
    ]
    return model_shards


def run_test(shards):
    for _ in range(10):
        random_data = torch.rand((1, 10))

        # Forward pass.
        print(ray.get(forward(shards, x=random_data)))

        # Backward pass. Fake target.
        label = torch.tensor(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
        )
        backward(shards, label)

        # Step.
        ray.wait([shard.step.remote() for shard in shards])


def load_test_model():
    model_shards = [
        TestLMShard1.remote(),
        TestLMShard2.remote()
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

    # Global mailman.
    mailman = Mailman.options(name="mailman").remote()

    try:
        run_test(load_test_model())
        #run_gpt_j(load_gpt_j(args), args)
    except Exception:
        import time
        # Give Ray a few seconds to stream back the error logs.
        time.sleep(3)
        raise

    ray.shutdown()
