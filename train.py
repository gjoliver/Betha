import argparse

from accelerate import init_empty_weights, load_checkpoint_in_model
import ray
import torch
from transformers import AutoTokenizer

from patched import Mailman, PatchedTestLM, PatchedGPTJ6B


def forward(shards, data):
    ray.get([shards[0].forward.remote(data)])
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


def run_gpt_j_6b(shards, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    inputs = tokenizer("i love large language model", return_tensors="pt")

    # Forward pass.
    out = ray.get(forward(shards, inputs))

    print("gpt-j: ", tokenizer.decode(out[0].tolist()))


def run_test(shards):
    for _ in range(10):
        random_data = torch.rand((1, 10))

        # Forward pass.
        print(ray.get(forward(shards, random_data)))

        # Backward pass. Fake target.
        label = torch.tensor(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
        )
        backward(shards, label)

        # Step.
        ray.wait([shard.step.remote() for shard in shards])


def load_test_model():
    model_shards = []
    refs = []
    for shard in PatchedTestLM.SHARDING_PLAN:
        model_shard = PatchedTestLM.remote()
        model_shards.append(model_shard)
        refs.append(model_shard.prepare.remote(shard))

    # Wait for all model shards to finish loading.
    ray.wait(refs)

    return model_shards


def load_gpt_j_6b(args):
    model_shards = []
    refs = []
    for shard in PatchedGPTJ6B.SHARDING_PLAN:
        model_shard = PatchedGPTJ6B.remote(args.model_dir)
        model_shards.append(model_shard)
        refs.append(model_shard.prepare.remote(shard))

    # Wait for all model shards to finish loading.
    ray.wait(refs)

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
        #run_test(load_test_model())
        run_gpt_j_6b(load_gpt_j_6b(args), args)
    except Exception:
        import time
        # Give Ray a few seconds to stream back the error logs.
        time.sleep(3)
        raise

    ray.shutdown()
