import json
import torch

from accelerate import init_empty_weights, load_checkpoint_in_model
import ray
from transformers import AutoTokenizer

from patched import PatchedModel, init
from test import TEST_SHARDING


MODEL_PATH = "/mnt/shared_storage/jungong/gpt_j/models/models--EleutherAI--gpt-j-6B/snapshots/6e35e2148e92edf096e94d39ac2b98ad59e25975/"


def run(shards):
    random_data = torch.rand((1, 1, 28, 28))
    x = ray.get(shards[0].call.remote(random_data))
    x = ray.get(shards[1].call.remote(random_data))
    print(x)
    return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    while True:
        x = input("prompt (press RETURN to exit): ")

        if not x: break

        inputs = tokenizer(x, return_tensors="pt")
        outputs = model.generate(**inputs)

        print("gpt-j: ", tokenizer.decode(outputs[0].tolist()))


def load_model():
    model_shards = []
    refs = []
    for shard in TEST_SHARDING:
        model_shard = PatchedModel.remote()
        model_shards.append(model_shard)
        refs.append(model_shard.load_and_patch.remote(shard))

    # Wait for all model shards to finish loading.
    ray.wait(refs)

    return model_shards


if __name__ == "__main__":
    mailman = init()  # Global mailman.
    run(load_model())