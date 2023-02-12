from accelerate import init_empty_weights, load_checkpoint_in_model
import ray
import torch
from transformers import AutoTokenizer

from patched import Mailman, PatchedTestLM


MODEL_PATH = "/mnt/shared_storage/jungong/gpt_j/models/models--EleutherAI--gpt-j-6B/snapshots/6e35e2148e92edf096e94d39ac2b98ad59e25975/"


def run(shards):
    for _ in range(100):
        random_data = torch.rand((1, 10))
        label = torch.tensor(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
        )

        # Forward pass.
        ray.wait([shards[0].forward.remote(random_data)])
        for shard in shards[1:]:
            out_ref = shard.forward.remote()
            ray.wait([out_ref], fetch_local=False)
        out = ray.get(out_ref)

        print(out)

        # Backward pass.
        ray.wait([shards[-1].backward.remote(label)])
        for shard in reversed(shards[:-1]):
            ray.wait([shard.backward.remote()], fetch_local=False)

        # Step.
        ray.wait([shard.step.remote() for shard in shards])


def load_model():
    model_shards = []
    refs = []
    for shard in PatchedTestLM.SHARDING_PLAN:
        model_shard = PatchedTestLM.remote()
        model_shards.append(model_shard)
        refs.append(model_shard.prepare.remote(shard))

    # Wait for all model shards to finish loading.
    ray.wait(refs)

    return model_shards


if __name__ == "__main__":
    ray.init()

    # Global mailman.
    mailman = Mailman.options(name="mailman").remote()
    run(load_model())

    ray.shutdown()