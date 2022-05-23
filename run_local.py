from run import ex
import os


def pretraining():
    config_updates = {
        "data_root": "./datasets",
        # "num_gpus": 8,
        "num_gpus": 1,
        "num_nodes": 1,
        "whole_word_masking": True,
        # "per_gpu_batchsize": 64,
        "per_gpu_batchsize": 4,
        "dist": False,  # not to use DistributedSampler.
        "datasets": ["coco"],
        "num_workers": 0,
    }
    ex.run(config_updates=config_updates, named_configs=["task_mlm_itm", "step200k"])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    pretraining()
