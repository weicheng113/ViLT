from run import ex


def pretraining():
    config_updates = {
        "data_root": "./datasets",
        "num_gpus": 1,
        "num_nodes": 1,
        "whole_word_masking": True,
        "per_gpu_batchsize": 64,
        "datasets": ["coco"],
    }
    ex.run(config_updates=config_updates, named_configs=["task_mlm_itm", "step200k"])


if __name__ == "__main__":
    pretraining()
