import vilt.utils.write_vg as write_vg
import vilt.utils.write_coco_karpathy as write_coco_karpathy


def prepare_dataset():
    write_coco_karpathy.make_arrow(root="./datasets/coco", dataset_root="./datasets")
    # write_vg.make_arrow(root="./datasets/vg", dataset_root="./datasets")


if __name__ == "__main__":
    prepare_dataset()
