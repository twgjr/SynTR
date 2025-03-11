import os
import json

def rename_json_field(path, field_name, new_name):
    with open(path, "r") as f:
        data = json.load(f)
    for item in data:
        item[new_name] = item.pop(field_name)
    with open(os.path.join(os.path.dirname(path), os.path.basename(path)+"_new.json"), "w") as f:
        json.dump(data, f, indent=4)


def save_json(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    # rename_json_field("vidore/docvqa_test_subsampled_beir/pseudo_qrel_truth.json", "query", "corpus-id")
    # rename_json_field("vidore/tatdqa_test_beir/pseudo_qrel_truth.json", "query", "corpus-id")