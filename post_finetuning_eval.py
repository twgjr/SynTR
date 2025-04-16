import os
import shutil
from evaluator import compute_metrics

def main():
    dataset_dir='vidore/docvqa_test_subsampled_beir/'
    destination_dir = 'metrics'
    checkpoint_dir="colqwen_beir_checkpoints/best"
    base_model_name="Metric-AI/colqwen2.5-3b-multilingual"
    source_path = os.path.join(dataset_dir, 'metrics.json')

    os.makedirs(destination_dir, exist_ok=True)

    # evaluate finetune model on test set
    compute_metrics(
        checkpoint_path=checkpoint_dir, 
        split_name="test", 
        base_model_name=base_model_name
    )

    new_filename = 'finetuned-testset.json'
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.move(source_path, destination_path)

    # evaluate base model on test set
    compute_metrics(
        checkpoint_path=None, 
        split_name="test", 
        base_model_name=base_model_name
    )

    new_filename = 'base-testset.json'
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.move(source_path, destination_path)

    # evaluate finetuned model on the private set
    compute_metrics(
        checkpoint_path=checkpoint_dir, 
        split_name=None, 
        base_model_name=base_model_name
    )

    new_filename = 'finetuned-privateset.json'
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.move(source_path, destination_path)

    # evaluate base model on the private set
    compute_metrics(
        checkpoint_path=None, 
        split_name=None, 
        base_model_name=base_model_name
    )

    new_filename = 'base-privateset.json'
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.move(source_path, destination_path)


if __name__=="__main__":
    main()