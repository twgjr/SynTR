import os
import shutil
from evaluator import compute_metrics
from fine_tune_colqwen import get_model_and_processor

def eval():
    dataset_dir='vidore/docvqa_test_subsampled_beir/'
    destination_dir = 'metrics'
    checkpoint_dir="trainer_output/best"
    base_model_name="Metric-AI/colqwen2.5-3b-multilingual"
    source_path = os.path.join(dataset_dir, 'metrics.json')

    os.makedirs(destination_dir, exist_ok=True)

    model, processor = get_model_and_processor(checkpoint_dir, use_peft=True)

    # evaluate finetune model on test set
    compute_metrics(
        model_name="finetuned",
        model= model,
        processor=processor,
        split_name="test", 
    )

    new_filename = 'finetuned-testset.json'
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.move(source_path, destination_path)

    model, processor = get_model_and_processor(checkpoint_dir, use_peft=False)

    # evaluate base model on test set
    compute_metrics(
        model_name="base",
        model= model,
        processor=processor,
        split_name="test", 
    )

    new_filename = 'base-testset.json'
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.move(source_path, destination_path)

if __name__=="__main__":
    eval()