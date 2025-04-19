import os
import shutil
from evaluator import compute_metrics
from fine_tune_colqwen import get_model_and_processor

def eval():
    out_dir = 'metrics'
    checkpoint_dir="checkpoints"
    split_name="test"

    model, processor, _ = get_model_and_processor(checkpoint_dir=checkpoint_dir,
                                                 use_peft=True)

    # evaluate finetune model on test set
    model_name="finetuned"
    compute_metrics(
        model_name=model_name,
        model= model,
        processor=processor,
        split_name=split_name, 
        out_dir=out_dir, 
        out_name=model_name+"-"+split_name,
    )

    # get base model
    model, processor, _ = get_model_and_processor(checkpoint_dir=None, 
                                                use_peft=False)

    # evaluate base model on test set
    model_name="base"
    compute_metrics(
        model_name=model_name,
        model= model,
        processor=processor,
        split_name=split_name, 
        out_dir=out_dir, 
        out_name=model_name+"-"+split_name,
    )

if __name__=="__main__":
    eval()