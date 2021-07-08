#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.


### AWS GPU:
export AWS_ACCESS_KEY_ID=`aws configure get aws_access_key_id`
export AWS_SECRET_ACCESS_KEY=`aws configure get aws_secret_access_key`
make shell-gpu

# (make sure case ##### GPU EC2 Training ##### is enabled), then run:
/opt/conda/lib/python3.6/pdb.py run_language_modeling.py --overwrite_output_dir --output_dir=/local/output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=/local/data/personal/combo.txt --per_gpu_train_batch_size=1 --save_steps=-1 --num_train_epochs=5
# (NOTE: --num_train_epochs=5 takes about an hour to train on g4dn.xlarge: 2.8it/s for 10k iterations)


### CPU:
export AWS_ACCESS_KEY_ID=`aws configure get aws_access_key_id`
export AWS_SECRET_ACCESS_KEY=`aws configure get aws_secret_access_key`
make shell-cpu

# (make sure case ##### Small Dataset Training ##### is enabled), then run:
/usr/local/lib/python3.6/pdb.py run_language_modeling.py --overwrite_output_dir --output_dir=/local/output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=/local/data/personal/combo.txt --per_gpu_train_batch_size=1 --save_steps=-1 --num_train_epochs=5

"""
import boto3
from dataclasses import dataclass, field
from glob import glob
import gzip
import json
import logging
import math
import mlflow.pytorch
import os
from pprint import pprint
import re
import s3_tools
import sys
from torch.utils.data import ConcatDataset
from typing import Optional
import utils

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={
                                  "help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    def _dataset(file_path, ref_path=None):
        if args.line_by_line:
            if ref_path is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError(
                        "You need to set world whole masking and mlm to True for Chinese Whole Word Mask")
                return LineByLineWithRefDataset(
                    tokenizer=tokenizer,
                    file_path=file_path,
                    block_size=args.block_size,
                    ref_path=ref_path,
                )

            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        else:
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=cache_dir,
            )

    if evaluate:
        return _dataset(args.eval_data_file, args.eval_ref_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file, args.train_ref_file)


def main(train_metadata):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(
            data_args.block_size, tokenizer.model_max_length)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer,
                    cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        if data_args.mlm and data_args.whole_word_mask:
            data_collator = DataCollatorForWholeWordMask(
                tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        mlflow.log_params(train_metadata)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    try:
        print('>> >>', mlflow.get_artifact_uri())
    except:
        print('>> >> EXCEPTION: mlflow.get_artifact_uri()', sys.exc_info()[0])

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    # Not sure why MLflow doesn't log this:
    mlflow.log_params({'run_id': mlflow.active_run().info.run_id,
                       'experiment_id': mlflow.active_run().info.experiment_id})
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()



def rm_pattern(path, pattern):
    ''' rm files under a path, which match a regex pattern
    '''
    for f in os.listdir(path):
        if re.search(pattern, f):
            print("rm: %s/%s" % (path, f))
            os.remove(os.path.join(path, f))

def get_latest_traindata_from_s3(train_data_fpath, train_ref_s3_uri):
    '''Download from S3 the latest/current training data and associated
    meta data.

    Use S3 json train_ref_s3_uri to get current training s3 URIs.

    Save training data into file `train_data_fpath`
    '''
    ### Get train data from S3:
    s3 = boto3.resource('s3')
    
    # Get ref dict:
    (s3_bucket, s3_subpath) = s3_tools.parse_s3_dir(train_ref_s3_uri)
    obj = s3.Object(s3_bucket, s3_subpath)
    body = obj.get()['Body'].read()
    ref_dict = json.loads(body)
    
    # Get metadata:
    (s3_bucket, s3_subpath) = s3_tools.parse_s3_dir( \
                                            ref_dict['train_metadata_s3_uri'])
    obj = s3.Object(s3_bucket, s3_subpath)
    body = obj.get()['Body'].read()
    metadata_dict = json.loads(body)
    pprint(metadata_dict)

    # Get train data:
    (s3_bucket, s3_subpath) = s3_tools.parse_s3_dir( \
                                            ref_dict['train_data_s3_uri'])
    obj = s3.Object(s3_bucket, s3_subpath)
    with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
        content = gzipfile.read()
    with open(train_data_fpath, "w") as fp:
        fp.write(content.decode('utf-8'))

    return metadata_dict

    
if __name__ == "__main__":

    os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = 'TRUE'
    # For HuggingFace Transformers + MLflow, I delete the existing
    # model rather than try to resume:
    rm_pattern('/local/data/personal', 'cached_lm*')

    local_config = utils.read_local_config('local_config.json')

    train_data_s3_path = local_config["train_data_s3_path"] # "s3://.../..."
    train_data_fpath = '/local/data/personal/combo.txt'

    if 0:
        ##### Small Dataset Training #####
        ### laptop -> local server
        #mlflow.set_tracking_uri("http://192.168.0.5:5000")
        ### laptop -> remote server:
        mlflow.set_tracking_uri(local_config["mlflow_tracking_uri_pub"])
        mlflow.set_experiment("Laptop Quick Train")
        ### smaller train dataset:
        train_ref_fname = "parse_personal_small_ref.json"
    else:
        ##### GPU EC2 Training #####
        ### GPU EC2 -> remote:
        mlflow.set_tracking_uri(local_config["mlflow_tracking_uri_priv"])
        mlflow.set_experiment("Personal GPU")
        ### larger train dataset:
        train_ref_fname = "parse_personal_large_ref.json"
        
    train_ref_s3_uri = f"{train_data_s3_path}/{train_ref_fname}"
    train_metadata = get_latest_traindata_from_s3(train_data_fpath,
                                                  train_ref_s3_uri)

    mlflow.pytorch.autolog()
    # this triggers an error since start_run() is run later:
    #mlflow.log_params(train_metadata)
    main(train_metadata)

    ### Get current model run_id:
    #run_id = mlflow.active_run().info.run_id
    
    # this generates a new mlflow run entry:
    #mlflow.log_params(train_metadata)
