'''
Load tokenizer and trained model

See: http://education.abcom.com/using-gpt-2-to-write-like-shakespeare/

# AWS GPU:
/opt/conda/lib/python3.6/pdb.py generate_text.py
# CPU:
/usr/local/lib/python3.6/pdb.py generate_text.py
'''
from backoff import on_exception, expo
from collections import Counter
import datetime
import json
import math
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
#import multiprocessing
import numpy as np
import os
import pandas as pd
from pprint import pprint
import re
import requests
import s3_tools
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import read_local_config


def greedy_search(model, tokenizer, ids):
    '''This is a very basic searching algorithm which selects the word
    with highest probability as its next word.
    '''
    # - this asks the model to guess the next 300 words after the seed.
    greedy_outputs = model.generate(ids, max_length=300)

    print("Output:\n" + 100 * '-')
    for i, greedy_output in enumerate(greedy_outputs):
        print("\n"+"==="*10)
        print("{}: {}".format(
            i+1, tokenizer.decode(greedy_output, skip_special_tokens=False)))


def beam_search(model, tokenizer, ids):
    '''Unlike greedy search that uses words with highest probability, the
    beam search considers the probabilities of the consequent number
    of words. It multiplies these probabilities with the previous ones
    for each case. Then, it selects the sequence of words which had
    overall higher probability after multiplication. The following
    statement performs a beam search.


    We set num_beams to be greater than 1 and early_stopping to true,
    so that generation finishes when all beam hypotheses reach the EOS
    token.

    '''
    beam_output = model.generate(
        ids,
        max_length=300,
        num_beams=4,
        early_stopping=True
    )
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(beam_output[0], skip_special_tokens=True))


def sample_condprobdist(model, tokenizer, ids):
    '''Sampling means randomly picking the next word according to its
    conditional probability distribution.

    '''
    sample_output = model.generate(
        ids,
        do_sample=True,
        max_length=300
    )
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


def sample_cpd_kp(model, tokenizer, ids, max_length=300, top_k=40,
                  top_p=0.95):
    '''Sampling means randomly picking the next word according to its
    conditional probability distribution.

    top-p & top-k both sample from truncated Neural LM distributions,
    differing only in the strategy of where to truncate.

    Top-K sampling:
        the K most likely next words are filtered, and it
        redistributes the probability mass among only those K next
        words.

    Top-p (Nucleus) Sampling
        It is the type of sampling which selects the highest
        probability tokens whose cumulative probability mass exceeds
        the pre-chosen threshold p. This threshold p can be randomly
        chosen, but we keep it above 0.9 for satisfactory results.

    '''
    # set top_k = 50 and set top_p = 0.95
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
    )

    result_texts = []
    for i, final_output in enumerate(final_outputs):
        decoded_text = tokenizer.decode(final_output, skip_special_tokens=True)
        #print("{}: {}".format(i, decoded_text))
        result_texts.append(decoded_text)
    return result_texts


def download_transformer_model(local_root_path, s3_uri):
    ''' Download model from S3 if local model is not the same.
    '''
    version_filename = "model_version.uri"
    model_version_fpath = "%s/%s" % (local_root_path, version_filename)
    download_model = True
    if os.path.exists(model_version_fpath):
        with open(model_version_fpath) as fp:
            ver_str = fp.read().strip()
            if ver_str == s3_uri:
                download_model = False

        # see if the local version file is the same as the remove url

        #model_version_uri = "%s/%s" % (s3_uri, version_filename)
        #data = get_s3_file_contents(model_version_uri)
        # TODO check if the same
    if download_model:
        try:
            shutil.rmtree(local_root_path)
            os.mkdir(local_root_path)
        except:
            pass
        s3_tools.copy_directory(s3_uri, local_root_path)

        # Write version to local path
        with open(model_version_fpath, 'w') as fp:
            fp.write(s3_uri)
    print("Running Model URI:", s3_uri)


def get_evaluation_data(evaluation_data_fpath=None):
    ''' Get data used for evaluating model.

    # evaluation_*.json has form:
    [
      {
        "seed_text": "Some sentence that is to be finished by the model ",
        "expected_words": []
      },
    '''
    # To generate file using a dict:
    # data = [{'seed_text':'',
    #         'expected_words':[]},
    # ]
    #evaluation_data_fpath = 'evaluation_interesting.json'
    # with open(evaluation_data_fpath, 'w') as fp:
    #    json.dump(data, fp)
    # cat evaluation_interesting.json | jq

    with open(evaluation_data_fpath) as fp:
        data = json.load(fp)
    return data


WORD = re.compile(r"\w+")


def get_cosine(vec1, vec2):
    ''' Simple pure-python cosine similarity implementation.
    https://stackoverflow.com/a/15174569/1106632
    '''
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def model_infer_task(atup):
    '''For a given text-seed task, use model to infer rest of text.
    Return inferred products and cosine similarity.  

    This can be a mapped task, intended to be run in parallel.
    Although it seems the task alread uses all cores, so parallelizing
    doesn't speed things up.
    '''
    (model, tokenizer, eval_pars, eval_dict) = atup
    ids = tokenizer.encode(eval_dict['seed_text'],
                           return_tensors='pt')
    result_list = sample_cpd_kp(model, tokenizer, ids,
                                max_length=eval_pars['max_length'],
                                top_k=eval_pars['top_k'],
                                top_p=eval_pars['top_p'])
    pred_text = result_list[0]  # only 1 text was evaluated
    v_pred = text_to_vector(pred_text.replace(
        eval_dict['seed_text'], ''))
    v_orig = text_to_vector(eval_dict['orig_text'].replace(
        eval_dict['seed_text'], ''))
    cosine = get_cosine(v_pred, v_orig)
    #print(f"{cosine:.6f}: {pred_text}\n{'-'*20}\n{eval_dict['orig_text']}\n{'-'*78}")
    return cosine


@on_exception(expo, (requests.exceptions.Timeout,
                     requests.exceptions.ConnectionError),
              max_time=60)
def log_metrics(adict):
    mlflow.log_metrics(adict)


@on_exception(expo, (requests.exceptions.Timeout,
                     requests.exceptions.ConnectionError),
              max_time=60)
def log_params(adict):
    mlflow.log_params(adict)


@on_exception(expo, (requests.exceptions.Timeout,
                     requests.exceptions.ConnectionError),
              max_time=60)
def start_run(*args, **kwargs):
    ''' DEBUG: not using this since I get an error:
           AttributeError: __enter__
    This function is being used with "with", unlike above functions.
    Might need to define a decorator class?  
    TODO look at:
    https://stackoverflow.com/questions/22417323/how-do-enter-and-exit-work-in-python-decorator-classes
    '''
    mlflow.start_run(*args, **kwargs)


def evaluate_model_gpt2(pars, eval_pars):
    '''Evaluate Huggingface transformers GPT2 model using seed text,
    upload resulting summary stats to cooresponding model run in
    MLflow tracking server.

    Older methods:
    ### Generate the output using model:
    greedy_search(model, tokenizer, ids)
    beam_search(model, tokenizer, ids)
    sample_condprobdist(model, tokenizer, ids)
    result_list = sample_cpd_kp(model, tokenizer, ids,
                                max_length=100,
                                top_k=40, top_p=0.95)
    '''
    client = MlflowClient()
    experiment = client.get_experiment_by_name(pars['mlflow_experiment_name'])
    exp_cur_id = experiment.experiment_id
    runs = client.list_run_infos(exp_cur_id)
    run_id_latest = runs[0].run_id
    s3_uri = runs[0].artifact_uri

    # Download model from S3 if local model is not the same:
    download_transformer_model(pars['local_model_path'], s3_uri)

    # Load tokenizer and trained model
    tokenizer = GPT2Tokenizer.from_pretrained('/local/output')
    model = GPT2LMHeadModel.from_pretrained('/local/output')
    
    eval_data_fpath = f"{pars['eval_data_dir']}/{eval_pars['fname']}"
    eval_data = get_evaluation_data(eval_data_fpath)
    cos_list = []
    dt1 = datetime.datetime.now()
    map_tasks = []
    for i in range(eval_pars['n_eval_sets']):
        print(f"{i}/{eval_pars['n_eval_sets']}")
        for eval_dict in eval_data:
            map_tasks.append((model, tokenizer, eval_pars, eval_dict))

    ### Non parallelized:
    cos_list = [model_infer_task(a) for a in map_tasks]
    ### This doesn't speed things up, full CPU cores already used:
    #cos_iter = map(model_infer_task, map_tasks)
    #cos_list = list(cos_iter)
    
    ### This results in a "bus error" probably unable to load multiple
    ### copies of input data.  Probably not worth using this due to
    ### relativly fast task:
    #pool = multiprocessing.Pool()
    #cos_iter = pool.map(model_infer_task, map_tasks)
    print("Time to infer:", datetime.datetime.now() - dt1)
    
    ss = pd.Series(cos_list)
    sstats = ss.describe()
    sstats['median'] = np.median(ss)
    temp = sstats.to_dict()
    sstats_dict = {}
    for k,v in temp.items():
        k_new = k.replace("%","")
        sstats_dict[f"{pars['key_prefix']}{k_new}"] = v
    print("Cosine similarity summary stats (1=same):")
    pprint(sstats_dict)

    renamed_eval_pars = {f"{pars['key_prefix']}{k}":v for k,v in eval_pars.items()}
    ### Add metrics to existing run in mlflow tracking server:
    with mlflow.start_run(run_id=run_id_latest):
        #retry(mlflow.log_metrics(sstats_dict),      max_retry=3, t_sleep=3)
        #retry(mlflow.log_params(renamed_eval_pars), max_retry=3, t_sleep=3)
        log_metrics(sstats_dict)
        log_params(renamed_eval_pars)
        # This could be a prediction dict or json:
        # (eval data used, eval results)
        #mlflow.log_artifact("fullpathtosomefile.json", "start_run_test")

    import pdb; pdb.set_trace()
    print()

    
def evaluate_models():
    '''Evaluate how the model(s) perform on certain tasks.  Store results
    as MLflow metrics for each model.
    '''
    local_config = read_local_config('local_config.json')

    ### GPU EC2 -> remote:
    #mlflow.set_tracking_uri(local_config["mlflow_tracking_uri_priv"])
    ### laptop -> remote server:
    mlflow.set_tracking_uri(local_config["mlflow_tracking_uri_pub"])
    ### laptop -> local server
    #mlflow.set_tracking_uri("http://192.168.0.5:5000")
    pars = {
        'local_model_path':'/local/output',
        'eval_data_dir':'/local/data/personal',
        'key_prefix':'seedeval_stats_',
        'mlflow_experiment_name':'Personal GPU',
        }
    #    'mlflow_experiment_name':'Laptop Quick Train',
    
    eval_pars = {'n_eval_sets': 1,
                 'fname':'evaluation_1.json',
                 'max_length':100,
                 'top_k':40,
                 'top_p':0.95,
    }

    evaluate_model_gpt2(pars, eval_pars)


if __name__ == "__main__":

    evaluate_models()
