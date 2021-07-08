# Compare Personality Models

This project sets up a framework for comparing the accuracy / utility of transformer language models that are fine tune trained on a corpus of texts generated by a single person.

The motivation for this project:
* How much utility is there with fine tune training a transformer using a large single-person text corpus?
* Since public use of GPT-3 doesn't allow fine tune training on large
  texts, I'll focus on other publically available models and see their
  limits for this application.
* How does GPT-2 compare to GPT-J-6B, GPT-Neo?


Currently, the corpus consists of:
* A directory of `.docx` files (downloaded from Google docs)
* An email `.mbox` containing all sent emails by a person.
* All google chats sent by a person.

Currently, we compare transformer models:
* Huggingface pytorch GPT-2


Docs:
  * [Running required MLflow server](mlflow_server/README.md)
  * [Infrastructure reference](docs/infrastructure.md)

External reference pages:
  * shakespeare GPT-2: http://education.abcom.com/using-gpt-2-to-write-like-shakespeare/
  * reference language model (which we adapted):
  wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/legacy/run_language_modeling.py

* Download shakespeare dataset for model training:
``` bash
mkdir -p data/tinyshakespeare
cd /local/data/tinyshakespeare
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

* To see if GPU is being used:
``` bash
nvidia-smi
nvidia-smi -i 0 -l -q -d UTILIZATION
```

* NOTE: huggingface repo says The `run_language_modeling.py` script is deprecated in favor of `language-modeling/run_{clm, plm, mlm}.py`.

* Starting up model training (on GPU EC2):
```bash
export AWS_ACCESS_KEY_ID=`aws configure get aws_access_key_id`
export AWS_SECRET_ACCESS_KEY=`aws configure get aws_secret_access_key`
make shell-gpu

# make sure the train data is up to date:
/local/data/personal/combo.txt

# To start a fresh model:
rm -Rf /local/data/personal/cached_lm*

# Run the model in PDB (to interact with error namespace):
/opt/conda/lib/python3.6/pdb.py run_language_modeling.py --overwrite_output_dir --output_dir=/local/output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=/local/data/personal/combo.txt --per_gpu_train_batch_size=1 --save_steps=-1 --num_train_epochs=1
```

* To evaluate / compare models using evaluation seed text:
```bash
/usr/local/lib/python3.6/pdb.py generate_text.py
```
