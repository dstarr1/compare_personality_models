'''
Parse personal text and store in dataset suitable for training.
Parse:
 - docx
 - google chats
 - emails sent

/usr/local/lib/python3.6/pdb.py parse_personal.py
'''
import datetime
from docx import Document
from email_mbox_parse import *
import glob
import json
import os
from pprint import pprint
import s3_tools
import utils


def parse_docxs():
    ''' parse .docx text (downloaded from google-docs via .zip)
    '''
    text_list = []
    fpaths = glob.glob('/local/data/personal/docx/*docx')
    for fpath in fpaths:
        print(fpath)
        document = Document(fpath)
        for p in document.paragraphs:
            txt = p.text
            if len(txt) > 0:
                text_list.append(txt)
    return text_list


def parse_googlechats(chat_creator_email):
    ''' Parse text from personal side of converstations in google chats.
    '''
    text_list = []
    fpaths = glob.glob('/local/data/personal/googlechat/*/messages.json')
    for json_fpath in fpaths:
        all_mess = json.load(open(json_fpath))
        for mess_dict in all_mess['messages']:
            if mess_dict.get('creator', {}).get('email', None) == chat_creator_email:
                if 'text' in mess_dict:
                    text_list.append(mess_dict['text'])
    return text_list


def filter_junk_text(text):
    ''' Filter out useless text lines
    '''
    new_text = []
    for line in text:
        if 'http' in line:
            continue
        elif '----' in line:
            continue
        elif '####' in line:
            continue
        new_text.append(line)
    return new_text


def upload_train_data(train_data, train_data_s3_path, train_ref_fname, metadata_dict):
    '''Upload to S3 3 files which describe the training data:

    train_data : text containing lines used for training model.  This
                 will be stored gzipped on s3.

    train_metadata : json containing information about data used in
                 training dataset, which could be used in monitoring,
                 mlflow, etc.

    train_ref : json which describes URI location of current/latest
                 training data that is to be used for model building.

    '''
    cur_dtime = datetime.datetime.utcnow()
    train_data_fname =     f"parse_personal_{cur_dtime}.txt.gz"
    train_metadata_fname = f"parse_personal_metadata_{cur_dtime}.json"
    train_data_s3_uri =    f"{train_data_s3_path}/{train_data_fname}"
    train_metadata_s3_uri =f"{train_data_s3_path}/{train_metadata_fname}"
    train_ref_s3_uri =     f"{train_data_s3_path}/{train_ref_fname}"

    metadata_dict.update({"train_data_s3_uri":train_data_s3_uri,
                          "train_metadata_s3_uri":train_metadata_s3_uri})

    ref_dict = {"train_data_s3_uri":train_data_s3_uri,
                "train_metadata_s3_uri":train_metadata_s3_uri}

    # Upload train data:
    (s3_bucket, s3_subpath) = s3_tools.parse_s3_dir(train_data_s3_uri)
    s3_tools.upload_str_gzipped(train_data, s3_bucket, s3_subpath)

    # Upload train metadata:
    (s3_bucket, s3_subpath) = s3_tools.parse_s3_dir(train_metadata_s3_uri)
    s3_tools.upload_str_uncompressed(json.dumps(metadata_dict), s3_bucket, s3_subpath)


    # Upload train reference json:
    (s3_bucket, s3_subpath) = s3_tools.parse_s3_dir(train_ref_s3_uri)
    s3_tools.upload_str_uncompressed(json.dumps(ref_dict), s3_bucket,
                                     s3_subpath)
    
    
def main():
    '''
    '''
    local_config = utils.read_local_config('local_config.json')
    
    mbox_fpath = '/local/data/personal/Sent.mbox/mbox'
    train_data_s3_path = local_config["train_data_s3_path"] # "s3://.../..."

    if 0:
        ##### Laptop Training (small dataset) #####
        train_ref_fname = "parse_personal_small_ref.json"
        #max_n_messages=50: 382 lines 15105 bytes, 3 model iters
        mbox_dict = get_mytext_from_mbox(mbox_fpath, max_n_messages=50)
        text_docs = mbox_dict['all_lines']
    else:
        ##### GPU EC2 Training (large dataset) #####
        train_ref_fname = "parse_personal_large_ref.json"
        text_docs = parse_docxs()
        chat_text = parse_googlechats(local_config["chat_creator_email"])
        text_docs.extend(chat_text)
        mbox_dict = get_mytext_from_mbox(mbox_fpath)
        text_docs.extend(mbox_dict['all_lines'])
    
    text = filter_junk_text(text_docs)
    # TODO strip '\n' lines in a seperate function called later.
    # TODO remove sensitive data

    metadata_dict = mbox_dict['meta_data']
    train_data = '\n'.join(text)
    upload_train_data(train_data, train_data_s3_path,
                      train_ref_fname, metadata_dict)
    

if __name__ == "__main__":
    main()
