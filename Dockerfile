#FROM python:3.7
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04

RUN apt-get update && apt-get install -y emacs-nox elpa-markdown-mode jq less

RUN pip install --upgrade pip
ADD requirements.txt /
RUN pip install -r /requirements.txt && rm -rf /root/.cache/pip/*

