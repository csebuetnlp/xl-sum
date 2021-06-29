#!/bin/bash

# install transformers and requirements
pip install --upgrade -r requirements.txt
pip install --upgrade transformers/

# install rouge module and dependecies
pip install -r ../multilingual_rouge_scoring/requirements.txt
python -m unidic download # for japanese segmentation
pip install --upgrade ../multilingual_rouge_scoring
python -m nltk.downloader punkt