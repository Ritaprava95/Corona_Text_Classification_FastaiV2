# corona_text_classification_fastaiV2
Text Classification Using FastAI V2

# creating dev enviroment
conda create -n fai_v2 python=3.8</br>
conda activate fai_v2</br>
conda install -c fastai -c pytorch fastai</br>
conda install jupyter</br>

## additional steps necessary for fastai2 text (this was nor required earlier but fastai did some changes on theie code so we have to do this in order to avoid errors)
pip install transformers</br>
pip install spacy-alignments


# Key points to be noted
1. This is for educationbal purpose
2. This repo involves end to end pipeline to classify text using fastai V2, I ignored pre and post processing steps.
3. I did not bother about accuracy of the mnodels
4. I am attaching the link of the data : https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?resource=download
