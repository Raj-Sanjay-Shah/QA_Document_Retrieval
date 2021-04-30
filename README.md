# Create a better system for answering questions

## Requirements
1. pip3 install -r requirements.txt
2. Download questions from [here](https://www.google.com/url?q=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpinafore-us-west-2%2Fqanta-jmlr-datasets%2Fqanta.train.2018.04.18.json&sa=D&sntz=1&usg=AFQjCNGf7EtqkO16UWbMx_eeAexvvoIXxw) and save as 'qanta.train.json'. Users can also change the value of the variable 'questions_file' in test_queries.py to the correct path.
3. Download documents from [here](https://www.google.com/url?q=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpinafore-us-west-2%2Fqanta-jmlr-datasets%2Fwikipedia%2Fwiki_lookup.json&sa=D&sntz=1&usg=AFQjCNFJ_cCrB0wkRniaZ9yRWg7dvBslMw) and save as 'wiki_lookup.json'. Users can also change the value of the variable 'file_name_documents' in Index_Creation_code.py to the correct path.

## Steps to run the code:
1. Install all the requirements in the file requirements.txt by using the above code.
2. python run.py (comment out the command to run the index creation code after running the script for the first time.)

## Document Retrieval
1. Index creation for tf-idf is done by the python file Index_Creation_code.py
2. Please note that location of files containing the corpus can be changed according to convenience
in the beginning of all the python files.
	- I have only retrieved the document titles while displaying the ranking. I assume that the titles are sufficient to identify the document. The top three documents are also stored in the for use in BERT-SQuAD.
## Supported improvements to tf-idf vector based retrieval:
1. Spell checking.
2. Using synoynms for equivalence classes.
3. Using zonal indexing to give weights to the title and the body.
### Results for retrieval
| Case |  Accuracy |
| ------------- | ------------- |
| If the prediction is considered to be correct when the actual document is in the top 1 results | 67%  |
| If the prediction is considered to be correct when the actual document is in the top 5 results | 80%  |
| If the prediction is considered to be correct when the actual document is in the top 10 results | 84%  |
| If the prediction is considered to be correct when the actual document is in the top 20 results | 84%  |
| If the prediction is considered to be correct when the actual document is in the top 30 results | 85%  |

| Case when only the first sentence of the document is considered as the query| Accuracy |
| ------------- | ------------- |
| If the prediction is considered to be correct when the actual document is in the top 1 results | 9%  |
| If the prediction is considered to be correct when the actual document is in the top 5 results | 20%  |
| If the prediction is considered to be correct when the actual document is in the top 10 results | 26%  |
| If the prediction is considered to be correct when the actual document is in the top 20 results | 30%  |
| If the prediction is considered to be correct when the actual document is in the top 30 results | 36%  |

## Answering system
I use the BERT-SQuAD pre-trained model from here [https://github.com/kamalkraj/BERT-SQuAD]. Given a document and a question, this model gives specific answers with confidence levels.
### Pretrained model download from [here](https://www.dropbox.com/s/8jnulb2l4v7ikir/model.zip)
unzip and move files to model directory
