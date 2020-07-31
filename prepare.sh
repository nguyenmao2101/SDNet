mkdir coqa
mkdir coqa/bert-base-uncased
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-vocab.txt -P coqa/bert-base-uncased/ 
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual.tar.gz -P coqa/bert-base-uncased/ 
tar -xzf coqa/bert-base-uncased/bert-base-multilingual.tar.gz -C coqa/bert-base-uncased
