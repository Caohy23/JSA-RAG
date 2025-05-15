
# This folder is used to create knowledge base embeddings and load embeddings to the server  

## Store document embeddings:  
```bash  
bash retrieve.sh  
```  
**Parameter description for the above script:**  
- `passage_path`: Loading path of the text knowledge base.  
- `retriever_model`: Path to the retrieval model.  

## Load embeddings into GPU and start the server:  
```bash  
bash run_server_start.sh  
```