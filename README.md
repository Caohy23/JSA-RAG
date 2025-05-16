## Environment Setup  
First, create your own Conda environment and execute the following commands:  

**TSrag**  
```bash  
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch  
conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3  
pip install -r requirements.txt  
```  


## Data Preparation  
**Download Wikipedia data:**  
```bash  
python data/preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory data  
```  

**Download question-answering datasets (including Natural Questions and TriviaQA):**  
```bash  
python data/preprocessing/prepare_qa.py --output_directory data/data/  
```  


## Start index_server Workflow  
Detailed steps are in [build_server](./build_server/).  
Port information can be modified synchronously in two places:  
- Sender: src/post.py  
- Receiver: build_server/server_start.py  


## Experiments  
The experiment directory is [egs](./egs/), where each subfolder represents an experiment. Folder naming convention: `{training-dataset}-{training-method}-{training-script}`.  

**Example command:**  
```bash  
bash egs/NaturalQuestion/JSA/run.sh  
```  

During training, you can set `--eval_freq` to test the model at regular intervals. To test saved models, run:  
```bash  
bash EVAL.sh  
```  

For other experiment combinations, create corresponding folders and refer to the instructions in [docs](./docs/) to complete the `run.sh` script in the experiment folder.
