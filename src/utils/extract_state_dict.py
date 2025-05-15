import torch
import os
checkpoint_path = 'atlas_data/experiments/jsa-bge/checkpoint/step-10000/model.pth.tar'
model_dict = torch.load(checkpoint_path, map_location='cpu')
passage_encoder_dict = {
    k.replace('retriever.passage_contriever.encoder.', ''):v for k,v in model_dict['model'].items() if k.startswith('retriever.passage_contriever')
    }
query_encoder_dict = {
    k.replace('retriever.query_contriever.encoder.', ''):v for k,v in model_dict['model'].items() if k.startswith('retriever.query_contriever')
    }
save_path = os.path.join(os.path.dirname(checkpoint_path), 'passage_encoder.pth.tar')
torch.save(passage_encoder_dict, save_path)
save_path = os.path.join(os.path.dirname(checkpoint_path), 'query_encoder.pth.tar')
torch.save(query_encoder_dict, save_path)
