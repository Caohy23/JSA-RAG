import torch
import transformers
from transformers import AutoConfig
from retrieve_post import call_retrieve_api

def get_class(model_name):
    if 'dpr' in model_name and 'question' in model_name:
        return transformers.DPRQuestionEncoder, transformers.DPRQuestionEncoderTokenizer
    elif 'dpr' in model_name and 'ctx' in model_name:
        return transformers.DPRContextEncoder, transformers.DPRContextEncoderTokenizer
    elif 'bge' in model_name or 'contriever' in model_name or 'nomic' in model_name or 'gte' in model_name:
        return transformers.AutoModel, transformers.AutoTokenizer
    else:
        return transformers.AutoModel, transformers.AutoTokenizer

def get_embeddings(model_output, model_name, model_input=None):
    if 'dpr' in model_name:
        embeddings = model_output.pooler_output # (B, H)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    elif 'bge' in model_name:
        embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    elif 'contriever' in model_name:
        embeddings = mean_pooling(model_output[0], model_input['attention_mask'].cuda())
    elif 'nomic' in model_name:
        embeddings = model_output[0]
        mask_expanded = model_input['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float().cuda()
        embeddings = torch.sum(embeddings*mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    elif 'gte' in model_name:
        embeddings = average_pool(model_output.last_hidden_state, model_input['attention_mask'].cuda())
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class RetModel(torch.nn.Module):

    def __init__(self, encoder_paths):
        # q_encoder: query encoder
        # p_encoder: passage encoder
        super().__init__()
        if len(encoder_paths)==1:
            self.q_encoder_path = encoder_paths[0]
            self.p_encoder_path = None
        else:
            self.q_encoder_path = encoder_paths[0]
            self.p_encoder_path = encoder_paths[1]
        
        q_model_cls, q_tokenizer_cls = get_class(self.q_encoder_path)
        if 'nomic' in self.q_encoder_path:
            self.q_encoder = q_model_cls.from_pretrained(self.q_encoder_path, trust_remote_code=True)
        else:
            self.q_encoder = q_model_cls.from_pretrained(self.q_encoder_path)
        self.q_tokenizer = q_tokenizer_cls.from_pretrained(self.q_encoder_path)
        self.q_conifg = AutoConfig.from_pretrained(self.q_encoder_path, trust_remote_code=True)
        #print("RET_SPECIAL_TOKEN")
        SPECIAL_TOKENS = {
        "additional_special_tokens": ["<speaker1>", "<speaker2>"]
        }
        self.q_tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.q_encoder.resize_token_embeddings(len(self.q_tokenizer))
        #print("SPECIALL_TOKEN",self.q_tokenizer.special_tokens_map["additional_special_tokens"])
        special_token_ids = self.q_tokenizer.all_special_ids
        speaker1_token_id = self.q_tokenizer.convert_tokens_to_ids("<speaker1>")
        # print("Token ID of <speaker1>:", speaker1_token_id)
        # print("Token ID of <speaker1>:", speaker1_token_id)
        # print("Special token IDs:", special_token_ids)
        if self.p_encoder_path:
            p_model_cls, p_tokenizer_cls = get_class(self.p_encoder_path)
            self.p_encoder = p_model_cls.from_pretrained(self.p_encoder_path)
            self.p_tokenizer = p_tokenizer_cls.from_pretrained(self.p_encoder_path)
            self.p_config = AutoConfig.from_pretrained(self.p_encoder_path)
        else:
            self.p_encoder = self.q_encoder
            self.p_tokenizer = self.q_tokenizer
            self.p_config = self.q_conifg
            self.p_encoder_path = self.q_encoder_path
            #print("PASSAGE_RET_SPECIAL_TOKEN")
            SPECIAL_TOKENS = {
            "additional_special_tokens": ["<speaker1>", "<speaker2>"]
            }
            self.p_tokenizer.add_special_tokens(SPECIAL_TOKENS)
            self.p_encoder.resize_token_embeddings(len(self.p_tokenizer))
            #print("SPECIALL_TOKEN",self.p_tokenizer.special_tokens_map["additional_special_tokens"])
            special_token_ids = self.p_tokenizer.all_special_ids
            speaker1_token_id = self.p_tokenizer.convert_tokens_to_ids("<speaker1>")
            # print("Token ID of <speaker1>:", speaker1_token_id)
            # print("Token ID of <speaker1>:", speaker1_token_id)
            # print("Special token IDs:", special_token_ids)
    @torch.no_grad()
    def embed(self, text_batch, is_passage=False):
        if is_passage:
            ids_batch = self.p_tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            model_output = self.p_encoder(**_to_cuda(ids_batch))
        else:
            ids_batch = self.q_tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            model_output = self.q_encoder(**_to_cuda(ids_batch))

        embeddings = get_embeddings(model_output, self.q_encoder_path, ids_batch)

        return embeddings
    
    @torch.no_grad()
    def retrieve(
        self,
        topk,
        query
    ):
        query_emb = self.embed(query, is_passage=False)
        #passages, scores, passages_emb = index.search_knn(query_emb, topk)
        passages, scores = call_retrieve_api(query_emb, topk)
        passages_emb = None
        return passages, scores, query_emb, passages_emb
