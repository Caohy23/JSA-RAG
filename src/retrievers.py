# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# In 2024/7/24, Cao Hongyu changed the original "Contriever" to "Embedding_con".
import copy
import os
import torch
import transformers
from transformers import AutoConfig
from src.modeling_bert import BertModel

EMBEDDINGS_DIM: int = 768

#修改名字，记住声明

class Embedding_con(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0).clone()
        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1).clone() / attention_mask.sum(dim=1)[..., None].clone()
        elif self.config.pooling == "sqrt":
            emb = last_hidden.sum(dim=1) / torch.sqrt(attention_mask.sum(dim=1)[..., None].float())
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1).clone()
        return emb


def get_class(model_name):
    if 'dpr' in model_name and 'question' in model_name:
        return transformers.DPRQuestionEncoder, transformers.DPRQuestionEncoderTokenizer
    elif 'dpr' in model_name and 'ctx' in model_name:
        return transformers.DPRContextEncoder, transformers.DPRContextEncoderTokenizer
    elif 'bge' in model_name or 'contriever' in model_name or 'nomic' in model_name or 'gte' in model_name:
        return transformers.AutoModel, transformers.AutoTokenizer
    else:
        print('Unseen class')

def get_embeddings(model_output, model_name, model_input=None):
    if 'dpr' in model_name:
        embeddings = model_output.pooler_output # (B, H)
        embeddings = embeddings.contiguous()
        #embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
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

class Embedding_Ret(torch.nn.Module):

    def __init__(self, encoder_path):
        # q_encoder: query encoder
        # p_encoder: passage encoder
        super().__init__()
        self.encoder_path = encoder_path
        model_cls, _ = get_class(self.encoder_path)
        if 'nomic' in self.encoder_path:
            self.encoder = model_cls.from_pretrained(self.encoder_path, trust_remote_code=True)
        else:
            self.encoder = model_cls.from_pretrained(self.encoder_path)

    def forward(self, *args, build_index = False, **kwargs):
        
        if build_index:
            with torch.no_grad():
                model_output = self.encoder(*args, **kwargs)
                embeddings = get_embeddings(model_output, self.encoder_path)
        else :
            model_output = self.encoder(*args, **kwargs)
            embeddings = get_embeddings(model_output, self.encoder_path)

        return embeddings
    
    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def save_models(self, save_directory):
        """Save the query and passage encoders using the save_pretrained method."""

        self.encoder.save_pretrained(save_directory)


class BaseRetriever(torch.nn.Module):
    """A retriever needs to be able to embed queries and passages, and have a forward function"""

    def __init__(self, *args, **kwargs):
        super(BaseRetriever, self).__init__()

    def embed_queries(self, *args, **kwargs):
        raise NotImplementedError()

    def embed_passages(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, is_passages=False,grad_no_pass=False, **kwargs):
        if is_passages:
            return self.embed_passages(*args,grad_no_pass=grad_no_pass, **kwargs)
        else:
            return self.embed_queries(*args, **kwargs)

    def gradient_checkpointing_enable(self):
        for m in self.children():
            m.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        for m in self.children():
            m.gradient_checkpointing_disable()


class DualEncoderRetriever(BaseRetriever):
    """Wrapper for standard contriever, or other dual encoders that parameter-share"""

    def __init__(self, opt, contriever):
        super(DualEncoderRetriever, self).__init__()
        self.opt = opt
        self.contriever = contriever

    def _embed(self, *args, **kwargs):
        return self.contriever(*args, **kwargs)

    def embed_queries(self, *args, **kwargs):
        return self._embed(*args, **kwargs)

    def embed_passages(self, *args, **kwargs):
        return self._embed(*args, **kwargs)


class UntiedDualEncoderRetriever(BaseRetriever):
    """Like DualEncoderRetriever, but dedicated encoders for passage and query embedding"""

    def __init__(self, opt, query_encoder, passage_encoder=None):
        """Create the module: if passage_encoder is none, one will be created as a deep copy of query_encoder"""
        super(UntiedDualEncoderRetriever, self).__init__()
        self.opt = opt
        self.query_retriever = query_encoder
        if opt.fix_encoder:
            self.query_retriever.requires_grad_(False)
        else:
            self.query_retriever.requires_grad_(True)
        if passage_encoder is None:
            if self.opt.decouple_encoder:
                passage_encoder = copy.deepcopy(query_encoder) 

            else:
                passage_encoder = query_encoder
        self.passage_retriever = passage_encoder
        if self.opt.query_side_retriever_training:
            if self.opt.decouple_encoder:
                self.passage_retriever.requires_grad_(False)

    def embed_queries(self, *args, **kwargs):
        return self.query_retriever(*args, **kwargs)

    def embed_passages(self, *args,grad_no_pass, **kwargs):
        if self.opt.query_side_retriever_training or grad_no_pass:
            self.passage_retriever.eval()
            with torch.no_grad():
                passage_emb = self.passage_retriever(*args, **kwargs)
        else:
            self.passage_retriever.train()
            passage_emb = self.passage_retriever(*args, **kwargs)
        return passage_emb



