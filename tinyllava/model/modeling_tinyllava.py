from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import faiss
import numpy as np
import os

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlava(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector = ConnectorFactory(config.connector_type)(config)

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        self.post_init()

    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_features = self.encode_images(images)

        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    # 加载连接器
    def load_connector(self, **kwargs):
        # 调用连接器的load_model方法，并传入参数kwargs
        self.connector.load_model(**kwargs)

            

        
        


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector = ConnectorFactory(config.connector_type)(config)
        self.connector2 = ConnectorFactory(config.connector2_type)(config)

    
        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        self.post_init()
        self.index = None
        self.values = None
        self._load_retrieval_memory()

    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features


    def _load_retrieval_memory(self):
        save_dir = getattr(self.config, "retrieval_memory_dir", None)
        if not save_dir:
            raise ValueError("`retrieval_memory_dir` must be set for TinyAlign retrieval.")

        index_path = os.path.join(save_dir, self.config.retrieval_index_file)
        value_path = os.path.join(save_dir, self.config.retrieval_value_file)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Retrieval index not found: {index_path}")
        if not os.path.exists(value_path):
            raise FileNotFoundError(f"Retrieval values not found: {value_path}")

        print(f"Loading retrieval memory from {save_dir}")
        self.index = faiss.read_index(index_path)
        data = torch.load(value_path, map_location="cpu")
        self.values = data["values"]
        print(f"Loaded FAISS index with {self.index.ntotal} entries.")

    def get_rag_items(self, query_key, top_k=None):
        if top_k is None:
            top_k = self.config.top_rag
        batch_size, nums, dims = query_key.shape
        query_key = query_key.view(-1, dims)
        
        query_vector = np.array(query_key.detach().cpu(), dtype=np.float32)
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / np.clip(query_norm, a_min=1e-12, a_max=None)
        _, indices = self.index.search(query_vector, top_k)
        
        similar_values = []
        for i in range(batch_size):
            similar_values.append(torch.concat([self.values[idx] for idx in indices[i]], dim=1))
        batch_sim_values = torch.concat(similar_values, dim=0)
        return batch_sim_values

    def encode_rag_items(self, rag_items):
        rag_items = rag_items.to(device=self.device, dtype=self.dtype)
        rag_items = self.connector2(rag_items)
        return rag_items

    def _build_retrieval_query(self, input_ids, image_features):
        retrieval_text_start = min(getattr(self.config, "retrieval_text_start", 35), input_ids.shape[1])
        retrieval_alpha = getattr(self.config, "retrieval_alpha", 0.4)

        text_input_ids = input_ids[:, retrieval_text_start:]
        if text_input_ids.shape[1] == 0:
            multimodal_features = image_features
        else:
            text_features = self.language_model.model.embed_tokens(text_input_ids)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            multimodal_features = torch.cat(
                (retrieval_alpha * image_features, (1 - retrieval_alpha) * text_features),
                dim=1,
            )

        attention_weights = torch.softmax(
            torch.matmul(multimodal_features, multimodal_features.transpose(1, 2)),
            dim=-1,
        )
        return torch.matmul(attention_weights.mean(dim=1, keepdim=True), multimodal_features)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #     images, image_sizes=None
    # ):
    #     vision_tower = self.vision_tower
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     image_features = self.encode_images(images)
    #     retrieval_txt_embedding = self.language_model.model.embed_tokens(input_ids[:,2:])
    #     alpha = 0.4  # 权重
    #     _image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     retrieval_txt_embedding = retrieval_txt_embedding / retrieval_txt_embedding.norm(dim=-1, keepdim=True)
    #     multimodal_features = torch.cat((alpha * _image_features, (1 - alpha) * retrieval_txt_embedding), dim=1)
        
    #     attention_weights = torch.softmax(torch.matmul(multimodal_features, multimodal_features.transpose(1, 2)), dim=-1)
    #     # using the average vec as the query vector.
    #     compressed_vector = torch.matmul(attention_weights.mean(dim=1, keepdim=True), multimodal_features)

    #     rag_item = self.get_rag_items(compressed_vector)
    #     rag_item = rag_item.to(device=input_ids.device)
    #     rag_embed = self.encode_rag_items(rag_item)
    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False):
    #         raise NotImplementedError

    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- FIXME
    #     _input_ids = input_ids
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0

    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            
    #         # rag_token_indices = [-1] + torch.where(cur_input_ids_noim == RAG_TOKEN_INDEX)[0].tolist() + [cur_input_ids_noim[1].shape[0]]
    #         # cur_input_ids_noimrag = []
    #         # cur_labels_noimrag = []
    #         # for i in range(len(rag_token_indices) - 1):
    #         #     cur_input_ids_noimrag.append(cur_input_ids_noim[rag_token_indices[i]+1:rag_token_indices[i+1]])
    #         #     cur_labels_noimrag.append(cur_labels_noim[rag_token_indices[i]+1:rag_token_indices[i+1]])
            
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]

    #         cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         cur_new_input_embeds = []
    #         cur_new_labels = []

    #         # for i in range(num_images + 1):
    #         num_images = 1
    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_image_idx += 1
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
    #         cur_rag_item = rag_item[batch_idx]
    #         cur_new_input_embeds.append(cur_rag_item)
    #         cur_new_labels.append(torch.full((cur_rag_item.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

        
    #     # Truncate sequences to max length as image embeddings can make the sequence longer
    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # Combine them
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 attention_mask[i, -cur_len:] = True
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 attention_mask[i, :cur_len] = True
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
        ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images(images)
        compressed_vector = self._build_retrieval_query(input_ids, image_features)
        rag_item = self.get_rag_items(compressed_vector, top_k=self.config.top_rag)
        rag_item = rag_item.to(device=input_ids.device)

        rag_embed = self.encode_rag_items(rag_item)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # 处理没有图像的情况
                cur_image_features = image_features[cur_image_idx]
                
                # 处理可能存在的RAG token（负数）
                valid_ids_mask = cur_input_ids >= 0
                valid_ids = cur_input_ids[valid_ids_mask]
                
                # 对有效ID进行嵌入
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(valid_ids)
                
                # 如果有RAG token，需要单独处理
                if not torch.all(valid_ids_mask):
                    # 获取所有输入嵌入的形状
                    full_embed_shape = list(cur_input_embeds_1.shape)
                    full_embed_shape[0] = len(cur_input_ids)
                    
                    # 创建完整的嵌入张量
                    full_embeds = torch.zeros(full_embed_shape, 
                                            dtype=cur_input_embeds_1.dtype, 
                                            device=cur_input_embeds_1.device)
                    
                    # 将有效ID的嵌入放入对应位置
                    full_embeds[valid_ids_mask] = cur_input_embeds_1
                    
                    # 处理RAG token
                    rag_mask = ~valid_ids_mask
                    if torch.any(rag_mask):
                        # 这里假设RAG token的嵌入已经在rag_embed中
                        # 根据实际情况可能需要调整
                        rag_indices = torch.where(rag_mask)[0]
                        for idx in rag_indices:
                            full_embeds[idx] = rag_embed[batch_idx]  # 使用对应批次的RAG嵌入
                    
                    cur_input_embeds = torch.cat([full_embeds, cur_image_features[0:0]], dim=0)
                else:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # 处理有图像的情况
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            # 存储RAG token的索引和对应的标签
            rag_indices = []
            rag_labels = []
            
            for i in range(len(image_token_indices) - 1):
                segment = cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]
                segment_labels = cur_labels[image_token_indices[i]+1:image_token_indices[i+1]]
                
                # 分离RAG token和普通token
                valid_mask = segment >= 0
                
                # 保存普通token
                valid_segment = segment[valid_mask]
                valid_segment_labels = segment_labels[valid_mask]
                cur_input_ids_noim.append(valid_segment)
                cur_labels_noim.append(valid_segment_labels)
                
                # 记录RAG token的位置和标签
                if not torch.all(valid_mask):
                    rag_mask = ~valid_mask
                    rag_positions = torch.where(rag_mask)[0]
                    for pos in rag_positions:
                        rag_indices.append((i, pos.item()))  # 保存(segment_idx, position)
                        rag_labels.append(segment_labels[pos].item())
            
            # 处理普通token的嵌入
            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            if any(split_sizes):  # 确保有普通token
                cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            else:
                cur_input_embeds_no_im = []
            
            cur_new_input_embeds = []
            cur_new_labels = []
            
            # # 按顺序重建嵌入序列，包括图像和RAG token
            # num_images = 1  # 假设每个批次只有1个图像
            for i in range(num_images + 1):
                if i < len(cur_input_embeds_no_im):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_rag_item = rag_embed[batch_idx]
                    cur_new_input_embeds.append(cur_rag_item)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_labels.append(torch.full((cur_rag_item.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            # # 添加RAG item
            # cur_rag_item = rag_embed[batch_idx]
            # cur_new_input_embeds.append(cur_rag_item)
            # cur_new_labels.append(torch.full((cur_rag_item.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            # 确保所有嵌入在同一设备上
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
        
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)

    def load_connector2(self, **kwargs):
        pretrained_connector2_path = get_value_from_kwargs(kwargs, 'pretrained_connector2_path')
        if pretrained_connector2_path is not None:
            kwargs['pretrained_connector_path'] = pretrained_connector2_path
        self.connector2.load_model(**kwargs)

        
        
