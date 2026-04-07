import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys

with open('/data/hyz/RAT/dataset/text_files/blip_laion_cc_sbu_558k.json', 'r') as f:
    pretrain = json.load(f)
# 指定设备为卡6
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

hf_path = '/data/hyz/RAT/checkpoints/tinyllava-phi'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.to(device)
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)

from PIL import Image
image = '/data/hyz/RAT/dataset/llava/llava_pretrain/images/'+pretrain[0]['image']
prompt = ' the caption of th image is: '
caption = pretrain[0]['conversations'][1]['value']

image_processor = model.vision_tower._image_processor

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

if image is not None:
    prompt = "<image>" + '\n' + prompt 

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

image = Image.open(image).convert("RGB")
image_tensor = process_images(image, image_processor, model.config).to(model.device)
input_ids = (
    tokenizer_image_token(prompt, tokenizer, -200, return_tensors="pt")
    .unsqueeze(0).to(model.device)
)

import torch.nn as nn
import torch
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor
from transformers.models.perceiver.modeling_perceiver import PerceiverMultimodalPreprocessor
from transformers.models.perceiver.modeling_perceiver import PerceiverEncoder
from transformers.models.perceiver.modeling_perceiver import PerceiverEmbeddings
from transformers.models.perceiver.modeling_perceiver import PerceiverConfig
from transformers import PerceiverTokenizer
class PerceiverMultiModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 创建图像和文本的预处理器
        image_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv1x1",  # 使用1x1卷积处理图像
            spatial_downsample=1,
            out_channels=256,
            position_encoding_type="trainable",
            concat_or_add_pos="concat",
            project_pos_dim=256,
            trainable_position_encoding_kwargs=dict(
                num_channels=256,
                index_dims=config.image_size**2,
            ),
        )
        
        text_preprocessor = PerceiverTextPreprocessor(config)
        
        # 创建多模态预处理器
        self.preprocessor = PerceiverMultimodalPreprocessor(
            modalities={
                "image": image_preprocessor,
                "text": text_preprocessor
            },
            min_padding_size=4
        )
        
        # 创建Perceiver编码器
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(
            config,
            kv_dim=self.preprocessor.num_channels
        )
        
    def forward(self, inputs):
        """
        Args:
            inputs: 字典,包含:
                - image: shape (batch_size, channels, height, width)
                - text: shape (batch_size, sequence_length)
                
        Returns:
            latent: shape (batch_size, num_latents, d_latents)
        """
        # 1. 预处理输入
        processed_inputs, _, _ = self.preprocessor(inputs)
        
        # 2. 获取latent的初始值
        batch_size = processed_inputs.shape[0]
        latent = self.embeddings(batch_size)
        # print(latent.shape)
        # 3. 通过编码器获取最终的latent表示
        encoder_outputs = self.encoder(
            latent,
            inputs=processed_inputs,
        )
        
        # 返回最后一层的latent表示
        return encoder_outputs[0]  # shape: (batch_size, num_latents, d_latents)

perceiver_tokenizer = PerceiverTokenizer.from_pretrained("/data/hyz/vlm/perceiver")

encoding = perceiver_tokenizer(caption)
encoding = torch.tensor(encoding['input_ids']).unsqueeze(dim=0).to(model.device)

config = PerceiverConfig(
    num_latents=32,          # latent序列的长度
    d_latents=96,           # latent的维度
    d_model=128,             # 预处理后的输入维度
    num_self_attends_per_block=8,  # self-attention层数
    num_blocks=1,            # 编码器block数量
    num_self_attention_heads=8,    # self-attention的头数
    num_cross_attention_heads=8,   # cross-attention的头数
    qk_channels=96,         # 添加这个参数，确保能被8整除
    v_channels=96, 
    image_size=384,          # 输入图像大小
    vocab_size=30522,        # 词表大小
    max_position_embeddings=512,  # 最大文本长度
)

perceiver_model = PerceiverMultiModalEncoder(config)
perceiver_model = perceiver_model.to(model.device)

inputs = {
    'image': image_tensor,
    'text' : encoding
}
latent = perceiver_model(inputs)
print("Latent shape:", latent.shape)  

from tqdm import tqdm
memory = {}
for i in tqdm(range(len(pretrain))):
    # if i < 56205:
    #     continue
    with torch.no_grad():  
        image = '/data/hyz/RAT/dataset/llava/llava_pretrain/images/'+pretrain[i]['image']
        image = Image.open(image).convert("RGB")
        image_tensor = process_images(image, image_processor, model.config).to(model.device)
        prompt = 'please describe the img.'
        caption = pretrain[i]['conversations'][1]['value']
        caption_new = '<image>\n' + ' the caption of the image is: ' + caption
        img_cap_token = tokenizer_image_token(caption_new, tokenizer, -200, return_tensors="pt").to(device)
        
        
        image_features = model.encode_images(image_tensor).squeeze(0).to(device)
        txt_embedding = model.language_model.model.embed_tokens(img_cap_token[1:].to(device))
        # multimodal_features = torch.cat((image_features, txt_embedding), dim=0)

        # input and key process logic
        alpha = 0.4  # 权重
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        txt_embedding = txt_embedding / txt_embedding.norm(dim=-1, keepdim=True)
        multimodal_features = torch.cat((alpha * image_features, (1 - alpha) * txt_embedding), dim=0)

        attention_weights = torch.softmax(torch.matmul(multimodal_features, multimodal_features.transpose(0, 1)), dim=-1)
        compressed_vector = torch.matmul(attention_weights.mean(dim=0), multimodal_features)
        compressed_vector = compressed_vector.unsqueeze(0)
        
        encoding = perceiver_tokenizer(caption)
        encoding = torch.tensor(encoding['input_ids']).unsqueeze(dim=0).to(model.device)

        inputs = {
            "image": image_tensor,
            "text": encoding,
        }
        latent = perceiver_model(inputs)
        
        del image_tensor, txt_embedding, image_features

        # multimodal_features = multimodal_features.to('cpu').clone()
        # latent = latent.to('cpu').clone()
        # memory[multimodal_features] = latent
        compressed_vector = compressed_vector.to('cpu').clone()
        latent = latent.to('cpu').clone()
        memory[compressed_vector] = latent
        torch.cuda.empty_cache()
        if i % 100000 == 0:
            print(f"Processed {i} images")
            torch.save(memory, f"Compressed_LLaVA_Dataset_Memory_{i}.pt")
            del memory
            memory = {}

# 保存 memory
torch.save(memory, "memory/LLaVA_Dataset_Memory.pt")

# 加载 memory
# memory = torch.load("memory.pt")

# import torch
# import os
# from PIL import Image
# import hashlib

# batch_size = 1000  # 每次处理 1000 条数据
# os.makedirs("memory", exist_ok=True)

# def hash_tensor(tensor):
#     return hashlib.md5(tensor.detach().numpy().tobytes()).hexdigest()

# for i in range(0, len(pretrain), batch_size):
#     memory = {}
#     for j in range(i, min(i + batch_size, len(pretrain))):
#         image_path = '/data/hyz/RAT/dataset/llava/llava_pretrain/images/' + pretrain[j]['image']
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = process_images(image, image_processor, model.config).to(model.device)

#         caption = pretrain[j]['conversations'][1]['value']
#         caption_new = '<image>\n' + ' the caption of the image is: ' + caption
#         img_cap_token = tokenizer_image_token(caption_new, tokenizer, -200, return_tensors="pt")

#         # 计算 multimodal_features
#         image_features = model.encode_images(image_tensor).squeeze(0).to(device)
#         txt_embedding = model.language_model.model.embed_tokens(img_cap_token[1:].to(device))

#         alpha = 0.4  # 权重
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         txt_embedding = txt_embedding / txt_embedding.norm(dim=-1, keepdim=True)
#         multimodal_features = torch.cat((alpha * image_features, (1 - alpha) * txt_embedding), dim=0)

#         # 计算 latent
#         encoding = perceiver_tokenizer(caption)
#         encoding = torch.tensor(encoding['input_ids']).unsqueeze(dim=0).to(model.device)

#         inputs = {
#             "image": image_tensor,
#             "text": encoding,
#         }
#         latent = perceiver_model(inputs)

#         # 移动到 CPU 并保存到当前批次的 memory
#         multimodal_features = multimodal_features.to('cpu')
#         latent = latent.to('cpu')
#         key = hash_tensor(multimodal_features)
#         memory[key] = {
#             "multimodal_features": multimodal_features,
#             "latent": latent
#         }

#         # 清理显存
#         torch.cuda.empty_cache()

#     # 保存当前批次的 memory
#     torch.save(memory, f"memory/LLaVA_Dataset_Memory_{i // batch_size}.pt")
#     print(f"Saved batch {i // batch_size}")
