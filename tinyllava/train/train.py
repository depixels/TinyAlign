from packaging import version
import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_connector2 = training_arguments.tune_type_connector2
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    model_args['connector2'] = _load_connector2_settings(model_arguments)
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args

def _load_connector2_settings(model_arguments):
    connector2_args = {}
    connector2_args['connector2_type'] = model_arguments.connector2_type
    return connector2_args

def _load_retrieval_settings(model_arguments):
    return {
        'retrieval_memory_dir': model_arguments.retrieval_memory_dir,
        'retrieval_index_file': model_arguments.retrieval_index_file,
        'retrieval_value_file': model_arguments.retrieval_value_file,
        'retrieval_text_start': model_arguments.retrieval_text_start,
        'retrieval_alpha': model_arguments.retrieval_alpha,
        'top_rag': model_arguments.top_rag,
    }

def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    
    model_args = training_recipe.add_args(model_args) #这一行在pretrain没用
    model_config = TinyLlavaConfig() # TinyllavaConfig配置了connector2但是还没设置<RAG>的相关token
    
    model_config.load_from_config(model_arguments)
    for key, value in _load_retrieval_settings(model_arguments).items():
        setattr(model_config, key, value)
    model = TinyLlavaForConditionalGeneration(model_config)
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        if 'tinyllava' not in training_arguments.pretrained_model_path:
            model = training_recipe.load(model, model_args)
        else:
            model.load_connector2(**model_args['connector2'])
            tinyllava = TinyLlava.from_pretrained(training_arguments.pretrained_model_path)
            model.language_model = tinyllava.language_model
            model.vision_tower = tinyllava.vision_tower
            model.connector = tinyllava.connector
            del tinyllava
            
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])
        model.load_connector2(**model_args['connector2'])
    
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
