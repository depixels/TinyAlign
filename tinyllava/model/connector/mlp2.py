import re

import torch.nn as nn

from . import register_connector
from .base import Connector


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}




import os

import torch
import torch.nn as nn


class RAGMLP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._connector = None

    def load_model(self, **kwargs):
        pretrained_connector_path = kwargs.get('pretrained_connector_path', None)
        if pretrained_connector_path is not None:
            pretrained_connector_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')
            connector_weights = torch.load(pretrained_connector_path, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self._connector.load_state_dict(get_w(connector_weights, '_connector'))
            print(f'Loading connector from {pretrained_connector_path}...')

        for p in self._connector.parameters():
            p.requires_grad = False
   

    
    
    def forward(self, x):
        return self._connector(x)
        

  


    
@register_connector('rag2x_gelu')    
class MLPConnector2(RAGMLP):
    def __init__(self, config):
        super().__init__()
        
        mlp_gelu_match = re.match(r'^rag(\d+)x_gelu$', config.connector2_type)
        act_type = config.connector2_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(96, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        self._connector = nn.Sequential(*modules)

   
        
#     @property
#     def config(self):
#         return {"connector_type": 'mlp',
#                 "in_hidden_size": self.in_hidden_size, 
#                 "out_hidden_size": self.out_hidden_size
#                }
    
