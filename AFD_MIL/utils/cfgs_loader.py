import yaml
import json

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = f.read()
    
    cfgs = yaml.load(config, Loader=yaml.FullLoader)
    return cfgs


def load_json(path):
    pass