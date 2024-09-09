# base_model.py

import configparser
from typing import Dict, Any

class BaseModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @staticmethod
    def read_parameters(config_file: str) -> Dict[str, Dict[str, Any]]:
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(config_file)
        return {section: {key: eval(val) for key, val in config[section].items()} 
                for section in config.sections()}