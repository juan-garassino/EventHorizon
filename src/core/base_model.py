# base_model.py

import configparser
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = self.config.get('verbose', False)
        logger.info("🔧 BaseModel initialized with configuration")
        if self.verbose:
            logger.info("🔍 Detailed initialization in progress")
            logger.info("📊 Analyzing configuration structure")

    @staticmethod
    def read_parameters(config_file: str) -> Dict[str, Dict[str, Any]]:
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(config_file)
        verbose = config.get('DEFAULT', {}).get('verbose', 'False').lower() == 'true'
        logger.info(f"📚 Reading configuration from {config_file}")
        if verbose:
            logger.info("🔬 Detailed configuration parsing in progress")
            logger.info("📑 Examining configuration sections")
        
        result = {section: {key: eval(val) for key, val in config[section].items()} 
                  for section in config.sections()}
        
        logger.info("✅ Configuration parsed successfully")
        if verbose:
            logger.info("🧐 Performing in-depth analysis of parsed data")
            logger.info("🗃️ Organizing parsed information")
        
        return result