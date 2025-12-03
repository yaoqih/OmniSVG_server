import torch.nn as nn
import torch
import yaml
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig


def load_config(config_path=None):
    """Load configuration from config.yaml"""
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "config.yaml"),
            os.path.join(current_dir, "..", "config.yaml"),
            "config.yaml"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        if config_path is None:
            raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_model_specific_value(config, model_size, *keys):
    """Get model-specific config value with fallback to shared config."""
    # Try model-specific config first
    model_cfg = config.get('models', {}).get(model_size, {})
    value = model_cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    # Fallback to shared config if not found
    if value is None:
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
    
    return value


class SketchDecoder(nn.Module):
    """
    Autoregressive generative model - supports both 8B and 4B models
    """
    def __init__(self, config_path=None, model_path=None, model_size=None, **kwargs):
        """
        Initialize SketchDecoder.
        
        Args:
            config_path: Path to config.yaml
            model_path: HuggingFace model path (overrides config if provided)
            model_size: Model size ("8B" or "4B"). If None, uses default from config.
            **kwargs: Additional arguments (e.g., torch_dtype, pix_len, text_len)
        """
        super().__init__()
        
        config_data = load_config(config_path)
        
        # Determine model size
        self.model_size = model_size or config_data.get('default_model_size', '8B')
        if self.model_size not in config_data.get('models', {}):
            raise ValueError(f"Invalid model_size: {self.model_size}. Must be one of: {list(config_data.get('models', {}).keys())}")
        
        print(f"[SketchDecoder] Initializing with model_size: {self.model_size}")
        
        # Get model-specific and shared configs
        model_config = config_data.get('model', {})
        
        self.bos_token_id = model_config['bos_token_id']
        self.eos_token_id = model_config['eos_token_id']
        self.pad_token_id = model_config['pad_token_id']
        
        # Get vocab_size from model-specific config
        self.vocab_size = get_model_specific_value(config_data, self.model_size, 'model', 'vocab_size')
        if self.vocab_size is None:
            self.vocab_size = model_config.get(
                'vocab_size', 
                max(self.bos_token_id, self.eos_token_id, self.pad_token_id) + 1
            )
        
        # Determine model path
        if model_path is None:
            model_path = get_model_specific_value(config_data, self.model_size, 'huggingface', 'qwen_model')
        
        print(f"[SketchDecoder] Using Qwen model: {model_path}")
        print(f"[SketchDecoder] Vocab size: {self.vocab_size}")
        
        # Get torch_dtype from kwargs or use default
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        
        config = AutoConfig.from_pretrained(
            model_path,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
        )
        
        self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype, 
            attn_implementation="sdpa",
            device_map="auto",
            ignore_mismatched_sizes=True
        )
        
        self.transformer.resize_token_embeddings(self.vocab_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass not included in open-source version")
