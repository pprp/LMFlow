#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Merge base model and lora model into a full model.
"""

import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional


from lmflow.models.auto_model import AutoModel


def main():
    model_args = {
        'path_after_merge': './output_models/merged_lora', 
        
    }

    model_args.use_lora = True
    model = AutoModel.get_model(model_args)
    model.merge_lora_weights()
    model.save(merge_lora_args.output_model_path, save_full_model=True)


if __name__ == '__main__':
    main()
