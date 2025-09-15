#!/usr/bin/env python3
"""
Setup script for LLaMA Factory
Configures the environment for fine-tuning LLMs
"""

import os
import subprocess
import sys
from pathlib import Path

def install_llama_factory():
    """Install LLaMA Factory and dependencies"""
    print("Installing LLaMA Factory...")
    
    try:
        # Clone LLaMA Factory repository
        subprocess.run([
            "git", "clone", 
            "https://github.com/hiyouga/LLaMA-Factory.git",
            "llama_factory"
        ], check=True)
        
        # Change to LLaMA Factory directory
        os.chdir("llama_factory")
        
        # Install dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        
        print("✅ LLaMA Factory installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing LLaMA Factory: {e}")
        return False

def create_training_config():
    """Create training configuration for TOEFL/IELTS model"""
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "dataset": "toefl_ielts_dataset",
        "template": "default",
        "finetuning_type": "lora",
        "lora_target": "all",
        "output_dir": "./saves/toefl_ielts_model",
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "max_samples": 1000,
        "cutoff_len": 1024,
        "val_size": 0.1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5.0e-5,
        "num_train_epochs": 3.0,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 1000,
        "warmup_steps": 0,
        "fp16": True,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "ddp_timeout": 180000000,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "group_by_length": False,
        "report_to": "none"
    }
    
    # Save configuration
    import json
    with open("toefl_ielts_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training configuration created!")

def create_dataset_format():
    """Create dataset format for TOEFL/IELTS training"""
    dataset_format = {
        "instruction": "You are an AI tutor helping students prepare for TOEFL/IELTS. Provide helpful, accurate, and educational responses.",
        "input": "",
        "output": ""
    }
    
    # Example training data format
    training_examples = [
        {
            "instruction": "Correct the grammar error in this sentence: 'I am go to school yesterday.'",
            "input": "I am go to school yesterday.",
            "output": "I went to school yesterday. The correct form uses the past tense 'went' instead of 'am go' because the action happened yesterday."
        },
        {
            "instruction": "Explain the difference between 'affect' and 'effect'.",
            "input": "",
            "output": "'Affect' is a verb meaning to influence or change something. 'Effect' is a noun meaning the result or consequence of an action. Example: The rain will affect the outdoor event, and the effect will be a smaller turnout."
        },
        {
            "instruction": "Help improve this IELTS writing task 2 introduction: 'Nowadays, technology is very important.'",
            "input": "Nowadays, technology is very important.",
            "output": "Here's an improved version: 'In the contemporary world, technology has become increasingly significant, fundamentally transforming various aspects of human life and society.' This version is more sophisticated and specific."
        }
    ]
    
    # Save dataset format
    import json
    with open("toefl_ielts_dataset.json", "w") as f:
        json.dump(training_examples, f, indent=2)
    
    print("✅ Dataset format created!")

def setup_web_ui():
    """Set up LLaMA Factory web UI"""
    print("Setting up LLaMA Factory Web UI...")
    
    # Create a simple script to start the web UI
    web_ui_script = """#!/bin/bash
cd llama_factory
python -m llamafactory.webui.interface
"""
    
    with open("start_webui.sh", "w") as f:
        f.write(web_ui_script)
    
    # Make it executable
    os.chmod("start_webui.sh", 0o755)
    
    print("✅ Web UI setup complete!")
    print("Run './start_webui.sh' to start the web interface")

def main():
    """Main setup function"""
    print("🚀 Setting up LLaMA Factory for TOEFL/IELTS AI Teaching Platform")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Install LLaMA Factory
    if install_llama_factory():
        # Create configurations
        create_training_config()
        create_dataset_format()
        setup_web_ui()
        
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the data collection script: python scripts/data_collection.py")
        print("2. Start the web UI: ./start_webui.sh")
        print("3. Upload your dataset and start training!")
    else:
        print("❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()