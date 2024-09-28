# Voa Audio Training Library

This repository provides a step-by-step guide to fine-tune the Whisper model using the `finetune.py` script. The script uses the Hugging Face Transformers library, Datasets, and Accelerate for distributed training across multiple GPUs with mixed precision.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Fine-Tuning Script](#running-the-fine-tuning-script)
- [Monitoring Training](#monitoring-training)
- [Additional Notes](#additional-notes)

## Prerequisites

Before you begin, ensure you have the following:

- **Operating System**: Linux (Ubuntu recommended)
- **Python Version**: Python 3.8 or higher
- **CUDA and NVIDIA Drivers**: Properly installed and configured for GPU usage
- **GPUs**: Access to multiple GPUs (e.g., 8 GPUs) for distributed training
- **Huggingface Key**: Required to download datasets and models from Voa's private Hub

## Installation

Follow these steps to set up your environment:

### 1. Clone the Repository

```bash
git clone https://github.com/Voa-Health/audio-trainer.git
cd audio-trainer
```

### 2. Install System-Level Dependencies

Update package lists and install `ffmpeg`:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### 3. Set Up a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

#### a. Install from `requirements.txt`

Ensure you have `pip` version 21.1 or higher. Then, install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt` Content:**

```text
git+https://github.com/huggingface/datasets.git@main#egg=datasets
git+https://github.com/huggingface/transformers.git@main#egg=transformers[torch]
librosa
evaluate>=0.3.0
jiwer
gradio
more-itertools
accelerate
tensorboard
```

#### b. Install Additional Packages (If Necessary)

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
```

*Replace `cu117` with your CUDA version.*

### 5. Verify Installation

Check that the packages are installed correctly:

```bash
python -c "import transformers; print(transformers.__version__)"
python -c "import datasets; print(datasets.__version__)"
```

## Configuration

### 1. Log in to Hugging Face CLI (If Pushing to Hub)

If you intend to push the fine-tuned model to the Hugging Face Hub, log in:

```bash
huggingface-cli login
```

Follow the prompts to authenticate.

### 2. Configure Accelerate

Set up `accelerate` for distributed training:

```bash
accelerate config
```

**Configuration Prompts:**

1. **In which compute environment are you running?**

   ```
   This machine
   ```

2. **Which type of machine are you using?**

   ```
   Multi-GPU
   ```

3. **How many different machines will you use (in total)?**

   ```
   1
   ```

4. **Do you want to use DeepSpeed?**

   ```
   No
   ```

5. **Do you want to use FP16 or BF16 (mixed precision)?**

   ```
   fp16
   ```

6. **What should be the batch size per GPU?**

   ```
   [Leave blank to use the script's default or specify if you want to override]
   ```

7. **Please enter the path to your training script**

   ```
   [Leave blank unless you want to set a default script]
   ```

8. **Do you wish to save this configuration to reuse it later?**

   ```
   Yes
   ```

This configuration ensures that `accelerate` is set up to use all available GPUs with mixed precision training.

## Running the Fine-Tuning Script

Use the following command to start the fine-tuning process:

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 finetune.py \
  --output_dir "./whisper-finetuned" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_steps 5000 \
  --logging_steps 25 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --seed 42 \
  --fp16 \
  --push_to_hub \
  --dataloader_num_workers 4
```

### Explanation of Command-Line Arguments

- `--output_dir`: Directory to save the fine-tuned model.
- `--num_train_epochs`: Number of training epochs.
- `--per_device_train_batch_size`: Training batch size per GPU.
- `--per_device_eval_batch_size`: Evaluation batch size per GPU.
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients.
- `--learning_rate`: Learning rate for the optimizer.
- `--max_steps`: Maximum number of training steps.
- `--logging_steps`: Interval (in steps) for logging training metrics.
- `--eval_steps`: Interval (in steps) for evaluation.
- `--save_steps`: Interval (in steps) for saving checkpoints.
- `--seed`: Random seed for reproducibility.
- `--fp16`: Enable mixed precision training.
- `--push_to_hub`: Push the fine-tuned model to Hugging Face Hub.
- `--dataloader_num_workers`: Number of subprocesses for data loading.

### Environment Variables (Optional)

Set environment variables to prevent potential NCCL errors:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

Include them when running the command:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch ...
```

## Monitoring Training

### 1. Console Output

The script provides detailed logs during training, including loss values and evaluation metrics.

### 2. TensorBoard

Since the script reports to TensorBoard, you can visualize training metrics:

```bash
tensorboard --logdir='./whisper-finetuned/runs'
```

Open the provided URL in your browser to monitor training progress.

### 3. System Monitoring

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Monitor CPU and memory usage:

```bash
htop
```

## Additional Notes

- **Dataset Preparation**: Ensure that your datasets (`train_dataset` and `eval_dataset`) are correctly formatted and accessible via the Hugging Face Datasets library.

- **Model and Processor**: The script uses the `pierreguillou/whisper-medium-portuguese` model by default. Modify the `--model_name_or_path` argument if you wish to use a different model.

- **Adjusting Batch Sizes**: The effective batch size is calculated as:

  ```
  Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * number of GPUs
  ```

  Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` based on your GPU memory.

- **Error Handling**: If you encounter errors related to batch sizes or data loading, ensure that your datasets are not using `IterableDataset` and that batch sizes are consistent.

- **Reproducibility**: Set the random seed (`--seed 42`) for reproducibility.

- **Push to Hub**: Ensure you're logged in to the Hugging Face CLI if using `--push_to_hub`.

- **Virtual Environment**: Using a virtual environment is recommended to manage dependencies and avoid conflicts.

## License

This project is licensed under the MIT License.

---

Feel free to modify the script and command-line arguments to suit your specific needs. If you have any questions or issues, please open an issue in the repository or reach out for support.
