#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import evaluate
import logging
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
    logging as transformers_logging,  # Import the transformers logging module
)
from datasets import load_dataset, Audio
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Custom Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Define a custom callback for logging
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            logs = logs or {}
            logging.info(f"Step {state.global_step}: {logs}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model for speech recognition")
    parser.add_argument('--model_name_or_path', type=str, default='pierreguillou/whisper-medium-portuguese', help='Pretrained model identifier')
    parser.add_argument('--output_dir', type=str, default='./whisper-finetuned', help='Directory to save the fine-tuned model')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=5000, help='Total number of training steps')
    parser.add_argument('--logging_steps', type=int, default=25, help='Logging interval')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--save_steps', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
    parser.add_argument('--push_to_hub', action='store_true', help='Push the model to Hugging Face Hub')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Number of subprocesses for data loading')
    args = parser.parse_args()

    # Set the seed for reproducibility
    set_seed(args.seed)

    # Set environment variables to prevent NCCL errors
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if int(os.environ.get("RANK", 0)) == 0 else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set transformers logging level
    transformers_logging.set_verbosity_info()
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()

    # Load the processor and the model
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Update model configuration
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Important for gradient checkpointing

    # Load your datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset("voa-engines/features_dataset_v1", split="train")
    eval_dataset = load_dataset("voa-engines/features_dataset_v1", split="validation")

    logger.info("Shuffling train dataset...")
    train_dataset = train_dataset.with_format(None)
    iterable_dataset = train_dataset.to_iterable_dataset(num_shards=128)
    train_dataset = iterable_dataset.shuffle(seed=42, buffer_size=400)

    # Define evaluation metric
    logger.info("Loading evaluation metric...")
    metric = evaluate.load("wer")

    # Define normalizer if needed
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    do_normalize_eval = True  # Set to False if you don't want to normalize during evaluation

    # Define compute_metrics function
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]
            # Filtering step to only evaluate samples that have non-empty references
            pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        max_steps=args.max_steps,
        fp16=args.fp16,
        eval_strategy="steps",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        logging_dir=args.output_dir,
        logging_strategy="steps",
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=args.seed,
        report_to=["tensorboard"],
        push_to_hub=args.push_to_hub,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # Removed 'logging_level' parameter
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
        callbacks=[LoggingCallback()],
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the trained model and processor
    logger.info("Training completed. Saving model and processor.")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()