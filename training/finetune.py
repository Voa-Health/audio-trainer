#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import evaluate
import logging
from functools import partial
from io import BytesIO
import torchaudio
import random
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
    logging as transformers_logging,  # Import the transformers logging module
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset, Audio, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from typing import Any, Dict, List, Union
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

# Custom Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
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
    parser.add_argument('--model_name_or_path', type=str, default='openai/whisper-large-v3-turbo', help='Pretrained model identifier')
    parser.add_argument('--output_dir', type=str, default='./whisper-finetuned', help='Directory to save the fine-tuned model')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for the learning rate scheduler')
    parser.add_argument('--num_cycles', type=float, default=0.5, help='Number of cycles for the cosine scheduler')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=25, help='Logging interval')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--save_steps', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
    parser.add_argument('--push_to_hub', action='store_true', help='Push the model to Hugging Face Hub')
    # Add argument for testing mode
    parser.add_argument('--test_mode', type=bool, default=False, help='Run the script in test mode with a small dataset and 10 steps for quick validation')

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
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # Update model configuration
    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate, language="sinhalese", task="transcribe", use_cache=True
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
    )

    # Load your datasets directly from Huggingface
    logger.info("Loading datasets from Huggingface...")

    voa_dataset = load_dataset("voa-engines/voa_audios_3_0", "validated", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True))
    resample_pt = load_dataset("voa-engines/common_voice_resample", "pt", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True)).rename_column("sentence", "transcription")
    resample_en = load_dataset("voa-engines/common_voice_resample", "en", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True)).rename_column("sentence", "transcription")
    resample_es = load_dataset("voa-engines/common_voice_resample", "es", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True)).rename_column("sentence", "transcription")
    resample_it = load_dataset("voa-engines/common_voice_resample", "it", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True)).rename_column("sentence", "transcription")
    resample_ro = load_dataset("voa-engines/common_voice_resample", "ro", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True)).rename_column("sentence", "transcription")
    resample_fr = load_dataset("voa-engines/common_voice_resample", "fr", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True)).rename_column("sentence", "transcription")

    train_dataset = concatenate_datasets([voa_dataset, resample_pt, resample_en, resample_es, resample_fr, resample_it, resample_ro])


    test_dataset = load_dataset("voa-engines/voa_audios_eval", split="train", num_proc=32).cast_column("audio", Audio(decode=False, sampling_rate=16000, mono=True))

    def prepare_dataset(batch):
        try:
            # Load MP3 audio bytes
            audio_bytes = batch['audio']['bytes']
            
            # Check if audio_bytes is empty or None
            if not audio_bytes:
                raise ValueError("Audio bytes are missing or null")
    
            # Wrap the audio bytes in a BytesIO object
            audio_file = BytesIO(audio_bytes)
    
            # Try to load the audio bytes to check if the audio is valid
            try:
                torchaudio.set_audio_backend("ffmpeg")  # Ensure FFmpeg is installed
                # Attempt to load the audio
                incoming_waveform, sample_rate = torchaudio.load(audio_file, format='mp3')
            except Exception as e:
                raise ValueError(f"Failed to load MP3 audio with torchaudio: {e}")
    
            # Optional resampling to 16kHz if required
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                incoming_waveform = resampler(incoming_waveform)
    
            # Compute input features using your processor's feature extractor
            try:
                batch["input_features"] = processor.feature_extractor(
                    incoming_waveform.squeeze().numpy(), sampling_rate=target_sample_rate
                ).input_features[0]
            except Exception as e:
                raise ValueError(f"Error extracting features: {e}")
            
            try:
                # Compute the input length in seconds
                batch["input_length"] = incoming_waveform.size(1) / target_sample_rate
            except Exception as e:
                raise ValueError(f"Error computing input length: {e}")
            
            # Process transcription and labels
            # Try both 'transcription' and 'sentence' keys if applicable
            transcription = batch.get("transcription")

            # Encode target text to label ids
            try:
                batch["labels"] = processor.tokenizer(transcription).input_ids
            except Exception as e:
                raise ValueError(f"Error tokenizing transcription: {e}")
    
            return batch

        except ValueError as ve:
            print(f"Skipping corrupted data: {ve}")
            return None  # Returning None will exclude this batch from the final dataset\
        
    max_input_length = 30.0

    def is_audio_in_length_range(length):
        return length < max_input_length

    def is_transcription_in_length_range(transcription):
        return len(transcription) < 688

    train_dataset = train_dataset.filter(
        is_transcription_in_length_range,
        input_columns=["transcription"],
        num_proc=32
    )

    test_dataset = test_dataset.filter(
        is_transcription_in_length_range,
        input_columns=["transcription"],
        num_proc=32
    )

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=['audio', 'transcription'],  # Remove unnecessary columns after processing
        batched=False,
        num_proc=16
    ).with_format("torch")

    test_dataset = test_dataset.map(
        prepare_dataset,
        remove_columns=['audio', 'transcription'],  # Remove unnecessary columns after processing
        batched=False,
        num_proc=16
    ).with_format("torch")

    train_dataset = train_dataset.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
        num_proc=32
    )

    test_dataset = test_dataset.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
        num_proc=32
    )

    # For testing purposes, select a random sample of the dataset based on its number of rows
    if args.test_mode:
        # Select random indices from the train dataset based on its total number of rows
        train_num_rows = train_dataset.num_rows
        eval_num_rows = eval_dataset.num_rows

        # Generate random indices
        train_indices = random.sample(range(train_num_rows), k=180)
        eval_indices = random.sample(range(eval_num_rows), k=20)

        # Use the random indices to select a subset
        train_dataset = train_dataset.select(train_indices)
        eval_dataset = eval_dataset.select(eval_indices)

        # Set the number of steps and epochs for quick testing
        args.num_train_epochs = 1  # Run only 1 epoch
        
        # Disable pushing to Hugging Face Hub when in test mode
        args.push_to_hub = False

    # Define evaluation metric
    logger.info("Loading evaluation metric...")
    metric = evaluate.load("wer")

    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # compute orthographic wer
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

        # compute normalised WER
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

        return {"wer_ortho": wer_ortho, "wer": wer}

    optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        # Remove or set max_steps to -1 to use num_train_epochs
        max_steps=-1,  # Set to -1 so it doesn't override num_train_epochs
        fp16=args.fp16,
        eval_strategy="steps",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        logging_dir=args.output_dir,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        dataloader_num_workers=8,
        greater_is_better=False,
        seed=args.seed,
        report_to=["tensorboard"],
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

        # Create the cosine scheduler
    lr_constant_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
    )

            # Create the cosine scheduler
    lr_cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
        num_cycles=args.num_cycles
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        optimizers=(optimizer, lr_constant_scheduler),
        callbacks=[LoggingCallback()],
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the trained model and processor
    logger.info("Training completed. Saving model and processor.")
    trainer.save_model()

if __name__ == "__main__":
    main()
