import pickle
# from datasets import load_dataset, Dataset
from transformers import DebertaV2Config
from transformers import DebertaV2ForMaskedLM
from pathlib import Path
import logging
from transformers import DebertaV2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset


tokenizer = DebertaV2Tokenizer.from_pretrained("./")
logging.basicConfig(level=logging.INFO)

config = DebertaV2Config(vocab_size=265, 
                      hidden_size=600,
                      num_attention_heads=12,
                      num_hidden_layers=12,
                      intermediate_size=512,
                      pad_token_id=3
                      )

model = DebertaV2ForMaskedLM(config=config)

# Resize token embedding to tokenizer
model.resize_token_embeddings(len(tokenizer))

dataset_train = Dataset.load_from_disk('dataset_tokenized_all/train')
dataset_test = Dataset.load_from_disk('dataset_tokenized_all/test')

dataset_train.set_format(type='torch', columns=['input_ids'])
dataset_test.set_format(type='torch', columns=['input_ids'])


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./model/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=30,
    per_device_eval_batch_size=30,
    save_steps=5_000,
    save_total_limit=1,
    fp16=True,
    logging_steps=1_000,
    prediction_loss_only=True,
    # disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
)

a = trainer.train()
trainer.save_model("./model_final/")



