---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:598
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L12-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L12-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: failure driven val
      type: failure-driven-val
    metrics:
    - type: accuracy
      value: 0.6818181818181818
      name: Accuracy
    - type: accuracy_threshold
      value: 4.245023727416992
      name: Accuracy Threshold
    - type: f1
      value: 0.7333333333333334
      name: F1
    - type: f1_threshold
      value: -5.949812889099121
      name: F1 Threshold
    - type: precision
      value: 0.5789473684210527
      name: Precision
    - type: recall
      value: 1.0
      name: Recall
    - type: average_precision
      value: 0.7320574162679427
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L12-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) <!-- at revision 7b0235231ca2674cb8ca8f022859a6eba2b1c968 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['What is multi-hop retrieval in RAG?', 'as they move through downstream reasoning steps in an agentic RAG system . TL ; DR - Vector Retrieval - Augmented Generation ( RAG ) is a strong default for single - hop semantic queries , with advantages in simple fact retrieval , response time , and lower implementation cost . - Graph RAG is strongest for multi - hop , relationship - intensive queries where explicit entity connections matter . - Hybrid approaches can outperform either method on mixed workloads , but naive combinations can reduce context relevance . - Fresh data , preserved permissions , and a reliable upstream pipeline matter as much as retrieval architecture for AI agents in production . How Do Graph RAG and Vector RAG Compare ? Vector RAG splits documents into chunks , embeds each chunk into a vector embedding store , and'],
    ['Can structured and unstructured data be combined?', 'as they move through downstream reasoning steps in an agentic RAG system . TL ; DR - Vector Retrieval - Augmented Generation ( RAG ) is a strong default for single - hop semantic queries , with advantages in simple fact retrieval , response time , and lower implementation cost . - Graph RAG is strongest for multi - hop , relationship - intensive queries where explicit entity connections matter . - Hybrid approaches can outperform either method on mixed workloads , but naive combinations can reduce context relevance . - Fresh data , preserved permissions , and a reliable upstream pipeline matter as much as retrieval architecture for AI agents in production . How Do Graph RAG and Vector RAG Compare ? Vector RAG splits documents into chunks , embeds each chunk into a vector embedding store , and'],
    ['Can structured and unstructured data be combined?', 'online evaluations , as part of production monitoring . It also depends on how much labeled data you have to design the test . Let ’ s take a look at 3 different approaches . First things first : retrieval isn ’ t a new problem . It ’ s the same task behind every search bar – from e - commerce sites to Google to internal company portals . It ’ s a classic machine learning use case , and there are well - established evaluation methods we can reuse for LLM - powered RAG setups . To apply them , you need a ground truth dataset – your custom retrieval benchmark . For each query , you define the correct sources that contain the answer – these could be document IDs , chunk IDs , or links .'],
    ['Can structured and unstructured data be combined?', 'like mean - reciprocal rank ( MRR ) , hit - rate , precision , and more . The core retrieval evaluation steps revolve around the following : - Dataset generation : Given an unstructured text corpus , synthetically generate ( question , context ) pairs . - Retrieval Evaluation : Given a retriever and a set of questions , evaluate retrieved results using ranking metrics . Integrations Section titled “ Integrations ” We also integrate with community evaluation tools . - UpTrain - Tonic Validate ( Includes Web UI for visualizing results ) - DeepEval - Ragas - RAGChecker - Cleanlab Usage Pattern Section titled “ Usage Pattern ” For full usage details , see the usage pattern below . Modules Section titled “ Modules ” Notebooks with usage of these components can be found in the module guides , see the usage pattern below . Modules Section titled “ Modules ” Notebooks with usage of these components can be found in the module guides . Evaluating with LabelledRagDataset ’ s Section titled “ Evaluating with LabelledRagDataset ’ s ” For details on how to perform evaluation of a RAG system with various evaluation datasets , called LabelledRagDataset ’ s see below :'],
    ['What is approximate nearest neighbor (ANN) search?', 'solutions to the different challenges and serving as a guide to systematically developing such applications . References & Citations export BibTeX citation Loading . . . Bibliographic and Citation Tools Bibliographic Explorer ( What is the Explorer ? ) Connected Papers ( What is Connected Papers ? ) Litmaps ( What is Litmaps ? ) scite Smart Citations ( What are Smart Citations ? ) Code , Data and Media Associated with this Article alphaXiv ( What is alphaXiv ? ) CatalyzeX Code Finder for Papers ( What is CatalyzeX ? ) DagsHub ( What is DagsHub ? ) Gotit . pub ( What is GotitPub ? ) Hugging Face ( What is Huggingface ? ) Papers with Code ( What is Papers with Code ? ) ScienceCast ( What is ScienceCast ? ) Demos Recommenders and Search Tools Influence'],
]
scores = model.predict(pairs)
print(scores)
# [ -4.8315   7.8104 -10.1379  -8.5325  -9.7408]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What is multi-hop retrieval in RAG?',
    [
        'as they move through downstream reasoning steps in an agentic RAG system . TL ; DR - Vector Retrieval - Augmented Generation ( RAG ) is a strong default for single - hop semantic queries , with advantages in simple fact retrieval , response time , and lower implementation cost . - Graph RAG is strongest for multi - hop , relationship - intensive queries where explicit entity connections matter . - Hybrid approaches can outperform either method on mixed workloads , but naive combinations can reduce context relevance . - Fresh data , preserved permissions , and a reliable upstream pipeline matter as much as retrieval architecture for AI agents in production . How Do Graph RAG and Vector RAG Compare ? Vector RAG splits documents into chunks , embeds each chunk into a vector embedding store , and',
        'as they move through downstream reasoning steps in an agentic RAG system . TL ; DR - Vector Retrieval - Augmented Generation ( RAG ) is a strong default for single - hop semantic queries , with advantages in simple fact retrieval , response time , and lower implementation cost . - Graph RAG is strongest for multi - hop , relationship - intensive queries where explicit entity connections matter . - Hybrid approaches can outperform either method on mixed workloads , but naive combinations can reduce context relevance . - Fresh data , preserved permissions , and a reliable upstream pipeline matter as much as retrieval architecture for AI agents in production . How Do Graph RAG and Vector RAG Compare ? Vector RAG splits documents into chunks , embeds each chunk into a vector embedding store , and',
        'online evaluations , as part of production monitoring . It also depends on how much labeled data you have to design the test . Let ’ s take a look at 3 different approaches . First things first : retrieval isn ’ t a new problem . It ’ s the same task behind every search bar – from e - commerce sites to Google to internal company portals . It ’ s a classic machine learning use case , and there are well - established evaluation methods we can reuse for LLM - powered RAG setups . To apply them , you need a ground truth dataset – your custom retrieval benchmark . For each query , you define the correct sources that contain the answer – these could be document IDs , chunk IDs , or links .',
        'like mean - reciprocal rank ( MRR ) , hit - rate , precision , and more . The core retrieval evaluation steps revolve around the following : - Dataset generation : Given an unstructured text corpus , synthetically generate ( question , context ) pairs . - Retrieval Evaluation : Given a retriever and a set of questions , evaluate retrieved results using ranking metrics . Integrations Section titled “ Integrations ” We also integrate with community evaluation tools . - UpTrain - Tonic Validate ( Includes Web UI for visualizing results ) - DeepEval - Ragas - RAGChecker - Cleanlab Usage Pattern Section titled “ Usage Pattern ” For full usage details , see the usage pattern below . Modules Section titled “ Modules ” Notebooks with usage of these components can be found in the module guides , see the usage pattern below . Modules Section titled “ Modules ” Notebooks with usage of these components can be found in the module guides . Evaluating with LabelledRagDataset ’ s Section titled “ Evaluating with LabelledRagDataset ’ s ” For details on how to perform evaluation of a RAG system with various evaluation datasets , called LabelledRagDataset ’ s see below :',
        'solutions to the different challenges and serving as a guide to systematically developing such applications . References & Citations export BibTeX citation Loading . . . Bibliographic and Citation Tools Bibliographic Explorer ( What is the Explorer ? ) Connected Papers ( What is Connected Papers ? ) Litmaps ( What is Litmaps ? ) scite Smart Citations ( What are Smart Citations ? ) Code , Data and Media Associated with this Article alphaXiv ( What is alphaXiv ? ) CatalyzeX Code Finder for Papers ( What is CatalyzeX ? ) DagsHub ( What is DagsHub ? ) Gotit . pub ( What is GotitPub ? ) Hugging Face ( What is Huggingface ? ) Papers with Code ( What is Papers with Code ? ) ScienceCast ( What is ScienceCast ? ) Demos Recommenders and Search Tools Influence',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `failure-driven-val`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.6818     |
| accuracy_threshold    | 4.245      |
| f1                    | 0.7333     |
| f1_threshold          | -5.9498    |
| precision             | 0.5789     |
| recall                | 1.0        |
| **average_precision** | **0.7321** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 598 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 598 samples:
  |         | sentence_0                                                                        | sentence_1                                                                            | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                                | float                                                         |
  | details | <ul><li>min: 7 tokens</li><li>mean: 10.54 tokens</li><li>max: 12 tokens</li></ul> | <ul><li>min: 144 tokens</li><li>mean: 164.59 tokens</li><li>max: 265 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:---------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is multi-hop retrieval in RAG?</code>               | <code>as they move through downstream reasoning steps in an agentic RAG system . TL ; DR - Vector Retrieval - Augmented Generation ( RAG ) is a strong default for single - hop semantic queries , with advantages in simple fact retrieval , response time , and lower implementation cost . - Graph RAG is strongest for multi - hop , relationship - intensive queries where explicit entity connections matter . - Hybrid approaches can outperform either method on mixed workloads , but naive combinations can reduce context relevance . - Fresh data , preserved permissions , and a reliable upstream pipeline matter as much as retrieval architecture for AI agents in production . How Do Graph RAG and Vector RAG Compare ? Vector RAG splits documents into chunks , embeds each chunk into a vector embedding store , and</code> | <code>0.0</code> |
  | <code>Can structured and unstructured data be combined?</code> | <code>as they move through downstream reasoning steps in an agentic RAG system . TL ; DR - Vector Retrieval - Augmented Generation ( RAG ) is a strong default for single - hop semantic queries , with advantages in simple fact retrieval , response time , and lower implementation cost . - Graph RAG is strongest for multi - hop , relationship - intensive queries where explicit entity connections matter . - Hybrid approaches can outperform either method on mixed workloads , but naive combinations can reduce context relevance . - Fresh data , preserved permissions , and a reliable upstream pipeline matter as much as retrieval architecture for AI agents in production . How Do Graph RAG and Vector RAG Compare ? Vector RAG splits documents into chunks , embeds each chunk into a vector embedding store , and</code> | <code>1.0</code> |
  | <code>Can structured and unstructured data be combined?</code> | <code>online evaluations , as part of production monitoring . It also depends on how much labeled data you have to design the test . Let ’ s take a look at 3 different approaches . First things first : retrieval isn ’ t a new problem . It ’ s the same task behind every search bar – from e - commerce sites to Google to internal company portals . It ’ s a classic machine learning use case , and there are well - established evaluation methods we can reuse for LLM - powered RAG setups . To apply them , you need a ground truth dataset – your custom retrieval benchmark . For each query , you define the correct sources that contain the answer – these could be document IDs , chunk IDs , or links .</code>                                                                                                                | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | failure-driven-val_average_precision |
|:-----:|:----:|:------------------------------------:|
| 1.0   | 75   | 0.4463                               |
| 2.0   | 150  | 0.7257                               |
| 3.0   | 225  | 0.7321                               |
| 4.0   | 300  | 0.7321                               |


### Training Time
- **Training**: 18.2 seconds

### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.7.1+cu126
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->