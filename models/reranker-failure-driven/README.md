---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:928
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
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
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: failure driven val
      type: failure-driven-val
    metrics:
    - type: accuracy
      value: 0.8958333333333334
      name: Accuracy
    - type: accuracy_threshold
      value: -4.960968017578125
      name: Accuracy Threshold
    - type: f1
      value: 0.9056603773584906
      name: F1
    - type: f1_threshold
      value: -4.960968017578125
      name: F1 Threshold
    - type: precision
      value: 0.8275862068965517
      name: Precision
    - type: recall
      value: 1.0
      name: Recall
    - type: average_precision
      value: 0.8631066411238826
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
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
    ['Can structured and unstructured data be combined?', 'application for the user to read . RAG data pipeline flow The following workflow describes a high - level flow for a data pipeline that supplies grounding data for a RAG application . - Documents or other media are either pushed or pulled into a data pipeline . - The data pipeline processes each media file individually by completing the following steps : - Chunking : Breaks down the media file into semantically relevant parts that ideally have a single idea or concept . - Enrich chunks : Adds metadata fields that the pipeline creates based on the content in the chunks . The data pipeline categorizes the metadata into discrete fields , such as title , summary , and keywords . - Embed chunks : Uses an embedding model to vectorize the chunk and any other metadata fields that'],
    ['Can structured and unstructured data be combined?', "DeepEval ⭐ , an open - source LLM evaluation framework . Let ’ s get started . TL ; DR - RAG pipelines are made up of a retriever and a generator , both of which contribute to the quality of the final response . - RAG metrics measures either the retriever and generator in isolation , focusing on relevancy , hallucination , and retrieval . - Retriever metrics include : Contextual recall , precision , and relevancy , used for evaluating things like top - K values and embedding models . - Generator metrics include : Faithfulness and answer relevancy , used for evaluating the LLM and prompt template . - RAG metrics are generic , and you ' ll want to use at least one additional custom metric to tailor towards your use case . - Agentic RAG"],
    ['What are good retrieval metrics for RAG?', "are generic , and you ' ll want to use at least one additional custom metric to tailor towards your use case . - Agentic RAG requires additional metrics such as task completion . - DeepEval ( 100 % OS ⭐ https : / / github . com / confident - ai / deepeval ) allows anyone to implement SOTA RAG metrics in 5 lines of code . What is RAG Evaluation ? RAG evaluation is the process of using metrics such as answer relevancy , faithfulness , and contextual relevancy to test the quality of a RAG pipeline ’ s “ retriever ” and the “ generator ” separately to measure each component ’ s contribution to the final response quality To do this , RAG evaluation involves 5 key industry - standard metrics : - Answer Relevancy :"],
    ['What is the role of data labeling in RAG?', 'control over which sources are used , real - time data access , authorization to data , guardrails / safety / compliance , traceability / source citations , retrieval strategies , cost , tune each component independently of the others - Cost - effective compared to alternatives like training / re - training your own model , fine - tuning , or stuffing the context window : foundation models are costly to produce and require specialized knowledge to create , as is fine - tuning ; the larger the context sent to the model , the higher the cost RAG in support of agentic workflows But this traditional RAG approach is simple , often with a vector database and a one - shot prompt with context sent to the model to generate output . With the rise of AI agents'],
    ['How can batching improve performance?', "eliminate differences that don ' t affect the meaning of the content . This method supports closeness matches . - Augment chunks . Consider augmenting your chunk data with common metadata fields and understand their potential uses in search . Learn about commonly used tools or techniques for generating metadata content . During the embedding phase , you should : - Understand the importance of the embedding model . An embedding model can significantly affect the relevancy of your vector search results . - Choose the right embedding model for your use case . - Evaluate embedding models . Evaluate embedding models by visualizing embeddings and calculating embedding distances . During the information retrieval phase , you should : - Create a search index . Apply the appropriate vector search configurations to your vector fields . - Understand search options"],
]
scores = model.predict(pairs)
print(scores)
# [ 8.1228  7.2082 -7.7935 -5.3628  7.9509]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Can structured and unstructured data be combined?',
    [
        'application for the user to read . RAG data pipeline flow The following workflow describes a high - level flow for a data pipeline that supplies grounding data for a RAG application . - Documents or other media are either pushed or pulled into a data pipeline . - The data pipeline processes each media file individually by completing the following steps : - Chunking : Breaks down the media file into semantically relevant parts that ideally have a single idea or concept . - Enrich chunks : Adds metadata fields that the pipeline creates based on the content in the chunks . The data pipeline categorizes the metadata into discrete fields , such as title , summary , and keywords . - Embed chunks : Uses an embedding model to vectorize the chunk and any other metadata fields that',
        "DeepEval ⭐ , an open - source LLM evaluation framework . Let ’ s get started . TL ; DR - RAG pipelines are made up of a retriever and a generator , both of which contribute to the quality of the final response . - RAG metrics measures either the retriever and generator in isolation , focusing on relevancy , hallucination , and retrieval . - Retriever metrics include : Contextual recall , precision , and relevancy , used for evaluating things like top - K values and embedding models . - Generator metrics include : Faithfulness and answer relevancy , used for evaluating the LLM and prompt template . - RAG metrics are generic , and you ' ll want to use at least one additional custom metric to tailor towards your use case . - Agentic RAG",
        "are generic , and you ' ll want to use at least one additional custom metric to tailor towards your use case . - Agentic RAG requires additional metrics such as task completion . - DeepEval ( 100 % OS ⭐ https : / / github . com / confident - ai / deepeval ) allows anyone to implement SOTA RAG metrics in 5 lines of code . What is RAG Evaluation ? RAG evaluation is the process of using metrics such as answer relevancy , faithfulness , and contextual relevancy to test the quality of a RAG pipeline ’ s “ retriever ” and the “ generator ” separately to measure each component ’ s contribution to the final response quality To do this , RAG evaluation involves 5 key industry - standard metrics : - Answer Relevancy :",
        'control over which sources are used , real - time data access , authorization to data , guardrails / safety / compliance , traceability / source citations , retrieval strategies , cost , tune each component independently of the others - Cost - effective compared to alternatives like training / re - training your own model , fine - tuning , or stuffing the context window : foundation models are costly to produce and require specialized knowledge to create , as is fine - tuning ; the larger the context sent to the model , the higher the cost RAG in support of agentic workflows But this traditional RAG approach is simple , often with a vector database and a one - shot prompt with context sent to the model to generate output . With the rise of AI agents',
        "eliminate differences that don ' t affect the meaning of the content . This method supports closeness matches . - Augment chunks . Consider augmenting your chunk data with common metadata fields and understand their potential uses in search . Learn about commonly used tools or techniques for generating metadata content . During the embedding phase , you should : - Understand the importance of the embedding model . An embedding model can significantly affect the relevancy of your vector search results . - Choose the right embedding model for your use case . - Evaluate embedding models . Evaluate embedding models by visualizing embeddings and calculating embedding distances . During the information retrieval phase , you should : - Create a search index . Apply the appropriate vector search configurations to your vector fields . - Understand search options",
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
| accuracy              | 0.8958     |
| accuracy_threshold    | -4.961     |
| f1                    | 0.9057     |
| f1_threshold          | -4.961     |
| precision             | 0.8276     |
| recall                | 1.0        |
| **average_precision** | **0.8631** |

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

* Size: 928 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 928 samples:
  |         | sentence_0                                                                        | sentence_1                                                                            | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                                | float                                                         |
  | details | <ul><li>min: 7 tokens</li><li>mean: 10.46 tokens</li><li>max: 13 tokens</li></ul> | <ul><li>min: 142 tokens</li><li>mean: 159.45 tokens</li><li>max: 257 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | label            |
  |:---------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Can structured and unstructured data be combined?</code> | <code>application for the user to read . RAG data pipeline flow The following workflow describes a high - level flow for a data pipeline that supplies grounding data for a RAG application . - Documents or other media are either pushed or pulled into a data pipeline . - The data pipeline processes each media file individually by completing the following steps : - Chunking : Breaks down the media file into semantically relevant parts that ideally have a single idea or concept . - Enrich chunks : Adds metadata fields that the pipeline creates based on the content in the chunks . The data pipeline categorizes the metadata into discrete fields , such as title , summary , and keywords . - Embed chunks : Uses an embedding model to vectorize the chunk and any other metadata fields that</code> | <code>1.0</code> |
  | <code>Can structured and unstructured data be combined?</code> | <code>DeepEval ⭐ , an open - source LLM evaluation framework . Let ’ s get started . TL ; DR - RAG pipelines are made up of a retriever and a generator , both of which contribute to the quality of the final response . - RAG metrics measures either the retriever and generator in isolation , focusing on relevancy , hallucination , and retrieval . - Retriever metrics include : Contextual recall , precision , and relevancy , used for evaluating things like top - K values and embedding models . - Generator metrics include : Faithfulness and answer relevancy , used for evaluating the LLM and prompt template . - RAG metrics are generic , and you ' ll want to use at least one additional custom metric to tailor towards your use case . - Agentic RAG</code>                                        | <code>1.0</code> |
  | <code>What are good retrieval metrics for RAG?</code>          | <code>are generic , and you ' ll want to use at least one additional custom metric to tailor towards your use case . - Agentic RAG requires additional metrics such as task completion . - DeepEval ( 100 % OS ⭐ https : / / github . com / confident - ai / deepeval ) allows anyone to implement SOTA RAG metrics in 5 lines of code . What is RAG Evaluation ? RAG evaluation is the process of using metrics such as answer relevancy , faithfulness , and contextual relevancy to test the quality of a RAG pipeline ’ s “ retriever ” and the “ generator ” separately to measure each component ’ s contribution to the final response quality To do this , RAG evaluation involves 5 key industry - standard metrics : - Answer Relevancy :</code>                                                                  | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 5
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 5
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
- `per_device_eval_batch_size`: 16
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
| 1.0   | 58   | 0.7885                               |
| 2.0   | 116  | 0.8855                               |
| 3.0   | 174  | 0.8631                               |
| 4.0   | 232  | 0.8686                               |
| 5.0   | 290  | 0.8631                               |


### Training Time
- **Training**: 16.1 seconds

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