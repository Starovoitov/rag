---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:549
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
      value: 0.8360655737704918
      name: Accuracy
    - type: accuracy_threshold
      value: 1.4130926132202148
      name: Accuracy Threshold
    - type: f1
      value: 0.8378378378378378
      name: F1
    - type: f1_threshold
      value: -0.332133948802948
      name: F1 Threshold
    - type: precision
      value: 0.7209302325581395
      name: Precision
    - type: recall
      value: 1.0
      name: Recall
    - type: average_precision
      value: 0.9099774943735934
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
    ['What is context stuffing?', 'query , it returns a few relevant snippets from your knowledge base . These snippets will then be fed to the Reader Model to help it generate its answer . So our objective here is , given a user question , to find the most relevant snippets from our knowledge base to answer that question . This is a wide objective , it leaves open some questions . How many snippets should we retrieve ? This parameter will be named top_k . How long should these snippets be ? This is called the chunk size . There ’ s no one - size - fits - all answers , but here are a few elements : - 🔀 Your chunk size is allowed to vary from one snippet to the other . - Since there will always be some noise'],
    ['How can batching improve performance?', 'elements : - 🔀 Your chunk size is allowed to vary from one snippet to the other . - Since there will always be some noise in your retrieval , increasing the top_k increases the chance to get relevant elements in your retrieved snippets . 🎯 Shooting more arrows increases your probability of hitting your target . - Meanwhile , the summed length of your retrieved documents should not be too high : for instance , for most current models 16k tokens will probably drown your Reader model in information due to Lost - in - the - middle phenomenon . 🎯 Give your reader model only the most relevant insights , not a huge pile of books ! In this notebook , we use Langchain library since it offers a huge variety of options for vector databases and allows'],
    ['How do you ensure consistency in outputs?', ', context , response , and combine these with LLM calls . These evaluation modules are in the following forms : - Correctness : Whether the generated answer matches that of the reference answer given the query ( requires labels ) . - Semantic Similarity Whether the predicted answer is semantically similar to the reference answer ( requires labels ) . - Faithfulness : Evaluates if the answer is faithful to the retrieved contexts ( in other words , whether if there ’ s hallucination ) . - Context Relevancy : Whether retrieved context is relevant to the query . - Answer Relevancy : Whether the generated answer is relevant to the query . - Guideline Adherence : Whether the predicted answer adheres to specific guidelines . Question Generation Section titled “ Question Generation ” In addition to evaluating queries'],
    ['What is context stuffing?', 'The RAG Triad ¶ RAGs have become the standard architecture for providing LLMs with context in order to avoid hallucinations . However , even RAGs can suffer from hallucination , as is often the case when the retrieval fails to retrieve sufficient context or even retrieves irrelevant context that is then weaved into the LLM ’ s response . TruEra has innovated the RAG triad to evaluate for hallucinations along each edge of the RAG architecture , shown below : The RAG triad is made up of 3 evaluations : context relevance , groundedness and answer relevance . Satisfactory evaluations on each provides us confidence that our LLM app is free from hallucination . Context Relevance ¶ The first step of any RAG application is retrieval ; to verify the quality of our retrieval , we want to make sure'],
    ['How do you ensure consistency in outputs?', ', users need to view its output as trustworthy . RAG models can include citations to the knowledge sources in their external data as part of their responses . When RAG models cite their sources , human users can verify those outputs to confirm accuracy while consulting the cited works for follow - up clarification and additional information . Corporate data storage is often a complex and siloed maze . RAG responses with citations point users directly toward the materials they need . Access to more data means that one model can handle a wider range of prompts . Enterprises can optimize models and gain more value from them by broadening their knowledge bases , in turn expanding the contexts in which those models generate reliable results . By combining generative AI with retrieval systems , RAG models can retrieve'],
]
scores = model.predict(pairs)
print(scores)
# [ 5.2533  7.9562  7.0701 -4.5685 -3.6864]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What is context stuffing?',
    [
        'query , it returns a few relevant snippets from your knowledge base . These snippets will then be fed to the Reader Model to help it generate its answer . So our objective here is , given a user question , to find the most relevant snippets from our knowledge base to answer that question . This is a wide objective , it leaves open some questions . How many snippets should we retrieve ? This parameter will be named top_k . How long should these snippets be ? This is called the chunk size . There ’ s no one - size - fits - all answers , but here are a few elements : - 🔀 Your chunk size is allowed to vary from one snippet to the other . - Since there will always be some noise',
        'elements : - 🔀 Your chunk size is allowed to vary from one snippet to the other . - Since there will always be some noise in your retrieval , increasing the top_k increases the chance to get relevant elements in your retrieved snippets . 🎯 Shooting more arrows increases your probability of hitting your target . - Meanwhile , the summed length of your retrieved documents should not be too high : for instance , for most current models 16k tokens will probably drown your Reader model in information due to Lost - in - the - middle phenomenon . 🎯 Give your reader model only the most relevant insights , not a huge pile of books ! In this notebook , we use Langchain library since it offers a huge variety of options for vector databases and allows',
        ', context , response , and combine these with LLM calls . These evaluation modules are in the following forms : - Correctness : Whether the generated answer matches that of the reference answer given the query ( requires labels ) . - Semantic Similarity Whether the predicted answer is semantically similar to the reference answer ( requires labels ) . - Faithfulness : Evaluates if the answer is faithful to the retrieved contexts ( in other words , whether if there ’ s hallucination ) . - Context Relevancy : Whether retrieved context is relevant to the query . - Answer Relevancy : Whether the generated answer is relevant to the query . - Guideline Adherence : Whether the predicted answer adheres to specific guidelines . Question Generation Section titled “ Question Generation ” In addition to evaluating queries',
        'The RAG Triad ¶ RAGs have become the standard architecture for providing LLMs with context in order to avoid hallucinations . However , even RAGs can suffer from hallucination , as is often the case when the retrieval fails to retrieve sufficient context or even retrieves irrelevant context that is then weaved into the LLM ’ s response . TruEra has innovated the RAG triad to evaluate for hallucinations along each edge of the RAG architecture , shown below : The RAG triad is made up of 3 evaluations : context relevance , groundedness and answer relevance . Satisfactory evaluations on each provides us confidence that our LLM app is free from hallucination . Context Relevance ¶ The first step of any RAG application is retrieval ; to verify the quality of our retrieval , we want to make sure',
        ', users need to view its output as trustworthy . RAG models can include citations to the knowledge sources in their external data as part of their responses . When RAG models cite their sources , human users can verify those outputs to confirm accuracy while consulting the cited works for follow - up clarification and additional information . Corporate data storage is often a complex and siloed maze . RAG responses with citations point users directly toward the materials they need . Access to more data means that one model can handle a wider range of prompts . Enterprises can optimize models and gain more value from them by broadening their knowledge bases , in turn expanding the contexts in which those models generate reliable results . By combining generative AI with retrieval systems , RAG models can retrieve',
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

| Metric                | Value    |
|:----------------------|:---------|
| accuracy              | 0.8361   |
| accuracy_threshold    | 1.4131   |
| f1                    | 0.8378   |
| f1_threshold          | -0.3321  |
| precision             | 0.7209   |
| recall                | 1.0      |
| **average_precision** | **0.91** |

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

* Size: 549 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 549 samples:
  |         | sentence_0                                                                       | sentence_1                                                                            | label                                                         |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | float                                                         |
  | details | <ul><li>min: 7 tokens</li><li>mean: 10.0 tokens</li><li>max: 12 tokens</li></ul> | <ul><li>min: 121 tokens</li><li>mean: 159.58 tokens</li><li>max: 257 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | label            |
  |:-------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is context stuffing?</code>                 | <code>query , it returns a few relevant snippets from your knowledge base . These snippets will then be fed to the Reader Model to help it generate its answer . So our objective here is , given a user question , to find the most relevant snippets from our knowledge base to answer that question . This is a wide objective , it leaves open some questions . How many snippets should we retrieve ? This parameter will be named top_k . How long should these snippets be ? This is called the chunk size . There ’ s no one - size - fits - all answers , but here are a few elements : - 🔀 Your chunk size is allowed to vary from one snippet to the other . - Since there will always be some noise</code>                                                                                                                                            | <code>1.0</code> |
  | <code>How can batching improve performance?</code>     | <code>elements : - 🔀 Your chunk size is allowed to vary from one snippet to the other . - Since there will always be some noise in your retrieval , increasing the top_k increases the chance to get relevant elements in your retrieved snippets . 🎯 Shooting more arrows increases your probability of hitting your target . - Meanwhile , the summed length of your retrieved documents should not be too high : for instance , for most current models 16k tokens will probably drown your Reader model in information due to Lost - in - the - middle phenomenon . 🎯 Give your reader model only the most relevant insights , not a huge pile of books ! In this notebook , we use Langchain library since it offers a huge variety of options for vector databases and allows</code>                                                                        | <code>1.0</code> |
  | <code>How do you ensure consistency in outputs?</code> | <code>, context , response , and combine these with LLM calls . These evaluation modules are in the following forms : - Correctness : Whether the generated answer matches that of the reference answer given the query ( requires labels ) . - Semantic Similarity Whether the predicted answer is semantically similar to the reference answer ( requires labels ) . - Faithfulness : Evaluates if the answer is faithful to the retrieved contexts ( in other words , whether if there ’ s hallucination ) . - Context Relevancy : Whether retrieved context is relevant to the query . - Answer Relevancy : Whether the generated answer is relevant to the query . - Guideline Adherence : Whether the predicted answer adheres to specific guidelines . Question Generation Section titled “ Question Generation ” In addition to evaluating queries</code> | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 12
- `per_device_eval_batch_size`: 12

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 12
- `num_train_epochs`: 3
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
- `per_device_eval_batch_size`: 12
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
| 1.0   | 46   | 0.4485                               |
| 2.0   | 92   | 0.5706                               |
| 3.0   | 138  | 0.9100                               |


### Training Time
- **Training**: 6.0 seconds

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