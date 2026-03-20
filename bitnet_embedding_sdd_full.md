# BitNet-Based Embedding Model Development  
## Software Design Document (SDD)

**Document status:** Final draft  
**Document language:** English  
**Prepared for:** BitNet embedding feasibility / PoC / service-readiness study  
**Last updated:** 2026-03-20  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)  
2. [Purpose](#2-purpose)  
3. [Background and Design Premises](#3-background-and-design-premises)  
4. [Goals and Non-Goals](#4-goals-and-non-goals)  
5. [System Scope](#5-system-scope)  
6. [Constraints and Assumptions](#6-constraints-and-assumptions)  
7. [Architecture Principles and Decision Log](#7-architecture-principles-and-decision-log)  
8. [High-Level Architecture](#8-high-level-architecture)  
9. [Recommended Technology Stack](#9-recommended-technology-stack)  
10. [Repository Layout](#10-repository-layout)  
11. [Data Contracts and Internal Schemas](#11-data-contracts-and-internal-schemas)  
12. [Detailed Model Design](#12-detailed-model-design)  
13. [Dataset Plan and Hugging Face Sources](#13-dataset-plan-and-hugging-face-sources)  
14. [Training Pipeline and Orchestration](#14-training-pipeline-and-orchestration)  
15. [Evaluation Plan](#15-evaluation-plan)  
16. [Deployment Plan](#16-deployment-plan)  
17. [Testing Strategy](#17-testing-strategy)  
18. [Risk Register and Mitigations](#18-risk-register-and-mitigations)  
19. [Deliverables and Milestones](#19-deliverables-and-milestones)  
20. [Appendix A: Example `pyproject.toml`](#20-appendix-a-example-pyprojecttoml)  
21. [Appendix B: Example YAML Config](#21-appendix-b-example-yaml-config)  
22. [Appendix C: Example Core Model Skeleton](#22-appendix-c-example-core-model-skeleton)  
23. [References](#23-references)  

---

## 1. Executive Summary

This document specifies a complete software design for building an **embedding model on top of a BitNet backbone**. The main engineering challenge is that the current public BitNet ecosystem is split into two operational paths:

1. **A training / experimentation path**, where the released **BF16 master checkpoint** is the practical source for adaptation and fine-tuning.  
2. **An efficiency / deployment path**, where BitNet’s primary performance advantages are exposed through **bitnet.cpp**, not through the standard Python `transformers` execution path.[^bitnet-model-card][^bitnet-cpp]

Because the standard `transformers` path is explicitly not the path that exposes BitNet’s main efficiency gains, this SDD separates:

- **Embedding quality validation**
- **Resource-feasible fine-tuning on a 16GB GPU**
- **Efficiency validation in a low-bit runtime**

The recommended implementation is therefore:

> **BitNet BF16 backbone + custom embedding head + LoRA-centered adaptation + Accelerate-based training loop**, with **Sentence Transformers** used for baselines, evaluators, and loss recipes, and **bitnet.cpp** reserved for a later efficiency-validation track.[^bitnet-model-card][^bitnet-docs][^st-training][^st-losses][^accelerate-docs][^peft-docs]

---

## 2. Purpose

The purpose of this SDD is to define the **system architecture, implementation strategy, dataset plan, training flow, evaluation methodology, deployment path, and coding standards** required to build and validate a BitNet-based text embedding model.

The system must answer four questions:

1. Can BitNet hidden states be turned into useful sentence embeddings?
2. Which adaptation method is realistic on a 16GB GPU: head-only, LoRA, or limited full fine-tuning?
3. How does the resulting model compare to modern BF16 embedding baselines on retrieval and semantic similarity?
4. Can BitNet’s claimed efficiency advantages still matter in a real embedding deployment path?

---

## 3. Background and Design Premises

BitNet b1.58 2B4T is released by Microsoft Research as an open native 1-bit / ternary-weight LLM family. The Hugging Face model card and Transformers documentation describe it as a Transformer architecture using **BitLinear** layers, **RoPE**, **ReLU²**, **SubLN**, and no bias terms in linear or normalization layers.[^bitnet-model-card][^bitnet-docs]

The official model card exposes three practical model variants:

- **Packed 1.58-bit weights** for deployment  
- **BF16 master weights** for training or fine-tuning  
- **GGUF weights** for `bitnet.cpp` inference[^bitnet-model-card]

The same model card also includes an explicit warning:

- **Do not expect speed / latency / energy gains when running BitNet through the standard `transformers` path**
- For the efficiency gains shown in the BitNet paper, **use `bitnet.cpp`**[^bitnet-model-card]

This single warning fundamentally shapes the design of this project. It means:

- **Quality benchmarking** should happen in the Python research stack.
- **True efficiency claims** must be validated in a separate low-bit runtime path.
- The project should **not** mix these two evaluation goals into one misleading benchmark chart.

A second premise comes from the Transformers BitNet documentation:

- `BitNetModel` is the **bare model** that returns raw hidden states.
- `last_hidden_state` and optional `hidden_states` are available in the forward output.[^bitnet-docs-hidden]

This makes BitNet usable as an embedding backbone without forcing a language-model head into the design.

A third premise comes from the Hugging Face quantization documentation for BitNet:

- BitNet models are **not quantized on the fly**.
- BitNet uses a **quantization-aware training (QAT)** approach.[^bitnet-quant-doc]

For this SDD, that means we will **not** attempt to reinvent BitNet pretraining. We will instead treat the released BF16 master checkpoint as the research-time starting point and focus on embedding adaptation and system evaluation.

---

## 4. Goals and Non-Goals

### 4.1 Goals

The system must:

- Build a sentence / text embedding model from a BitNet backbone.
- Support symmetric semantic similarity training and asymmetric retrieval training.
- Support a realistic 16GB-GPU adaptation path.
- Compare quality against practical BF16 baselines.
- Produce a serving path compatible with a standard `/v1/embeddings` style API.
- Preserve a separate path for validating BitNet-specific efficiency in a specialized runtime.

### 4.2 Non-Goals

The system does **not** aim to:

- Reproduce BitNet pretraining
- Re-implement BitNet kernels
- Build a production-scale vector database platform
- Build a cross-encoder reranker
- Prove final production-worthiness in one iteration

---

## 5. System Scope

### 5.1 In Scope

- BitNet-based embedding backbone integration
- Pooling and projection-head design
- Head-only, LoRA, and limited full fine-tuning
- Contrastive and triplet-based training
- Retrieval / STS / clustering evaluation
- Latency / throughput / VRAM measurement
- Research-serving API
- Phase-2 low-bit efficiency investigation

### 5.2 Out of Scope

- Distributed multi-node training
- Native BitNet kernel development
- Full ANN serving infrastructure
- Production traffic management
- Hybrid lexical + dense search fusion in the first iteration

---

## 6. Constraints and Assumptions

### 6.1 Hardware Constraint

Primary development target:

- **Single 16GB GPU**
- Linux-first development
- Optional CPU smoke-test path

### 6.2 Runtime Constraint

The design assumes:

- The first high-control training path is **PyTorch + Transformers + PEFT + Accelerate**
- The first deployment path is **Python service**, not `bitnet.cpp`
- The second deployment / efficiency path is **experimental**, not the initial service baseline

### 6.3 Model Constraint

The design assumes that:

- The released BF16 BitNet checkpoint is the correct fine-tuning source[^bitnet-model-card]
- `BitNetModel` can provide usable hidden states for embedding extraction[^bitnet-docs-hidden]
- The standard Python runtime will not be used to claim BitNet’s final hardware efficiency[^bitnet-model-card]

### 6.4 Evaluation Constraint

Public benchmarks should be used **after** the recipe stabilizes. Sentence Transformers documentation emphasizes training components and post-training evaluation workflows, while MTEB-style public reporting should not become the optimization target too early.[^st-training]

---

## 7. Architecture Principles and Decision Log

### 7.1 Principle 1 — Separate Quality from Efficiency

**Decision:** Use a Python/HF path for training and quality benchmarking, and a separate low-bit path for efficiency validation.

**Why:** The official BitNet model card explicitly says the standard `transformers` path is not the path for observing BitNet’s real speed / latency / energy gains.[^bitnet-model-card]

### 7.2 Principle 2 — Use the Bare Model, Not the LM Head

**Decision:** Use `BitNetModel` or `AutoModel` for embedding extraction.

**Why:** The Transformers documentation defines `BitNetModel` as the bare model that returns raw hidden states.[^bitnet-docs-hidden]

### 7.3 Principle 3 — Keep the First Embedding Head Minimal

**Decision:** Start with:

- masked mean pooling
- linear projection
- L2 normalization

**Why:** This minimizes moving parts and isolates whether the BitNet backbone itself is embedding-capable.

### 7.4 Principle 4 — LoRA Is the Main Practical Adaptation Path

**Decision:** Treat **LoRA** as the default serious fine-tuning path, while preserving head-only and limited full FT experiments.

**Why:** PEFT is designed specifically for parameter-efficient adaptation, and LoRA reduces trainable parameter count and memory pressure.[^peft-docs][^lora-docs]

### 7.5 Principle 5 — Use Sentence Transformers as a Helper Layer, Not the Only Core Abstraction

**Decision:** Keep the core model in plain PyTorch modules and use Sentence Transformers for:

- baseline recipes
- evaluator reuse
- loss reference
- data-format compatibility

**Why:** Sentence Transformers is modular and flexible, but a custom BitNet embedding system requires tighter control than a purely framework-driven implementation.[^st-training]

### 7.6 Principle 6 — Use Accelerate for the Main Training Loop

**Decision:** Implement the main training loop with **Accelerate**.

**Why:** Accelerate lets the same PyTorch code run across different execution setups with minimal code changes, while preserving control over data flow and losses.[^accelerate-docs]

### 7.7 Principle 7 — Do Not Depend on TEI in Phase 1

**Decision:** Use a custom FastAPI service first.

**Why:** Text Embeddings Inference documents support for Qwen2, Qwen3, Alibaba GTE, MPNet, Gemma3, etc., but BitNet is not listed as a supported embedding model family.[^tei-supported]

---

## 8. High-Level Architecture

```text
[Training / Research Path]
    Dataset Loader
        -> Tokenizer
        -> BitNet BF16 Backbone
        -> Hidden-State Selector
        -> Pooling Layer
        -> Projection Head
        -> L2 Normalize
        -> Contrastive / Triplet Loss
        -> Checkpoint / Eval / Export

[Python Serving Path]
    Exported HF Checkpoint (+ optional LoRA adapter)
        -> FastAPI embedding service
        -> Offline benchmark harness
        -> ANN validation harness

[Efficiency Validation Path]
    Packed / GGUF BitNet artifact
        -> bitnet.cpp feasibility track
        -> low-bit runtime experiments
        -> latency / energy / memory validation
```

### 8.1 System Layers

- **Data Layer**: Hugging Face Datasets or local JSONL / Parquet
- **Model Layer**: BitNet backbone + embedding head
- **Training Layer**: Accelerate loops + PEFT + config-driven experiments
- **Evaluation Layer**: retrieval, STS, clustering, latency, VRAM, throughput
- **Serving Layer**: Python API
- **Phase-2 Runtime Layer**: bitnet.cpp efficiency research

### 8.2 Why This Architecture

This architecture avoids a common failure mode: trying to force the same code path to represent both **research quality** and **BitNet-specific efficiency**. The official BitNet documentation makes clear that those are not the same execution path today.[^bitnet-model-card][^bitnet-cpp]

---

## 9. Recommended Technology Stack

### 9.1 Required Libraries

- `torch`
- `transformers`
- `peft`
- `accelerate`
- `datasets`
- `safetensors`
- `numpy`
- `scikit-learn`

### 9.2 Strongly Recommended

- `sentence-transformers`
- `mteb`
- `fastapi`
- `pydantic`
- `uvicorn`
- `orjson`
- `pytest`
- `ruff`
- `mypy`
- `pre-commit`

### 9.3 Optional

- `wandb` or `mlflow`
- `faiss-cpu` / `faiss-gpu`
- `qdrant-client`
- `hydra-core` or `omegaconf`
- `prometheus-client`

### 9.4 Why These Libraries

**Transformers** is required for BitNet model access and API compatibility.[^bitnet-docs]  
**PEFT** is the standard Hugging Face path for efficient adapter-based fine-tuning.[^peft-docs]  
**Accelerate** is the preferred way to keep a custom PyTorch training loop simple while retaining compatibility with mixed precision and scalable execution.[^accelerate-docs]  
**Sentence Transformers** provides training abstractions, evaluator support, and documented loss recommendations for high-performing embedding systems.[^st-training][^st-losses]  
**Datasets** gives a uniform loading path for Hugging Face datasets, including the retrieval and semantic similarity corpora used in this project.[^all-nli][^msmarco-bm25]

### 9.5 Package Management Recommendation

Use:

- `uv` for environment and dependency management
- `pyproject.toml` as the single source of dependency truth
- lockfile-based reproducibility

### 9.6 Versioning Policy

Because BitNet support is version-sensitive and the official model card itself pins a specific Transformers revision for the documented Python path, all dependencies should be pinned and tested together.[^bitnet-model-card]

---

## 10. Repository Layout

```text
bitnet-embed/
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ docs/
│  └─ sdd.md
├─ configs/
│  ├─ model/
│  │  ├─ bitnet_base.yaml
│  │  ├─ lora.yaml
│  │  └─ full_ft.yaml
│  ├─ data/
│  │  ├─ semantic.yaml
│  │  ├─ nq.yaml
│  │  ├─ msmarco.yaml
│  │  └─ beir_eval.yaml
│  ├─ train/
│  │  ├─ smoke.yaml
│  │  ├─ head_only.yaml
│  │  ├─ lora_retrieval.yaml
│  │  └─ full_ft_retrieval.yaml
│  └─ eval/
│     ├─ mteb.yaml
│     ├─ retrieval.yaml
│     └─ latency.yaml
├─ src/bitnet_embed/
│  ├─ __init__.py
│  ├─ modeling/
│  │  ├─ backbone.py
│  │  ├─ pooling.py
│  │  ├─ projection.py
│  │  ├─ prompts.py
│  │  ├─ lora.py
│  │  └─ model.py
│  ├─ data/
│  │  ├─ schemas.py
│  │  ├─ loaders.py
│  │  ├─ preprocess.py
│  │  └─ collators.py
│  ├─ losses/
│  │  ├─ contrastive.py
│  │  ├─ triplet.py
│  │  └─ matryoshka.py
│  ├─ train/
│  │  ├─ loops.py
│  │  ├─ trainer.py
│  │  ├─ optim.py
│  │  └─ callbacks.py
│  ├─ eval/
│  │  ├─ retrieval.py
│  │  ├─ sts.py
│  │  ├─ clustering.py
│  │  ├─ mteb_runner.py
│  │  └─ benchmark.py
│  ├─ serve/
│  │  ├─ api.py
│  │  ├─ schemas.py
│  │  ├─ runtime.py
│  │  └─ health.py
│  └─ utils/
│     ├─ seed.py
│     ├─ logging.py
│     ├─ metrics.py
│     └─ io.py
├─ scripts/
│  ├─ train_smoke.py
│  ├─ train_head_only.py
│  ├─ train_lora.py
│  ├─ train_full.py
│  ├─ eval_retrieval.py
│  ├─ eval_mteb.py
│  └─ benchmark_latency.py
└─ tests/
   ├─ test_pooling.py
   ├─ test_model_forward.py
   ├─ test_lora_targets.py
   ├─ test_data_pipeline.py
   └─ test_api.py
```

### 10.1 Coding Rules

- Keep model logic out of trainer abstractions.
- Keep losses as standalone `torch.nn.Module` classes.
- Keep preprocessing deterministic and testable.
- Use config-driven experiments only.
- Avoid notebook-only logic for production or benchmark code.
- Keep evaluation outputs versioned and comparable.
- Require type hints for all public functions.

---

## 11. Data Contracts and Internal Schemas

### 11.1 Pair Schema

```json
{
  "anchor": "text A",
  "positive": "text B",
  "task": "semantic_similarity",
  "source": "all-nli"
}
```

### 11.2 Triplet Schema

```json
{
  "anchor": "query",
  "positive": "relevant passage",
  "negative": "hard negative passage",
  "task": "retrieval",
  "source": "msmarco-bm25"
}
```

### 11.3 Query-Document Schema

```json
{
  "query": "What causes high LDL?",
  "document": "LDL levels can rise because of diet, genetics, and metabolic factors.",
  "label": 1,
  "source": "nfcorpus"
}
```

### 11.4 Encoded Embedding Record

```json
{
  "id": "doc-000123",
  "text": "Some text here",
  "embedding": [0.004, -0.021, 0.118, "..."],
  "normalized": true,
  "dim": 768,
  "model": "bitnet-embed-lora-v1"
}
```

---

## 12. Detailed Model Design

## 12.1 Backbone Loader

### Decision

Use the released **BF16 master checkpoint** as the training backbone.

### Rationale

The official model card explicitly states:

- packed 1.58-bit weights are for deployment
- BF16 master weights are for training / fine-tuning
- GGUF weights are for `bitnet.cpp` CPU inference[^bitnet-model-card]

### Implementation Rules

- Prefer `BitNetModel` / `AutoModel` over `BitNetForCausalLM`
- Use `output_hidden_states=True` for layer experiments
- Set `use_cache=False` for training
- Standardize padding behavior
- Standardize prompt format for query vs. document tasks

### Suggested Interface

```python
class BitNetBackbone(nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype, output_hidden_states: bool):
        ...
    def forward(self, input_ids, attention_mask) -> dict:
        ...
```

## 12.2 Hidden-State Selection

### Default

- `last_hidden_state`

### Alternatives to support

- last-layer mean pooling
- EOS pooling
- last-token pooling
- average of last 4 layers
- learned weighted sum of layers

### Rationale

Transformers documentation confirms that `BitNetModel` returns:

- `last_hidden_state`
- optional `hidden_states` when `output_hidden_states=True`[^bitnet-docs-hidden]

### Recommended Experiment Order

1. Last layer + masked mean pooling
2. Last-4 average + masked mean pooling
3. EOS pooling
4. Learned attention pooling

## 12.3 Pooling Layer

### Default Strategy

Use **masked mean pooling**.

### Why

BitNet is a decoder-style model and does not naturally expose a BERT-style `[CLS]` representation. Mean pooling is the least assumption-heavy starting point and aligns with common Sentence Transformers-style embedding construction.[^st-training]

### Reference Implementation

```python
def masked_mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts
```

## 12.4 Projection Head

### Default

- input: backbone hidden size
- output: 768-d embedding
- structure: single `Linear(hidden_size, emb_dim, bias=False)`
- post-process: `L2 normalize`

### Optional Variants

- 1024 / 768 / 512 dimensions
- optional dropout
- optional LayerNorm
- 2-layer MLP in later experiments

### Rationale

The first goal is to verify whether **BitNet hidden states are naturally adaptable**. The simplest projection head is the clearest test.

## 12.5 Normalization

Default to:

- `F.normalize(embedding, p=2, dim=-1)`

Normalization should be configurable, but retrieval evaluations should use normalized embeddings unless a specific ablation disables it.

## 12.6 Prompt Formatting

The system must support task-specific formatting such as:

- query prefix
- document prefix
- optional instruction prefix

Example:

```text
query: What are the symptoms of lupus?
document: Lupus symptoms commonly include fatigue, joint pain, and skin rash.
```

Prompt formatting must be configurable because decoder-style backbones often benefit from stable input conventions.

## 12.7 LoRA Design

### Target Modules

The current Transformers BitNet implementation exposes the following projection modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`[^bitnet-source]

### Initial LoRA Recommendation

- `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `bias = "none"`

### Experiment Order

1. Attention-only LoRA
2. Attention + MLP LoRA
3. Upper-layer-only LoRA
4. Reduced-rank LoRA for memory stress tests

### Why LoRA

LoRA is explicitly documented as a parameter-efficient adaptation method that reduces the number of trainable parameters and memory burden.[^lora-docs][^peft-docs]

## 12.8 Loss Design

### Default Loss Family

Use **Multiple Negatives / InfoNCE-style** contrastive losses.

### Why

Sentence Transformers documentation states that:

- `MultipleNegativesRankingLoss` is commonly used for top-performing embedding models on `(anchor, positive)` data
- `CachedMultipleNegativesRankingLoss` is often used to increase effective batch size and improve performance[^st-losses]

### Priority Order

1. Symmetric InfoNCE
2. Cached Multiple Negatives
3. Triplet loss for mined hard negatives
4. Matryoshka losses later

### Default Symmetric InfoNCE

```python
scores = query_emb @ doc_emb.T / temperature
labels = torch.arange(scores.size(0), device=scores.device)
loss = 0.5 * (
    F.cross_entropy(scores, labels) +
    F.cross_entropy(scores.T, labels)
)
```

### Design Rule

Keep losses in project code as explicit modules rather than burying them inside a trainer.

## 12.9 Matryoshka Option

Matryoshka support is optional and belongs to **post-stability optimization**, not the first milestone.

Order of use:

1. stabilize the backbone + pooling + LoRA recipe
2. test truncation at 768 / 512 / 256
3. add Matryoshka only if reduced-dimension storage / ANN cost matters

## 12.10 Tokenization and Batching

### Defaults

- `max_length = 256` or `384`
- dynamic padding
- shared tokenizer with the backbone
- explicit EOS / PAD policy

### Suggested Starting Points on 16GB GPU

- Head-only: seq 512, micro-batch 16
- LoRA: seq 384, micro-batch 4–8
- Limited full FT: seq 256–384, micro-batch 1–2

### Rules

- maximize **effective batch size** through gradient accumulation
- avoid duplicate positives in-batch
- keep retrieval negatives diverse
- separate query and document collators only if prompt formatting diverges

---

## 13. Dataset Plan and Hugging Face Sources

## 13.1 Dataset Strategy

The dataset plan is intentionally **staged** rather than monolithic.

The model should move through the following progression:

1. **Symmetric semantic warm-up**
2. **Question-answer retrieval warm-up**
3. **Hard-negative retrieval training**
4. **Large-scale retrieval training**
5. **Held-out domain-shift validation**
6. **Final external reporting**

This staged design is important because the project is testing not just raw embedding quality, but the practicality of adapting BitNet into a retrieval encoder under strict hardware constraints.

## 13.2 Dataset Inventory

### 13.2.1 Semantic Warm-Up

| HF dataset ID | Approx. rows / subsets | Role |
|---|---:|---|
| `sentence-transformers/all-nli` | pair 328k / pair-class 981k / pair-score 981k / triplet 571k | semantic warm-up, head-only feasibility, early LoRA |
| `sentence-transformers/quora-duplicates` | pair 149k / pair-class 404k / triplet 102k / triplet-all 2.79M | duplicate-question / paraphrase robustness |
| `sentence-transformers/stsb` | 8,628 rows | fast semantic regression checks |
| `mteb/stsbenchmark-sts` | 8.63k rows (train 5.75k / val 1.5k / test 1.38k) | external STS-style reporting only |

### 13.2.2 Retrieval Warm-Up

| HF dataset ID | Approx. rows / subsets | Role |
|---|---:|---|
| `sentence-transformers/natural-questions` | pair 100k | first asymmetric query→answer retrieval training |
| `tomaarsen/natural-questions-hard-negatives` | triplet-5 96.7k / triplet-all 484k | retrieval hard-negative expansion |

### 13.2.3 Main Retrieval Training

| HF dataset ID | Approx. rows / subsets | Role |
|---|---:|---|
| `sentence-transformers/msmarco-bm25` | triplet 503k / triplet-50 503k / triplet-all 26.6M / triplet-hard 19.1M | main retrieval fine-tuning corpus |
| `sentence-transformers/msmarco-corpus` | passage 8.84M / query 1.01M / total 9,852,739 | text mapping for ID-based MS MARCO subsets |

### 13.2.4 Held-Out Validation / Domain Shift

| HF dataset ID | Approx. size | Role |
|---|---:|---|
| `irds/beir_nfcorpus` + `irds/beir_nfcorpus_dev` | corpus 3,633 docs / 3,237 queries / dev 324 queries + 11,385 qrels | biomedical retrieval validation |
| `irds/beir_fiqa` + `irds/beir_fiqa_train` + `irds/beir_fiqa_test` | corpus 57,638 docs / 6,648 queries / train 5,500 queries + 14,166 qrels / test 648 queries + 1,706 qrels | financial-domain retrieval validation |
| `irds/beir_scifact` + `irds/beir_scifact_test` | corpus 5,183 docs / 1,109 queries / test 300 queries + 339 qrels | scientific evidence retrieval |
| `irds/beir_arguana` | 8,674 docs / 1,406 queries / 1,406 qrels | argument / counterargument retrieval |
| `irds/beir_hotpotqa` + `irds/beir_hotpotqa_test` | corpus 5,233,329 docs / 97,852 queries / test 7,405 queries + 14,810 qrels | large-corpus stress test |

## 13.3 Why These Datasets

### `sentence-transformers/all-nli`

AllNLI is a standard semantic warm-up corpus already documented as usable for training / fine-tuning embedding models for semantic textual similarity.[^all-nli]

### `sentence-transformers/quora-duplicates`

Quora duplicates is useful for short-form paraphrase robustness and duplicate-intent alignment, which matters for search-style user queries.[^quora]

### `sentence-transformers/natural-questions`

Natural Questions gives the first asymmetrical query→answer training step without forcing the project into massive triplet corpora too early.[^nq]

### `tomaarsen/natural-questions-hard-negatives`

This adds harder negatives after the first NQ warm-up. The dataset card states that negatives were mined from the NQ base set using `all-MiniLM-L6-v2`.[^nq-hard]

### `sentence-transformers/msmarco-bm25`

This is the main large-scale retrieval training dataset. It provides string-based and ID-based mined triplets, plus harder negative subsets, which makes it the natural main retrieval corpus for LoRA and upper-bound experiments.[^msmarco-bm25]

### `sentence-transformers/msmarco-corpus`

This dataset exists specifically to map MS MARCO query and passage IDs back to raw text, which is necessary for ID-based mined triplet variants.[^msmarco-corpus]

### BEIR / IRDS Datasets

The selected BEIR subsets introduce valuable domain shifts:

- biomedical IR
- finance retrieval
- scientific evidence retrieval
- argumentative retrieval
- large-corpus QA retrieval[^nfcorpus][^nfcorpus-dev][^fiqa][^fiqa-train][^fiqa-test][^scifact][^scifact-test][^arguana][^hotpotqa][^hotpotqa-test]

## 13.4 Recommended Dataset Bundles by Project Stage

### Minimal 16GB Feasibility Bundle

Use:

- `sentence-transformers/all-nli`
- `sentence-transformers/quora-duplicates`
- `sentence-transformers/natural-questions`
- `sentence-transformers/stsb`

This is enough to validate hidden-state extraction, pooling, projection-head training, and the first asymmetric retrieval adaptation.

### Practical PoC Bundle

Use:

- `sentence-transformers/all-nli`
- `sentence-transformers/quora-duplicates`
- `sentence-transformers/natural-questions`
- `tomaarsen/natural-questions-hard-negatives`
- `sentence-transformers/msmarco-bm25`
- BEIR validation on:
  - `irds/beir_nfcorpus_dev`
  - `irds/beir_fiqa_test`
  - `irds/beir_scifact_test`

This is the smallest bundle that meaningfully tests whether BitNet can become a practical retrieval encoder.

### Final Report Bundle

Add:

- `sentence-transformers/msmarco-corpus` when ID-based triplets are used
- `irds/beir_arguana`
- `irds/beir_hotpotqa_test`
- `mteb/stsbenchmark-sts` for external reporting

## 13.5 Dataset Schedule by Experiment Stage

### Stage 0 — Smoke Test

Use:

- tiny slices of `sentence-transformers/stsb`
- tiny slices of `sentence-transformers/all-nli`

Purpose:

- verify tokenizer behavior
- verify hidden-state extraction
- verify pooling correctness
- verify backward pass and checkpoint save/load

### Stage 1 — Head-Only Feasibility

Use:

- `sentence-transformers/all-nli`
- `sentence-transformers/quora-duplicates`

Purpose:

- determine whether BitNet hidden states already contain enough semantic structure for useful embeddings

### Stage 2 — Initial Retrieval Adaptation

Use:

- `sentence-transformers/natural-questions`
- `tomaarsen/natural-questions-hard-negatives`

Purpose:

- transition from sentence similarity to asymmetric retrieval

### Stage 3 — Main Retrieval Training

Use:

- `sentence-transformers/msmarco-bm25`
- optionally `sentence-transformers/msmarco-corpus`

Purpose:

- produce the main LoRA model and limited full-FT upper bound

### Stage 4 — Held-Out Validation and Stress Testing

Use:

- `irds/beir_nfcorpus_dev`
- `irds/beir_fiqa_test`
- `irds/beir_scifact_test`
- optionally `irds/beir_arguana`
- optionally `irds/beir_hotpotqa_test`

Purpose:

- test domain shift and corpus scale robustness

### Stage 5 — External Reporting

Use:

- `mteb/stsbenchmark-sts`
- MTEB-compatible reporting scripts

Purpose:

- final public-style reporting only, after the training recipe is frozen

## 13.6 Dataset Loading Examples

```python
from datasets import load_dataset

# Semantic warm-up
allnli_triplet = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
quora_pair = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train")
stsb_val = load_dataset("sentence-transformers/stsb", split="validation")

# Retrieval warm-up
nq_pair = load_dataset("sentence-transformers/natural-questions", "pair", split="train")
nq_hard = load_dataset("tomaarsen/natural-questions-hard-negatives", "triplet-5", split="train")

# Main retrieval training
msmarco_triplet = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
msmarco_hard = load_dataset("sentence-transformers/msmarco-bm25", "triplet-hard", split="train")

# BEIR validation
fiqa_docs = load_dataset("irds/beir_fiqa", "docs")
fiqa_test_queries = load_dataset("irds/beir_fiqa_test", "queries")
fiqa_test_qrels = load_dataset("irds/beir_fiqa_test", "qrels")

scifact_docs = load_dataset("irds/beir_scifact", "docs")
scifact_test_queries = load_dataset("irds/beir_scifact_test", "queries")
scifact_test_qrels = load_dataset("irds/beir_scifact_test", "qrels")
```

## 13.7 Data Processing Rules

- normalize whitespace
- strip control characters
- preserve source metadata
- truncate extreme outliers
- optional query/document prefixes
- batch-level duplicate filtering
- task-specific collator support
- cache tokenized datasets by experiment hash
- keep held-out benchmark splits strictly separate from training data

---

## 14. Training Pipeline and Orchestration

## 14.1 Preferred Training Flow

```text
Dataset -> Preprocess -> Tokenize -> Batch Collator -> BitNet Backbone
       -> Pooling -> Projection -> Normalize -> Loss -> Backprop -> Eval -> Save
```

## 14.2 Orchestration Choice

Use a custom **Accelerate** training loop.

Why:

- cleaner control over BitNet-specific logic
- easier experiment parity between head-only / LoRA / full-FT
- easier logging of exact memory / throughput metrics
- lower framework lock-in[^accelerate-docs]

Sentence Transformers can still be used to cross-check recipes and evaluator behavior.[^st-training]

## 14.3 Training Modes

### Mode A — Head-Only

- freeze all backbone weights
- train pooling + projection only

Purpose:
- test whether BitNet hidden states are already structurally useful

### Mode B — LoRA

- freeze the base model
- inject adapters into projection-heavy modules

Purpose:
- main practical fine-tuning path on 16GB hardware

### Mode C — Limited Full Fine-Tuning

- train the full stack with strict memory controls

Purpose:
- estimate upper bound, not the default production path

## 14.4 Memory Optimization

Required:

- BF16 mixed precision
- `use_cache=False`
- gradient checkpointing
- gradient accumulation
- dataloader pinning / worker tuning
- evaluation batch-size decoupled from training batch-size

Optional:

- `torch.compile()`
- SDPA / optimized attention path if stable
- CPU offload experiments only if necessary

## 14.5 Checkpointing Policy

Save:

- model weights
- adapter weights
- tokenizer
- config snapshot
- optimizer / scheduler states (for resumable runs)
- evaluation summaries
- seed / git revision metadata

Checkpoint names should be deterministic:

```text
runs/{date}/{exp_name}/
  ├─ checkpoints/
  ├─ metrics/
  ├─ configs/
  └─ artifacts/
```

## 14.6 Training Loop Responsibilities

The training loop is responsible for:

- dataloader iteration
- forward pass orchestration
- loss computation
- backward pass
- optimizer / scheduler step
- distributed reduction
- eval hooks
- checkpoint save
- benchmark logging

The training loop is **not** responsible for embedding logic, tokenization rules, or dataset semantics.

---

## 15. Evaluation Plan

## 15.1 Quality Metrics

### Retrieval

- Recall@K
- MRR@10
- nDCG@10
- MAP@100 (optional)

### Semantic Similarity

- Spearman correlation
- Pearson correlation (optional)

### Clustering / Auxiliary

- NMI
- ARI
- pair classification accuracy (optional)

## 15.2 Internal Evaluation Sets

- STSB for semantic sanity
- NQ held-out slice for early retrieval quality
- BEIR subsets for domain robustness

## 15.3 External Evaluation

Use MTEB-style reporting after the recipe is frozen.

## 15.4 Efficiency Metrics

Measure:

- p50 latency
- p95 latency
- throughput (texts/s)
- peak VRAM
- memory on startup
- artifact size
- cold start vs. warm start

## 15.5 Benchmark Conditions

Test with:

- batch size: 1 / 8 / 32
- sequence length: 32 / 128 / 512
- normalize: on / off
- query mode vs. document mode

## 15.6 Fairness Rules

### Quality Comparison

Compare BitNet against BF16 baselines in the same Python stack.

### Efficiency Comparison

Compare low-bit runtime efficiency only in the specialized path.

### Avoid This Mistake

Do **not** use standard `transformers` inference numbers to claim final BitNet deployment efficiency; the model card explicitly warns against that.[^bitnet-model-card]

## 15.7 Baseline Models

Recommended baselines:

- `sentence-transformers/all-mpnet-base-v2`
- `Qwen/Qwen3-Embedding-0.6B`
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct`

These are practical baselines because the TEI supported-models page explicitly includes MPNet, Qwen2, Qwen3, and Alibaba GTE families.[^tei-supported]

---

## 16. Deployment Plan

## 16.1 Phase 1 — Python Embedding Service

### Stack

- FastAPI
- Pydantic
- Uvicorn
- orjson
- Prometheus metrics
- Docker

### Endpoints

- `POST /v1/embeddings`
- `GET /health`
- `GET /metrics`

### Request Schema

```json
{
  "input": ["text1", "text2"],
  "task": "query",
  "normalize": true,
  "truncate_dim": 768
}
```

### Response Schema

```json
{
  "model": "bitnet-embed-lora-v1",
  "data": [
    {"index": 0, "embedding": [0.01, -0.02, "..."]},
    {"index": 1, "embedding": [0.03, 0.11, "..."]}
  ],
  "usage": {
    "input_texts": 2,
    "tokens": 123
  }
}
```

## 16.2 Phase 2 — bitnet.cpp Efficiency Validation Track

This phase should be treated as a separate engineering spike.

Objectives:

- confirm how a BitNet-derived embedding path can be represented in a low-bit runtime
- validate memory / latency / throughput advantages of the BitNet runtime path
- decide whether a dedicated embedding extraction implementation is justified

The official BitNet repository describes `bitnet.cpp` as the official inference framework for 1-bit LLMs and highlights hardware/runtime efficiency advantages there.[^bitnet-cpp]

## 16.3 Why TEI Is Not the Initial Deployment Path

Text Embeddings Inference documents support for:

- Nomic
- BERT
- XLM-R
- JinaBERT
- Mistral
- Alibaba GTE
- Qwen2
- MPNet
- ModernBERT
- Qwen3
- Gemma3

BitNet is not listed there, so the safer initial path is a custom Python service.[^tei-supported]

---

## 17. Testing Strategy

## 17.1 Unit Tests

- pooling mask correctness
- normalization correctness
- hidden-state selection correctness
- prompt-formatting correctness
- LoRA target-module existence
- projection dimension correctness

## 17.2 Integration Tests

- one-batch forward + backward
- checkpoint save/load
- adapter merge / reload
- `encode()` deterministic output shape
- API request/response test
- small retrieval benchmark smoke test

## 17.3 Regression Tests

- cosine similarity drift on fixed reference texts
- latency drift on fixed batch sizes
- VRAM drift across releases
- config compatibility across checkpoints

## 17.4 CI Policy

CI should run:

- formatting
- lint
- static typing
- CPU smoke test
- tiny-data training loop test
- minimal API boot test

---

## 18. Risk Register and Mitigations

| Risk | Description | Mitigation |
|---|---|---|
| Hidden-state quality risk | BitNet hidden states may not transfer well to embeddings | start with head-only, quickly move to LoRA, test layer averaging |
| Memory risk | Full fine-tuning may be unrealistic on 16GB | treat LoRA as primary path, keep full FT narrow |
| Runtime mismatch risk | Python inference may misrepresent BitNet efficiency | separate quality path from efficiency path |
| Dataset overfitting risk | small semantic datasets may overstate progress | use staged retrieval + BEIR validation |
| Deployment uncertainty | low-bit embedding path may require custom work | keep Python service as phase 1 and bitnet.cpp as phase 2 |
| Version fragility | BitNet support depends on specific library revisions | pin versions and save environment metadata |

---

## 19. Deliverables and Milestones

## 19.1 Deliverables

### Model Artifacts

- HF-compatible embedding model package
- tokenizer
- projection-head weights
- optional LoRA adapters

### Evaluation Artifacts

- retrieval metrics report
- STS report
- latency report
- VRAM report
- benchmark comparison summary

### Operational Artifacts

- `Dockerfile`
- service config
- OpenAPI schema
- benchmark scripts

## 19.2 Suggested Milestones

### Week 1

- environment setup
- backbone smoke test
- pooling + projection implementation

### Week 2

- head-only experiments
- early semantic validation
- LoRA injection validation

### Week 3

- NQ + hard-negative training
- MS MARCO main training
- internal retrieval benchmarks

### Week 4

- BEIR validation
- latency / VRAM benchmarking
- final recommendation:
  - adopt
  - continue validation
  - not recommended

---

## 20. Appendix A: Example `pyproject.toml`

```toml
[project]
name = "bitnet-embed"
version = "0.1.0"
description = "BitNet-based embedding model research and serving"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.4",
  "transformers",
  "peft",
  "accelerate",
  "datasets",
  "safetensors",
  "sentence-transformers",
  "mteb",
  "fastapi",
  "uvicorn[standard]",
  "pydantic>=2",
  "orjson",
  "numpy",
  "scikit-learn",
  "faiss-cpu",
  "prometheus-client"
]

[dependency-groups]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit"
]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.10"
strict = true
```

---

## 21. Appendix B: Example YAML Config

```yaml
experiment_name: bitnet_lora_retrieval_v1
seed: 42

model:
  backbone_name: microsoft/bitnet-b1.58-2B-4T-bf16
  projection_dim: 768
  pooling: masked_mean
  use_last_k_layers: 1
  normalize: true
  dtype: bfloat16

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

data:
  train_sets:
    - name: sentence-transformers/natural-questions
      subset: pair
      split: train
    - name: tomaarsen/natural-questions-hard-negatives
      subset: triplet-5
      split: train
    - name: sentence-transformers/msmarco-bm25
      subset: triplet
      split: train
  eval_sets:
    - name: irds/beir_fiqa_test
    - name: irds/beir_scifact_test

tokenization:
  max_length: 384
  dynamic_padding: true

training:
  mode: lora
  epochs: 1
  micro_batch_size: 4
  grad_accum_steps: 8
  lr: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_checkpointing: true
  bf16: true
  log_every_steps: 10
  eval_every_steps: 500
  save_every_steps: 500

loss:
  name: symmetric_infonce
  temperature: 0.05

serving:
  normalize_default: true
  truncate_dim_default: 768
```

---

## 22. Appendix C: Example Core Model Skeleton

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def masked_mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@dataclass
class EncodeConfig:
    batch_size: int = 32
    normalize: bool = True
    task: Literal["query", "document"] = "document"


class BitNetEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        projection_dim: int = 768,
        pooling: str = "masked_mean",
        use_last_k_layers: int = 1,
        normalize: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.pooling = pooling
        self.use_last_k_layers = use_last_k_layers
        self.normalize = normalize

        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.use_last_k_layers > 1,
            use_cache=False,
            return_dict=True,
        )
        if self.use_last_k_layers > 1:
            hs = outputs.hidden_states[-self.use_last_k_layers :]
            token_embeddings = torch.stack(hs, dim=0).mean(dim=0)
        else:
            token_embeddings = outputs.last_hidden_state
        return token_embeddings

    def pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "masked_mean":
            return masked_mean_pool(token_embeddings, attention_mask)
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.forward_features(input_ids, attention_mask)
        sentence_embeddings = self.pool(token_embeddings, attention_mask)
        sentence_embeddings = self.projection(sentence_embeddings)
        if self.normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        return sentence_embeddings
```

---

## 23. References

[^bitnet-model-card]: Microsoft Research, **BitNet b1.58 2B4T (BF16) model card**, Hugging Face. Confirms the three released weight variants (packed deployment, BF16 training/fine-tuning, GGUF for `bitnet.cpp`), architecture summary, and the explicit warning that standard `transformers` usage should not be expected to show BitNet’s main speed / latency / energy gains. Source: <https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16>.

[^bitnet-docs]: Hugging Face Transformers, **BitNet model documentation**. Documents BitNet architecture details such as BitLinear layers, RoPE, ReLU², SubLN, and ternary-weight / 8-bit activation design. Source: <https://huggingface.co/docs/transformers/model_doc/bitnet>.

[^bitnet-docs-hidden]: Hugging Face Transformers, **BitNetModel forward output documentation**. Documents that `BitNetModel` is the bare model outputting raw hidden states, with `last_hidden_state` and optional `hidden_states` available. Source: <https://huggingface.co/docs/transformers/model_doc/bitnet>.

[^bitnet-quant-doc]: Hugging Face Transformers, **BitNet quantization documentation**. States that BitNet models cannot be quantized on the fly and require quantization during pretraining or fine-tuning because BitNet is a QAT technique. Source: <https://huggingface.co/docs/transformers/quantization/bitnet>.

[^bitnet-cpp]: Microsoft, **`microsoft/BitNet` GitHub repository**. Describes `bitnet.cpp` as the official inference framework for 1-bit LLMs and documents its optimized low-bit inference path. Source: <https://github.com/microsoft/BitNet>.

[^tei-supported]: Hugging Face, **Text Embeddings Inference supported models**. Lists supported embedding families such as Qwen2, Qwen3, Alibaba GTE, MPNet, ModernBERT, Gemma3, etc.; BitNet is not listed there. Source: <https://huggingface.co/docs/text-embeddings-inference/supported_models>.

[^st-training]: Sentence Transformers documentation, **Training Overview**. Documents the modular structure of embedding training (model, dataset, loss, arguments, evaluator, trainer). Source: <https://sbert.net/docs/sentence_transformer/training_overview.html>.

[^st-losses]: Sentence Transformers documentation, **Loss Overview**. States that `MultipleNegativesRankingLoss` is commonly used for top-performing embedding models and that `CachedMultipleNegativesRankingLoss` is often used to increase batch size and performance. Source: <https://sbert.net/docs/sentence_transformer/loss_overview.html>.

[^accelerate-docs]: Hugging Face, **Accelerate documentation**. Describes Accelerate as a library that lets the same PyTorch code run across distributed configurations with minimal changes. Source: <https://huggingface.co/docs/accelerate/index>.

[^peft-docs]: Hugging Face, **PEFT documentation**. Documents PEFT as the Hugging Face library for parameter-efficient fine-tuning. Source: <https://huggingface.co/docs/peft/index>.

[^lora-docs]: Hugging Face PEFT, **LoRA documentation**. Documents LoRA as a low-rank adaptation method that reduces trainable parameters and memory cost. Source: <https://huggingface.co/docs/peft/package_reference/lora>.

[^bitnet-source]: Hugging Face Transformers source code, **`modeling_bitnet.py`**. Confirms the relevant BitNet module names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) used to define LoRA target modules. Source: <https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/bitnet/modeling_bitnet.py>.

[^all-nli]: Hugging Face dataset card / viewer, **`sentence-transformers/all-nli`**. Documents dataset purpose and current subset sizes such as pair 328k, pair-class 981k, pair-score 981k, triplet 571k. Source: <https://huggingface.co/datasets/sentence-transformers/all-nli>.

[^quora]: Hugging Face dataset card / viewer, **`sentence-transformers/quora-duplicates`**. Documents pair / pair-class / triplet / triplet-all formats and current subset sizes. Source: <https://huggingface.co/datasets/sentence-transformers/quora-duplicates>.

[^nq]: Hugging Face dataset card / viewer, **`sentence-transformers/natural-questions`**. Documents the query-answer training format and current 100k-row pair subset. Source: <https://huggingface.co/datasets/sentence-transformers/natural-questions>.

[^nq-hard]: Hugging Face dataset card / viewer, **`tomaarsen/natural-questions-hard-negatives`**. Documents mined hard negatives, the `triplet-5` and `triplet-all` subsets, and mining with `all-MiniLM-L6-v2`. Source: <https://huggingface.co/datasets/tomaarsen/natural-questions-hard-negatives>.

[^msmarco-bm25]: Hugging Face dataset card / viewer, **`sentence-transformers/msmarco-bm25`**. Documents the mined MS MARCO triplet subsets and current sizes such as triplet 503k, triplet-all 26.6M, triplet-hard 19.1M. Source: <https://huggingface.co/datasets/sentence-transformers/msmarco-bm25>.

[^msmarco-corpus]: Hugging Face dataset card / viewer, **`sentence-transformers/msmarco-corpus`**. Documents the passage/query ID-to-text mapping corpus with 9,852,739 total rows. Source: <https://huggingface.co/datasets/sentence-transformers/msmarco-corpus>.

[^stsb]: Hugging Face dataset card / viewer, **`sentence-transformers/stsb`**. Documents normalized STS scores and the current 8,628-row dataset. Source: <https://huggingface.co/datasets/sentence-transformers/stsb>.

[^mteb-stsb]: Hugging Face dataset card / viewer, **`mteb/stsbenchmark-sts`**. Documents the current 8.63k-row external STS reporting dataset with train/validation/test splits. Source: <https://huggingface.co/datasets/mteb/stsbenchmark-sts>.

[^nfcorpus]: Hugging Face dataset card, **`irds/beir_nfcorpus`**. Documents the corpus size (3,633 docs / 3,237 queries). Source: <https://huggingface.co/datasets/irds/beir_nfcorpus>.

[^nfcorpus-dev]: Hugging Face dataset card, **`irds/beir_nfcorpus_dev`**. Documents the dev query and qrels counts (324 queries / 11,385 qrels). Source: <https://huggingface.co/datasets/irds/beir_nfcorpus_dev>.

[^fiqa]: Hugging Face dataset card, **`irds/beir_fiqa`**. Documents the corpus size (57,638 docs / 6,648 queries). Source: <https://huggingface.co/datasets/irds/beir_fiqa>.

[^fiqa-train]: Hugging Face dataset card, **`irds/beir_fiqa_train`**. Documents the train split (5,500 queries / 14,166 qrels). Source: <https://huggingface.co/datasets/irds/beir_fiqa_train>.

[^fiqa-test]: Hugging Face dataset card, **`irds/beir_fiqa_test`**. Documents the test split (648 queries / 1,706 qrels). Source: <https://huggingface.co/datasets/irds/beir_fiqa_test>.

[^scifact]: Hugging Face dataset card / commit-backed dataset card, **`irds/beir_scifact`**. Documents the corpus size (5,183 docs / 1,109 queries). Source: <https://huggingface.co/datasets/irds/beir_scifact>.

[^scifact-test]: Hugging Face dataset card, **`irds/beir_scifact_test`**. Documents the test split (300 queries / 339 qrels). Source: <https://huggingface.co/datasets/irds/beir_scifact_test>.

[^arguana]: Hugging Face dataset card, **`irds/beir_arguana`**. Documents the dataset size (8,674 docs / 1,406 queries / 1,406 qrels). Source: <https://huggingface.co/datasets/irds/beir_arguana>.

[^hotpotqa]: Hugging Face dataset card, **`irds/beir_hotpotqa`**. Documents the corpus size (5,233,329 docs / 97,852 queries). Source: <https://huggingface.co/datasets/irds/beir_hotpotqa>.

[^hotpotqa-test]: Hugging Face dataset card, **`irds/beir_hotpotqa_test`**. Documents the test split (7,405 queries / 14,810 qrels). Source: <https://huggingface.co/datasets/irds/beir_hotpotqa_test>.
