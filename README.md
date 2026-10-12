# AmbiCal: Calibration-Guided Adaptive Retrieval for Multilingual Agri-Technical Question Answering

This repository contains the implementation of **AmbiCal**, a retrieval-augmented generation (RAG) method for cross-lingual technical question answering over domain documents, together with its **Dense fixed-k baseline**, six frozen 200-question benchmark datasets, and the experiment framework used in the paper.

AmbiCal answers English questions against non-English technical manuals through: dense retrieval on the original English query → LLM translation into the document language with a second retrieval pass → true-score recomputation for all candidates → robust normalization with dynamic dual-branch fusion → false-positive soft correction (from document-level calibration) → a v2 context gate → counterfactual phrase generation, re-translation and re-retrieval with positive promotion of still-relevant candidates → soft redundancy → Null gate → adaptive-k stopping → structured evidence merge. Each component can be toggled independently via configuration switches.

## Repository Layout

```
RAG/
├── retrieval_pipelines/
│   ├── ambical/                  # AmbiCal method (config, models, numerics, phrases, selection, calibration, retriever)
│   ├── embedding_retriever.py    # BGE-M3 embedding retriever (Dense fixed-k baseline)
│   └── base_retriever.py
├── rag_tester.py                 # Shared QA / evaluation / result-writing orchestration
├── rag_tester_ambical.py         # AmbiCal entry point (argparse, method isolation, profile validation)
├── rag_tester_dense_fixed.py     # Dense fixed-k baseline entry point
├── build_ambical_profile.py      # Calibration profile builder
├── llm_connectors/  query_translation/  evaluation/  analysis/  parameter_tuning/  utils/  prompt_templates/
├── profiles/<doc>/               # Published calibration profiles (profile + manifest + completion marker + SHA-256 sidecars)
├── chroma_db_<doc>_bge_m3_fp16/  # Six document databases (BGE-M3, chunk 200 / overlap 100)
├── question_datasets/<doc>_english_qa/   # Frozen 200-question datasets (100 answerable + 100 unanswerable)
├── config_ambical_<doc>_v51_formal200_full.json   # AmbiCal Full formal experiment configs (six documents)
├── config_dense_<doc>_v51_*.json # Dense fixed-k formal configs (k = 1/5/10)
├── config.json                   # Default entry config (= my_tkpm AmbiCal formal config)
├── my_tkpm_v51_method_parameter_lock_20260926_01.json   # Frozen method parameter lock
├── testers/                      # Unit tests (method, entry point, calibration, dataset structure)
├── cross_domain_data/            # Source corpora of the six documents + English manual
├── Dockerfile, requirements.txt, main.py
```

The six benchmark documents are Indonesian Brigade Pangan (`id_brigade_pangan`), Indonesian OJK finance handbook (`ojk_finance`), Filipino Bungkalan organic agriculture manual (`ph_bungkalan`), Malay TKPM guideline (`my_tkpm`), Vietnamese infrastructure manual (`vn_infrastructure`), and Vietnamese low-emission rice handbook (`vn_low_emission_rice`).

## Environment

- Windows (formal experiments were run on an RTX 5070 Ti), Python 3.10 with `RAG/requirements.txt`
- [Ollama](https://ollama.com) serving `qwen2.5:7b` (answering and translation) and `gemma3:12b` (judge)
- BGE-M3 embedding model (`BAAI/bge-m3`): downloaded from Hugging Face or loaded from a local directory; set the `embedding_model_name` field in each config to your copy's path
- Each formal config records the exact model digests, hardware, and software versions at run time

## Usage

All commands are run from `RAG/`.

**AmbiCal formal experiment** (one config per document; results go to a fresh output directory):

```bash
python rag_tester_ambical.py --config config_ambical_my_tkpm_v51_formal200_full.json
```

**Dense fixed-k baseline (k = 1/5/10):**

```bash
python rag_tester_dense_fixed.py
```

**Rebuild a calibration profile:**

```bash
python build_ambical_profile.py --config <calibration_config.json>
```

**Tests:**

```bash
python -m unittest discover -s testers -t .
```

In configs, the method is selected with `"retrieval_algorithms_to_test": ["ambical"]` and parameterized via the `ambical_parameters` block (`method_config` + `profile_path`). Method version identifier: `ambical-online-v5`; calibration algorithm version: `ambical-calibration-v2`. Precomputed calibration profiles are loaded from `RAG/profiles/<doc>/` and validated against the method configuration (utility signature) and the runtime collection scope before any experiment starts.

**Docker:** a containerized setup (Ollama service + dependencies) is available via `RAG/Dockerfile`:

```bash
docker build -t ambical RAG/
```

## Notes

The frozen method parameter lock pins all method hyperparameters (SHA-256 recorded in the calibration profiles). All method identifiers (`ambical-online-v5`, `ambical-calibration-v2`, `ambical-translation-v1`) are uniform across code, configurations, and artifacts. The dataset integrity manifests and freeze markers ship with the separate data package accompanying this code.
