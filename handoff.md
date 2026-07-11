# Comprehensive Project Handoff: Master's Course Project on Vulnerability Detection & Analysis

> **Document Purpose & Status**: This document serves as the complete, self-contained handoff for an **MSc / Graduate Course Project** on software vulnerability detection (`Identify Attacks`). It preserves 100% of the historical context, dataset investigations, technical architecture, and brainstorming discussions across all sessions.
> 
> **IMPORTANT NOTE ON PROJECT DIRECTION**: **No final decision has been locked in regarding which research direction to pursue next.** All ideas discussed below—including the original Source vs. LLVM IR comparison, the multi-threaded concurrency/remediation ("kill it") pivot, and joint multi-modal fusion—are presented as **Open Research Options & Ideas** for the student, team, and course instructors to evaluate and choose from.

---

## 1. Course & Project Background

### Core Topic: Identifying Attacks / Software Vulnerability Detection
This Master's research project explores automated vulnerability detection and analysis in C/C++ software using deep learning (`CodeBERT` / Large Language Models) and program analysis.

### The Foundational Research Question: C Source Code vs. LLVM IR
A primary motivation of the project is investigating how the **representation of code** impacts the ability of machine learning models to identify vulnerabilities:
* **C Source Code**: Contains high-level developer semantics, variable names, macros, comments, and human-readable syntactic structures. However, macros and typedefs can obscure underlying control flow.
* **LLVM Intermediate Representation (LLVM IR)**: Contains compiler-lowered semantics in Static Single Assignment (SSA) form, explicit control flow graphs (CFGs), exact memory allocation types, and unambiguous instruction ordering. However, variable names and high-level abstractions are stripped.

**Key Questions Explored**:
1. Does LLVM IR provide superior performance or generalization compared to raw C Source Code when detecting vulnerabilities?
2. How do models trained on synthetic benchmark datasets (`Juliet`) transfer to real-world vulnerabilities (`CompRealVul`) when using Source versus IR?
3. Can we extract deeper structural insights (e.g., thread boundaries, concurrency interactions) from Source or IR to actively remediate ("kill") vulnerabilities?

---

## 2. Infrastructure & Codebase Built (`src/`)

We have successfully designed, built, and verified the core dataset loading and processing infrastructure (`Milestone 1`). All code is located under `/Users/art/Desktop/MSC/Identify Attacks/project/src/`:

* `src/data_loader.py`:
  * **Function Body Extraction (`extract_function_body`)**: Implements precise curly-brace matching (`{` to `}`) to extract exact C function headers and bodies from multi-function source files.
  * **Dataset Loaders (`load_aligned_juliet`, `load_aligned_realvul`)**: Loads and aligns Hugging Face datasets into standardized `DatasetDict` structures (`train`, `validation`, `test`).
  * **`--toy` Mode**: Features a rapid verification flag (`toy=True`) that slices the first 50–100 samples for immediate local testing without downloading heavy models or running long computations.
* `src/train.py`:
  * Implements a Hugging Face `Trainer` loop for `microsoft/codebert-base` configured for binary classification (vulnerable vs. benign), including metric callbacks and model checkpointing.
* `src/eval.py`:
  * Computes standard evaluation metrics: `Accuracy`, `Precision`, `Recall`, `Micro/Macro F1-Score`, and `ROC-AUC`, exporting classification reports.
* `requirements.txt`:
  * Declares required dependencies: `torch`, `transformers`, `datasets`, `pandas`, `scikit-learn`, `tqdm`.
* `run_toy.sh`:
  * End-to-end local validation script that verifies data loading, function extraction, alignment, and tokenization.

---

## 3. Dataset Discoveries & Exact Alignment Status

We performed deep-dive inspections across both synthetic and real-world datasets and achieved high-precision alignments:

### A. Synthetic Benchmark: Juliet (`C Source` + `LLVM IR`)
* **Source Dataset**: `hwiwonl/nist-juliet-c` (Contains `good` and `bad` C/C++ source snippets).
* **IR Dataset**: `CCompote/Juliet_LLVM` (Contains LLVM IR functions across `train`, `validation`, `test` splits).
* **Alignment Accomplished**: Achieved **100% 1-to-1 alignment** by mapping instance IDs and stripping `-good`/`-bad` filename suffixes (`cweXYZ_...`).
* **Concurrency / Threading Subset Discovered**:
  * Juliet models synchronization using standardized helper abstractions: `stdThreadLockCreate()`, `stdThreadLockAcquire()`, `stdThreadLockRelease()`, and `stdThreadLockDestroy()`.
  * Specific concurrency/threading CWEs verified in Juliet:
    * `CWE-366` (Race Condition within a Thread): 36 samples.
    * `CWE-367` (Time-of-check Time-of-use / TOCTOU Race Condition): 36 samples.
    * `CWE-667` (Improper Locking): 18 samples.
    * `CWE-832` (Unlock of a Resource that is not Locked): 18 samples.
    * `CWE-364` (Signal Handler Race Condition): 18 samples.
    * *Resource/Synchronization Management*: `CWE-404` (384 samples), `CWE-563` (512 samples).

### B. Real-World C: CompRealVul (`C Source` + `LLVM IR`)
* **Source Dataset**: `CCompote/CompRealVul_C` (Raw C function code from open-source repositories).
* **IR Dataset**: `CCompote/CompRealVul_LLVM` (Lowered LLVM IR across `train`, `validation`, `test`).
* **Alignment Accomplished**: Achieved **98.81% 1-to-1 alignment** by matching exact function names (`fun_name`).
* **Concurrency / Threading Subset Discovered**:
  * **Total Samples**: 18,538 function samples.
  * **Threading/Concurrency Subset**: **4,111 samples (~22.18% of the dataset)** contain concurrency keywords (`mutex`, `lock`, `race`, `pthread`, etc.) or concurrency-related CWEs.
  * **CWE-362 (Race Condition / Shared Resource Improper Synchronization)**: Contains **294 explicit samples**.
  * **Real-World Patterns**: Concurrent network request modifications (`req_inet->opt`), file operation TOCTOU races (`my_redel`), and unreleased mutexes.

---

## 4. Hardware & Environment Strategy

> [!CRITICAL]
> **Local Mac Environment (`/Users/art/Desktop/MSC/Identify Attacks/project`)**:
> - **Role**: Code development, pipeline architecture, data inspection/filtering, static extraction testing, and running fast pipeline verification via `--toy` mode (`bash run_toy.sh`).
> - **Limitation**: Do not run full-scale PyTorch training loops (`CodeBERT` on 18,000+ samples across multiple epochs) locally on the Mac due to compute/memory limits observed during Milestone 1 setup.

> [!CRITICAL]
> **Remote Kaggle Execution**:
> - **Role**: All heavy GPU model training, multi-epoch fine-tuning, cross-dataset transfer experiments, and full-scale evaluations must be executed on **Kaggle** (or equivalent cloud GPU environments). Scripts designed locally (`src/*.py`) are structured so they can be uploaded directly to Kaggle notebooks or scheduled tasks.

---

## 5. Brainstormed Research Ideas & Direction Options

**No decision has been made yet.** The following four research directions represent the full menu of ideas brainstormed across our sessions. The next agent should assist the student/team in evaluating, combining, or choosing among these ideas based on course requirements and personal interest:

### Option 1: The Core Source vs. LLVM IR Comparative & Transfer Study (Original Scope)
* **Concept**: Complete the comprehensive 2x2 experimental matrix comparing CodeBERT vulnerability detection on **C Source Code** versus **LLVM IR**.
* **Key Experiments**:
  1. Train & evaluate on `Juliet Source` vs. `Juliet LLVM IR`.
  2. Train & evaluate on `CompRealVul Source` vs. `CompRealVul LLVM IR`.
  3. **Cross-Dataset Transferability**: Train on synthetic `Juliet` (Source & IR) and test zero-shot generalization on real-world `CompRealVul` (Source & IR).
* **Pros**: Clean, rigorous experimental setup; direct baseline against existing literature; infrastructure (`data_loader.py`, `train.py`) is already 100% built and aligned.

### Option 2: Multi-Threaded Concurrency Analysis, Thread Division, & Automated Remediation ("Kill It")
* **Concept**: Pivot to a targeted analysis of concurrent/multi-threaded code. Extract thread divisions from source code, identify thread-specific vulnerabilities, and automatically remediate ("kill") them.
* **Key Steps & Ideas**:
  1. **Thread Division Extraction**: Build `src/concurrency_parser.py` to identify thread creation (`pthread_create`, `std::thread`), entry points, shared variables, and lock/unlock boundaries (`stdThreadLockAcquire`).
  2. **Thread-Level Detection**: Filter datasets for the ~4,111 concurrency samples (`CWE-362`, `CWE-667`, `CWE-832`, `CWE-366`) and train a specialized concurrency vulnerability detector.
  3. **Automated Remediation ("Kill It")**: Implement `src/remediate.py` using a **Hybrid Static + LLM Patch Generation** engine that localizes unguarded shared variables or improper lock orders, generates synchronization fixes (adding mutexes / reordering locks), and tracks a **Remediation Success Rate (RSR)** metric.
* **Pros**: Highly novel and impressive for an MSc thesis; moves beyond passive classification into active, self-healing agentic code repair; ~22% of real-world data is ready.

### Option 3: Multi-Modal Fusion (Joint C Source + LLVM IR Representation Learning)
* **Concept**: Instead of comparing Source vs. IR separately, build a dual-encoder or fused neural architecture that ingests **both** C Source Code (high-level syntax/semantics) and LLVM IR (low-level SSA/CFG semantics) simultaneously for the same function.
* **Key Experiments**:
  1. Concatenate or cross-attend Source tokens with IR tokens inside a Transformer architecture (`CodeBERT` / `GraphCodeBERT`).
  2. Evaluate whether joint representation learning catches complex vulnerabilities that neither representation catches in isolation.
* **Pros**: Directly leverages our 1-to-1 dataset alignment (`load_aligned_juliet`, `load_aligned_realvul`); strong contribution to representation learning in software security.

### Option 4: Graph-Based & Data-Flow Concurrency Analysis
* **Concept**: Move from sequence-based token classification (`CodeBERT`) to Graph Neural Networks (`GNNs`) or `GraphCodeBERT` using Program Dependence Graphs (PDGs), Abstract Syntax Trees (ASTs), or LLVM IR Control Flow Graphs (CFGs).
* **Key Experiments**:
  1. Convert LLVM IR basic blocks or C source ASTs into graph representations where nodes are operations and edges are data/control flow.
  2. Specifically model data-flow edges across thread boundaries to detect race conditions (`CWE-362`) and memory corruption (`CWE-119`).
* **Pros**: Graph representations naturally capture compiler-lowered data flow and SSA properties better than flat token sequences.

---

## 6. How the Next Agent Should Proceed

When taking over this project in a new conversation:
1. **Acknowledge Full Context**: Confirm understanding that this is an MSc course project with all core data-loading and alignment infrastructure already implemented and tested (`src/data_loader.py`).
2. **Review Options Without Bias**: Present the 4 research options (`Source vs IR Comparison`, `Multi-Threaded Kill-It Remediation`, `Multi-Modal Fusion`, `Graph/Data-Flow Analysis`) or any combination to the user without assuming a direction has been chosen.
3. **Help Finalize Direction**: Answer any questions the user has regarding difficulty, compute requirements (Kaggle), or academic impact to help them make their final decision for the course.
4. **Execute the Chosen Path**: Once the user explicitly selects their preferred research direction, follow the existing modular code structure in `src/` to implement the required models, parsers, or evaluation loops.
