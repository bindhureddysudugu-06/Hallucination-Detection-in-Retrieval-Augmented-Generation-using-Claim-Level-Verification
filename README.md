# Claim-Level Hallucination Detection in RAG (Local No-API Baseline)

## Overview
This project adopts a local no-API baseline of claim-level hallucination detection in Retrieval-Augmented Generation (RAG). The primary objective is to examine whether an answer contains claims that cannot be substantiated by breaking down the answer into smaller units of fact and verifying each claim against the evidence that is retrieved.

Instead of using an external LLM API, this implementation uses:
- **HotpotQA subset** of questionanswer data.
- semantic retrieval with sentence-transformers
- local similarity-based claim verification.
- rule-based answer-level aggregation

## Motivation
RAG enhances factual grounding by accessing supportive evidence prior to generating answers. Nevertheless, despite retrieved context, answers can still have hallucinated or unsubstantiated claims. Response-level assessment is usually too coarse as it might obscure a partial hallucination within an otherwise accurate response.

This project solves that problem by applying the concept of claim-level verification where:
1. an answer is divided into smaller claims,
2. every claim is compared to retrieved evidence,
3. claims are given the labels, Supported, Partially Supported or Unsupported,
4. labels are pooled together into an answer-level assessment of hallucinations.

## Project Objective
The goal of the current project is to construct and test a claim-level hallucination detection pipeline to do retrieval-augmented question answering and how retrieval depth and similarity thresholds influence unsupported claim rates.
### Primary Research Question
Does claim-level verification give more informative and reliable analysis of hallucinations in retrieval-augmented question answering than response-level evaluation?

## Dataset
This project utilizes a subset of HotpotQA, a benchmark of multi-hop question answering. HotpotQA would be appropriate in this work due to the fact that, in most questions, a combination of evidence based on multiple supporting documents is needed and hence is applicable in the study of evidence grounding and unsupported claims.

In the case of the experiments implemented:
- **100 questions** were used
- **992 documents** were processed
- Chunks After the preprocessing, 1023 chunks were created.

## Project Structure
Rag_project/
│
|-- hotpotqa_for_rag.py
|-- rag_hallucination_local.py
|-- graphs.py
|-- hotpot_dev_distractor_v1.json
│
|-- prepared_data/
│   |-- questions.jsonl
│   |-- documents.jsonl
│
|-- exp1_outputs/
|-- exp2_outputs/
|-- exp3_outputs/
│
|-- poster_graphs
