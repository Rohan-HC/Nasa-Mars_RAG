# NASA Mars RAG benchmark

- Dataset: `evaluation\datasets\mars_eval_v1.json`
- Execution mode: `live`
- Cases: 20
- Retrieval K: 4

| Metric | Result |
|---|---:|
| Retrieval Hit@K | 94.1% |
| Source Recall@K | 94.1% |
| Answer correctness | 65.0% |
| Citation validity | 85.0% |
| Evidence-supported answers | 58.8% |
| Correct refusal | 100.0% |
| Mean response latency | 23.262 s |

## Interpretation

Answer correctness is deterministic concept coverage, not an LLM judge. Citation validity checks syntax and whether labels exist; it does not prove that a cited passage supports a claim. Evidence support checks whether required benchmark concepts occur in the cited passages and remains a lexical proxy rather than full factual verification.

## Case results

| ID | Category | Correct | Citation | Evidence | Latency |
|---|---|---:|---:|---:|---:|
| fact-01 | factual_retrieval | yes | yes | yes | 13.809 s |
| fact-02 | factual_retrieval | yes | yes | yes | 10.454 s |
| fact-03 | factual_retrieval | yes | yes | yes | 15.860 s |
| fact-04 | factual_retrieval | no | yes | no | 14.133 s |
| fact-05 | factual_retrieval | yes | yes | yes | 14.621 s |
| fact-06 | factual_retrieval | yes | yes | yes | 12.223 s |
| fact-07 | factual_retrieval | yes | yes | yes | 15.669 s |
| fact-08 | factual_retrieval | yes | yes | yes | 10.438 s |
| fact-09 | factual_retrieval | yes | yes | yes | 14.392 s |
| multi-01 | multi_evidence | no | yes | no | 14.233 s |
| multi-02 | multi_evidence | yes | yes | yes | 34.428 s |
| multi-03 | multi_evidence | yes | yes | no | 27.091 s |
| multi-04 | multi_evidence | no | yes | no | 36.209 s |
| compare-01 | document_comparison | no | no | no | 55.260 s |
| compare-02 | document_comparison | no | yes | yes | 58.317 s |
| compare-03 | document_comparison | no | no | no | 26.234 s |
| unanswerable-01 | unanswerable | yes | yes | n/a | 19.357 s |
| unanswerable-02 | unanswerable | yes | yes | n/a | 17.593 s |
| hallucination-01 | hallucination_prone | yes | yes | n/a | 15.865 s |
| hallucination-02 | hallucination_prone | no | no | no | 39.044 s |
