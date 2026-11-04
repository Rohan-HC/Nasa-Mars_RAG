# NASA Mars RAG benchmark

- Dataset: `evaluation\datasets\mars_eval_v1.json`
- Execution mode: `live`
- Cases: 20
- Retrieval K: 4

| Metric | Result |
|---|---:|
| Retrieval Hit@K | 94.1% |
| Source Recall@K | 89.2% |
| Answer correctness | 70.0% |
| Citation validity | 95.0% |
| Evidence-supported answers | 64.7% |
| Correct refusal | 100.0% |
| Mean response latency | 10.221 s |

## Interpretation

Source recall counts each verified document-page pair. Answer correctness is deterministic concept coverage, not an LLM judge. Citation validity checks syntax and whether labels exist; it does not prove that a cited passage supports a claim. Evidence support checks whether required benchmark concepts occur in the cited passages and remains a lexical proxy rather than full factual verification.

## Case results

| ID | Category | Correct | Citation | Evidence | Latency |
|---|---|---:|---:|---:|---:|
| fact-01 | factual_retrieval | yes | yes | yes | 15.910 s |
| fact-02 | factual_retrieval | yes | yes | yes | 5.086 s |
| fact-03 | factual_retrieval | yes | yes | yes | 8.374 s |
| fact-04 | factual_retrieval | no | yes | no | 7.252 s |
| fact-05 | factual_retrieval | yes | yes | yes | 10.464 s |
| fact-06 | factual_retrieval | yes | yes | yes | 5.925 s |
| fact-07 | factual_retrieval | yes | yes | yes | 7.412 s |
| fact-08 | factual_retrieval | yes | yes | yes | 4.829 s |
| fact-09 | factual_retrieval | yes | yes | yes | 5.986 s |
| multi-01 | multi_evidence | no | yes | no | 10.797 s |
| multi-02 | multi_evidence | yes | yes | yes | 11.013 s |
| multi-03 | multi_evidence | yes | yes | no | 11.173 s |
| multi-04 | multi_evidence | no | yes | no | 10.012 s |
| compare-01 | document_comparison | yes | yes | yes | 28.291 s |
| compare-02 | document_comparison | no | yes | yes | 23.251 s |
| compare-03 | document_comparison | no | yes | no | 20.041 s |
| unanswerable-01 | unanswerable | yes | yes | n/a | 3.081 s |
| unanswerable-02 | unanswerable | yes | yes | n/a | 2.994 s |
| hallucination-01 | hallucination_prone | yes | yes | n/a | 8.358 s |
| hallucination-02 | hallucination_prone | no | no | no | 4.167 s |
