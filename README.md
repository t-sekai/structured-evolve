# Structured Evolution for Search-Space Generation in ML Kernel Scheduling
Kevin Chan (tsekchan@stanford.edu), Newton Chen (hsinchen@stanford.edu)

## Summary

We are going to build and evaluate a structured LLM-guided system for generating faster ML kernels by combining OpenEvolve-style evolutionary search with Apache TVM MetaSchedule. We will demonstrate success with benchmark plots comparing three levels of LLM intervention: direct kernel/schedule generation, LLM-generated search spaces refined by TVM's autotuner, and LLM-evolved search-space generators refined by TVM's autotuner. The goal is to study whether LLMs are most effective as kernel writers, search-space designers, or search-space-generator designers, while keeping TVM MetaSchedule as a strong, structured baseline.

## Project Proposal

[View Proposal](CS348K_Project_Proposal.pdf)
