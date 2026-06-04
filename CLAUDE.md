# CUDA-Triton-Learn

## What this repo is
Learning path from zero to AI infrastructure engineer.
Goal: hireable at CoreWeave, Baseten, vLLM team, Together AI, Modal, Replicate.

## Curriculum order
1. `p0_cpp_prereqs/` — C++ prerequisites (pointers, memory, CPU matmul)
2. CUDA Course P1 — GPU Architecture + first kernels
3. CUDA Course P2 — Kernel optimization + profiling
4. Triton Course — write production ML kernels in Python
5. Portfolio project (one of the 5 from AI_Infra_FastTrack.pdf)
6. OSS contribution + interview prep

## Session structure
- Each session has a folder with `notes.md` and one or more `.cpp` / `.cu` / `.py` files
- Build incrementally — one concept at a time
- Every file is committed to GitHub with a short note on what it demonstrates

## How we work
- Mentor (Claude) writes code in actual files, explains incrementally
- Student asks questions until concept clicks, then we move on
- Notes stay in `notes.md` in the session folder

## GPU setup
- GPU via RunPod or Lambda Labs, connected to this local session
- For P0 (C++ prereqs): no GPU needed, just g++ locally
