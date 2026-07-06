# ICML-Salus

This repository contains the code for **Salus: Strategic Diagnostic Testing for Complex Diagnosis via Multi-Agent Reinforcement Learning**, including the CompDiag-Bench
benchmark pipeline and training scripts for medical LLM agents.

For technical details, please see our ICML page:
https://icml.cc/virtual/2026/poster/64732. The page includes the poster,
slides, and video.

## Project Structure

```text
ICML-Salus/
├── benchmark/          # CompDiag-Bench evaluation and data-construction code
│   ├── OSCE-extract/   # Convert raw medical records into OSCE-style cases
│   ├── ThinkGenerator/ # Generate reasoning traces and distillation data
│   ├── static_diagnosis/
│   │                   # Full-record diagnosis baseline
│   ├── diagnostic_test/
│   │                   # Interactive auxiliary-exam diagnosis benchmark
│   └── utils/          # Shared LLM, data-loading, and evaluation utilities
└── train/
    ├── SFT/            # Supervised fine-tuning scripts
    └── GRPO/           # GRPO training notes and scripts
```

## Benchmark

The benchmark evaluates medical LLM agents on OSCE-style clinical cases.
It supports:

- **Static diagnosis**: the model receives the full patient record and returns a
  diagnosis.
- **Diagnostic test**: the model starts from history and physical examination
  information, requests auxiliary examinations, and then makes a final diagnosis.

See `benchmark/README.md` for detailed usage.

## Training

Training scripts are provided under `train/`.

- `train/SFT/` contains supervised fine-tuning scripts based on
  `ms-swift` Megatron.



## Data

The structured patient records are being de-identified. We will release the data
as soon as the privacy-removal process is complete.

## Citation

If you use this repository, please cite our paper:

```bibtex
@inproceedings{gao2026salus,
  title={Salus: Strategic Diagnostic Testing for Complex Diagnosis via Multi-Agent Reinforcement Learning},
  author={Gao, Shuohao and Chen, Xuanzhong and Luo, Lingxiao and Ding, Zilin and Han, Rong and Jiang, Rui and Chen, Ting},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
