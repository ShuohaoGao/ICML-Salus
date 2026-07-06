## CompDiag-Bench & Salus

### Introduction

This benchmark evaluates medical LLM agents on OSCE-style clinical cases. It supports both static diagnosis from a full record and a simplified interactive setting where the agent can request auxiliary examinations before giving a final diagnosis.

### Project Structure

```text
Raw medical records -> [OSCE-extract] -> Standardized OSCE data
                                             |
                         +-------------------+-------------------+
                         |                                       |
                [static_diagnosis]                    [diagnostic_test]
              Full-record diagnosis             Auxiliary-exam interaction
                         |                                       |
                    Evaluation metrics                  Evaluation metrics
```

### 1. Static Diagnosis (`static_diagnosis/`)

This baseline sends the complete medical record to the LLM in one pass and evaluates the top-3 diagnosis accuracy.

```bash
cd static_diagnosis
python run_API.py
```

You can modify the diagnosis prompt or implement multi-agent diagnosis strategies under `static_diagnosis/doctor_implement`.

### 2. Diagnostic Test (`diagnostic_test/`)

This setting starts from the patient's history and physical examination results. The doctor agent can request auxiliary examinations, then must provide a final diagnosis.

```bash
cd diagnostic_test
python run_API.py
```

You can add new doctor-agent architectures under `diagnostic_test/doctor_implement`.

### 3. OSCE Extraction (`OSCE-extract/`)

This directory contains scripts for extracting and preprocessing raw medical records into standardized OSCE-style case data.

### 4. Think Generator (`ThinkGenerator/`)

This directory contains utilities for generating reasoning traces and distillation-oriented data.

### 5. Shared Utilities (`utils/`)

Shared infrastructure lives in `utils`, including:

- LLM API wrappers in `call_LLM.py`
- OSCE data loading in `data_loader.py`
- Parallel processing and XML-label extraction in `tools.py`
- Shared evaluation arguments in `evaluate_config.py`
- Diagnosis grading utilities in `diagnosis_eval.py`
- Auxiliary-exam simulation utilities in `medical_exam.py`

LLM API configuration should follow `utils/API_key_example.json`.

### Environment

- Python >= 3.12
- Core dependencies: `openai`, `requests`, `tqdm`
- Optional data analysis dependencies: `polars`, `matplotlib`, `seaborn`
- Optional local serving dependency: `vllm`
- LLM API configuration: copy `utils/API_key_example.json` to `utils/API_key.json` and fill in your API keys.

### Installation With pip

```bash
pip install openai requests tqdm
```
