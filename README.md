# BioGuard: Bio-Inspired Sparse Prompt Guard

BioGuard is a research prototype exploring a small, sparse neural classifier as an inference-time filter for prompt-injection and jailbreak-like inputs. It uses fixed sparse graph topologies with trainable edge weights and a local tokenizer option.

This repository is experimental. It is not a complete or certified safety system, and it does not guarantee that malicious prompts will be detected or that benign prompts will always pass.

## Responsible AI principles

BioGuard is developed with four principles in mind:

- **Fairness:** evaluate false positives and false negatives across diverse prompt styles, languages, and user groups. The bundled seed data is too small to establish fairness.
- **Transparency:** expose the model score and threshold-based decision, document the data and evaluation setup, and report limitations. A score is not an explanation of model reasoning.
- **Accountability:** keep reproducible training settings and evaluation artifacts, review failures, and require human oversight for consequential decisions.
- **Privacy and safety:** prefer local inference when practical, avoid sending prompts to external services by default, and do not treat this classifier as a substitute for model-side safety controls.

## Status and evidence

The project provides code for model construction, training, inference, and a smoke test. The smoke test checks graph creation, forward passes, mini-training, and checkpoint loading. Its prompt checks are explicitly described as untrained/random-score expectations; passing the smoke test demonstrates software-path operation, not detection accuracy.

Use a held-out dataset with documented splits and baselines before reporting AUROC, F1, robustness, fairness, or generalization claims. The included compact suffix sweep is exploratory and is not a comprehensive GCG evaluation. No comparison against hosted ChatGPT, Gemini, or Claude models is included.

## Requirements

Python 3.10+ is recommended. Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Commands below assume the repository root is the current directory:

```bash
python smoke_test.py
python train.py --tokenizer local --topology bio_layered --n_nodes 200 --epochs 20
```

The training script writes checkpoints and metrics to `outputs/`. These generated artifacts are ignored by Git by default; keep any research artifact you intend to share in a separately reviewed release or data repository.

Run the optional attack sweep after training:

```bash
python multi_prompt_attack.py --checkpoint outputs/best_scg.pt --tokenizer local
```

## Use as an inference guard

```python
from guard import BioGuard

guard = BioGuard.from_checkpoint("outputs/best_scg.pt")
result = guard.check("Example prompt to inspect")
print(result.action, result.confidence)
```

Choose and validate the threshold against the intended use case. Do not use a single threshold or classifier decision as the sole basis for high-impact decisions.

## Main files

- `scg_model.py`: sparse graph model and topology builders
- `dataset.py`: seed and optional dataset loading
- `local_tokenizer.py`: offline tokenizer
- `train.py`: training and evaluation loop
- `guard.py`: checkpoint loading and inference wrapper
- `smoke_test.py`: software-path smoke test, not an accuracy benchmark
- `multi_prompt_attack.py`: exploratory suffix-optimization sweep

## External models and API keys

API keys are unnecessary to train and evaluate this local classifier. They are needed only for experiments that call hosted model APIs. Such experiments should document provider/model versions, prompt sets, decoding settings, costs, and safety boundaries.

## License

MIT. See [LICENSE](LICENSE).
