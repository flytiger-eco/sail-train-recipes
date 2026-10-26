# SAIL Training Recipes

Performance tuning recipes and best practices for training large models on Zhenwu AI accelerators.

```text
sail-train-recipes/
├── dockers/              # Dockerfiles and patches for SAIL training images
└── best_practice/        # Training/tuning recipes for mainstream models
    ├── cosmos3/
    ├── qwen3.5/
    ├── qwen3Omni/
    └── wan2.2/
```

## Supported Models

| Model Family | Path | Description |
|--------------|------|-------------|
| Qwen3.5-VL | [`best_practice/qwen3.5`](./best_practice/qwen3.5) | Dense (2B/4B/9B/27B) and MoE (35B-A3B) recipes. |
| Cosmos3 | [`best_practice/cosmos3`](./best_practice/cosmos3) | Official source plus M890 Nano/Super training changes. |
| WAN 2.2 5B | [`best_practice/wan2.2`](./best_practice/wan2.2) | M890 TI2V full-parameter training recipe. |
| Qwen3-Omni | [`best_practice/qwen3Omni`](./best_practice/qwen3Omni) | Qwen3-Omni-30B-A3B thinker-only training (single / multi-node). |

## Contributing


Contributions are welcome. If you have a new model recipe, Dockerfile, or tuning result:

1. Add the script or Dockerfile under the appropriate folder.
2. Add or update the `README.md` in the model folder with usage details.
3. Update the supported-models table in this README with a link to the new folder.
4. Open a pull request with a concise description of the Zhenwu AI accelerator variant and performance notes.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full contribution guide.


## License

This repository is released under the [MIT License](./LICENSE). Individual scripts retain their upstream licenses.
