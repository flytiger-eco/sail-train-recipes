# Contributing

Thanks for your interest in contributing to the SAIL training recipes.

## Reporting Issues

- For a bug report, include the model/recipe path, target hardware (e.g. M890), image tag, and the full error message or training log.
- For a new recipe request, describe the model, the workload, and the hardware you want it tuned for.

## Local Setup

Most recipes run inside the Docker images built from [`dockers/`](./dockers):

1. Build or pull the image for your model (see the model's `README.md` under `best_practice/`).
2. Launch the container with the model repository and datasets mounted, following the run commands in the model README.
3. Run the tuning script and collect logs from `${OUTPUT_DIR}/${tag}/`.

## Submitting Pull Requests

1. Open an issue first and describe the change. **Every PR must be linked to an issue**; this repository enforces a PR-issue-link check.
2. Create a feature branch from `master` and keep the change scoped to one topic.
3. Follow the existing layout: training scripts and tuning entry points under `best_practice/<model>/`, image builds and patches under `dockers/`.
4. Update the model's `README.md` with usage details and performance numbers (hardware, image tag, workload shape, throughput). If you add a new model, also update the supported-models table in the root `README.md`.
5. Use concise commit messages with a conventional prefix (`feat:`, `fix:`, `docs:`, `chore:`).
6. Open the PR with a short description of the change, the target hardware, and the performance impact.

## Code Style

- Keep tuning scripts self-contained; expose knobs through environment variables with safe defaults, and avoid hard-coded machine-specific paths.
- Do not commit large binaries, datasets, checkpoints, logs, or credentials (`.gitignore` covers the common cases).
- Shell scripts should pass `bash -n`; Python scripts should run under the image's Python without extra dependencies.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](./LICENSE). Individual scripts copied from upstream projects keep their original licenses.
