# Test-Model-Thing (TMT)

[YouTube Video](https://youtu.be/9UERVVwpNew)

This is a small proof-of-concept language model (not an LLM) that incorporates the following (and some smaller features as well):
* Latent-space prediction
* Internal state + recurrent trace units (RTUs)
* Byte input/output
* Continuous data streaming
* Test-time training

The model is built with MLX, so it should run fine on all Apple Silicon devices. MLX on Linux has been tested by community members and it should work fine as well. On Windows, there are a few unofficial work-in-progress Pytorch ports but they aren't 1:1 compatible yet. I have managed to get the model running on WSL however, and it does train.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. Given further training and scaling up the model more interesting results could occur. I'm also using a very small dataset, so there is a lot more that can be fed into the model.

This model architecture was designed between July and August 2026 by me (a solo high school dev) and some Gemini (only pair programming, no agents). I wrote about a dozen prototypes before finalizing on this architecture. I write READMEs myself without AI.

Feel free to fork the training and benchmark code (everything is under MIT). I really encourage you to try things out and submit issues and pull requests. If you have compute (e.g. you are a lab or just have GPUs lying around), feel free to train larger models for longer periods of time as well, with credit. I really appreciate contributions to the project.

## Related papers

I’ve added the following references locally in the [`papers/`](./papers/README.md) directory for quick access:

- [DiffusionGemma Technical Report](https://arxiv.org/pdf/2608.00146)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/pdf/2412.06769)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
- [DeepSeek-Coder: When the Large Language Model Meets Programming](https://arxiv.org/pdf/2401.14196v2)

The project now also incorporates the ideas from those papers directly into the model stack. The default run uses a combined paper suite with latent reasoning, selective state-space memory, diffusion-style denoising, and code-oriented conditioning. You can switch the mix with:

```bash
python main.py my-model.safetensors chat --paper-suite latent
python main.py my-model.safetensors chat --paper-suite mamba
python main.py my-model.safetensors chat --paper-suite diffusion
python main.py my-model.safetensors chat --paper-suite coder
python main.py my-model.safetensors chat --paper-suite full
python main.py my-model.safetensors chat --paper-suite none
```

## Model output

For reproduction purposes, the dataset I trained my model on is ```simplewiki-20260801-pages-articles.xml.bz2```, from the Wikipedia dumps. The default model is now a larger ```dim = 768```, ```layers = 20``` configuration designed to fit comfortably in an 8 GB RTX 5060 while leaving room for AdamW optimizer state.

On CUDA-capable MLX builds, the entrypoint selects the GPU by default. Use ```--cpu``` only when a GPU is unavailable. The model is intentionally wider and deeper than the original 4.5m-parameter prototype, but the byte-at-a-time recurrent interface still limits kernel occupancy; GPU selection prevents accidental CPU execution, while larger matrix layers provide substantially more work per step.

Below is ```--frozen``` mode output after training a model for 5 minutes (you could train it for much longer, feel free to send in the results as a GitHub issue).

<img width="1127" height="634" alt="Screenshot 2026-09-19 at 4 53 44 PM" src="https://github.com/user-attachments/assets/ca1553de-0248-4f6d-a67d-2f6e30938d9e" />

Also below is some results from CoLA after training a model for ~30 minutes. GPT-1, as a comparison point, scored 45.4, so there is still some distance to go.

<img width="449" height="273" alt="Screenshot 2026-09-19 at 4 53 25 PM" src="https://github.com/user-attachments/assets/66cd054d-61c0-442a-a0bd-24352dac3f58" />


## Training your own model

Model weights (in ```.safetensors```) are not provided because GitHub doesn't like very large files. But, you can train your own model simply by initializing a ```venv``` and installing dependencies with ```pip install mlx``` on Mac or ```pip install mlx[cuda]``` on Linux with GPU or ```pip install mlx[cpu]``` on Linux with CPU. Then run ```main.py```.

```bash
python main.py <path> train
python main.py <path> chat

# explicit CPU fallback
python main.py <path> train --cpu

# customize size if your GPU has more or less memory
python main.py <path> train --dim 768 --layers 20

# does not save to disk
python main.py <path> chat --no-save

# does not modify weights
python main.py <path> chat --frozen

# does not save to disk or modify weights
python main.py <path> chat --no-save --frozen

# benchmark with CoLA
python benchmark.py <path> <epochs> <slice>

# benchmark on CPU instead
python benchmark.py <path> <epochs> <slice> --cpu
```

You will have to configure your own dataset to run dataset mode, but you should be able to run chat mode without modifying anything if you have weights already.

Once it begins training, you can safely ^C the program and it will save weights. It will also periodically save weights every so often. The saved weights include the internal memory so the model will remember that the next time it runs. You can launch into chat mode and the memory should carry on from whatever it was learning in training.

## A graphical view (partially outdated)

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference. Note that the arrow connecting the target latent to the CE loss should instead be the target byte to the CE loss.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />
