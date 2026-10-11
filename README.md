# Test-Model-Thing (TMT)

[YouTube Video](https://youtu.be/9UERVVwpNew)

This is a small proof-of-concept language model (not an LLM) that aims to incorporate the following (and some smaller features as well):
* Latent-space prediction
* Internal state + recurrent trace units (RTUs)
* Byte input/output
* Continuous data streaming
* Test-time training

It is not a finished work.

The model is built with MLX, so it should run fine on all Apple Silicon devices. MLX on Linux has been tested by community members and it should work fine as well. On Windows, there are a few unofficial work-in-progress Pytorch ports but they aren't 1:1 compatible yet. I have managed to get the model running on WSL however, and it does train.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. Given further training and scaling up the model more interesting results could occur. I'm also using a very small dataset, so there is a lot more that can be fed into the model.

This model architecture was designed between July and August 2026 by me (a solo high school dev) and some Gemini (only pair programming, no agents). I wrote about a dozen prototypes before finalizing on this architecture. I write READMEs myself without AI.

Feel free to fork the training and benchmark code (everything is under MIT). I really encourage you to try things out and submit issues and pull requests. If you have compute (e.g. you are a lab or just have GPUs lying around), feel free to train larger models for longer periods of time as well, with credit. I really appreciate contributions to the project.

## Model output

For reproduction purposes, the dataset I trained my model on is ```simplewiki-20260801-pages-articles.xml.bz2```, from the Wikipedia dumps. The 4.5m model has ```dim = 512``` and ```layers = 16```.

Below is ```--frozen``` mode output after training a model for 5 minutes (you could train it for much longer, feel free to send in the results as a GitHub issue).

<img width="1127" height="634" alt="Screenshot 2026-09-19 at 4 53 44 PM" src="https://github.com/user-attachments/assets/ca1553de-0248-4f6d-a67d-2f6e30938d9e" />

Also below is some results from CoLA after training a model for ~30 minutes. GPT-1, as a comparison point, scored 45.4, so there is still some distance to go.

<img width="449" height="273" alt="Screenshot 2026-09-19 at 4 53 25 PM" src="https://github.com/user-attachments/assets/66cd054d-61c0-442a-a0bd-24352dac3f58" />


## Install and train a model

Use Python 3.12, 3.13, or 3.14 and a functional MLX runtime.
From the checkout, create a virtual environment and install the CLI:

```bash
# Use python3.12, python3.13, or python3.14.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
tmt --version
```

Activation adds `.venv/bin` to PATH for this terminal.
Developers can use `python -m pip install -e '.[test]'` so source edits take effect directly and pytest is available.
For an optional persistent PATH configuration, see [the CLI guide](CLI.md#installation-and-path).
`python -m tmt` also runs the CLI.
The version uses installed package metadata.

Create a workspace in any folder, then supply raw-byte files:

```bash
mkdir my-workspace
cd my-workspace
tmt init
# Select a tiny model for this example.
printf '{"dim":4,"layers":1,"spread":4}\n' > model.json
printf abcd > data/train/tiny.bin
tmt train --data 'data/train/*' --run first --updates 3
tmt train runs/first/model.safetensors --data 'data/train/*' --resume --updates 2
tmt evaluate runs/first/model.safetensors --data 'data/train/*' --windows 1 2
tmt generate runs/first/model.safetensors --prompt 'The ' --output sample.bin --bytes 4
tmt benchmark runs/first/model.safetensors
```

Fresh train and sweep commands use `model.json` in the current folder by default.
`--config PATH` selects another model JSON for a fresh run.

Resume uses checkpoint settings and adds updates with a new data pass. It does not restore the file cursor or RNG state.

The trainer saves every 500 updates and at normal completion. Ctrl-C does not save. Keep the last successful checkpoint.

Read [the CLI guide](CLI.md) for workspace files, explicit checkpoint paths, sweeps, optional CoLA, and limits.
Read the [command manual](src/tmt/commands.md) or use `tmt help`.
Use [train](src/tmt/commands.md#train) for run and resume options.
The tiny example checks commands. It does not establish model quality.

## A graphical view (partially outdated)

_I'm probably going to redo this in Manim or similar software sometime soon. Expect this part to change significantly soon_

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference. Note that the arrow connecting the target latent to the CE loss should instead be the target byte to the CE loss.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />
