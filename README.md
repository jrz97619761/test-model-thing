# Test-Model-Thing (TMT)

[YouTube Video (Nov '26)](https://youtu.be/QdUGeTN_ouM)

[Old YouTube Video (Sep '26)](https://youtu.be/9UERVVwpNew)

## Overview

TMT is a work-in-progress MLX-based framework and command-line tool for experimenting with alternative language model architectures. It is in very, very early stages, so expect lots of missing, but roadmapped features and unimplemented ideas. I have a lot of plans for this, so stay tuned!

**TMT will forever be permissively open source (including all training code and data), because I firmly believe in the importance of open source in a world where frontier AIs are either closed or only open weight (with no code attached). Open source for the win!**

TMT runs on macOS (any Apple Silicon) and Linux and WSL (any CUDA-supported GPU, or CPU). AMD support is not available currently, as MLX doesn't natively support it, but feel free to port it yourself. Installation instructions in a below section.

As per the MIT License, please feel free to fork the project, try new ideas, and submit PRs. If you're building your own projects with TMT as a base, just make sure to credit appropriately. Contributions, feedback, and constructive criticism are very much welcome!

**Currently included:**

* A small proof-of-concept RNN block, inspired by recurrent trace units (RTUs) in place of backpropagation through time (BPTT)

* A base framework for implementing various layers/blocks and loss functions in code, that can directly plug into the default encoder and decoder. (Documentation WIP.)

* Byte pair encoding support, although you will need to import a BPE file from either HF or tiktoken yourself.

* CoLA (Corpus of Linguistic Acceptability) and bit-per-byte benchmarks. CoLA is a manual download now but will hopefully be automatic fetch soon.

* A command-line tool for training and benchmarking language models using these alternative designs, alongside sweeping through various combinations of sizes and blocks for comparison and experimentation.

**Planned:**
* A catalog of various recurrent blocks, such as Mamba, GDN, xLSTM, Titans/TTT, etc, in addition to lots of experimentation with custom designs.

* A catalog of various benchmarks.

* On-device parallel/batched training.

* Long term: RL and decentralized training. To really scale up the TMT effort internet-based decentralized training is the only way other than data centers, and the latter costs $$$. My vision is for ~10^5 to 10^7 consumer GPUs connected through the internet, training large scale models built off recurrent architectures.

* More things are planned, both short and long term, so stay tuned :)

_While I personally don't use LLM agents (due to $$$ requirements), LLM contributions are 100% welcome. I appreciate any suggestions for improvement! Note that, I also write documentation, like this README, by hand. The AGENTS.md in the repo is solely for other contributors._

## Installation

### TMT module (macOS)

First clone the repo via git. Python 3.12 to 3.14 are supported.

Then, run the following commands (you should see the version of TMT installed):

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install .
tmt --version
```

To install the dependencies for running unit tests:
```bash
pip install -e '.[test]'
```

You can also run the CLI with
```bash
python3 -m tmt
```

### TMT module (Linux/WSL)
Instructions WIP. Main difference is that the installation of MLX is slightly more complicated, especially for CUDA.

### Workspace initialiation

Once TMT is installed, you can initialize a TMT workspace by running:
```bash
mkdir workspace
cd workspace

tmt init
```

Some example TMT commands below:
```bash
tmt train --seed 325 --run run
tmt train runs/run/model.safetensors --seed 325 --resume

tmt generate runs/run/model.safetensors sample.bin "The " --seed 325

tmt evaluate runs/run/model.safetensors eval.json

tmt benchmark runs/run/model.safetensors benchmark.json --seed 325

tmt sweep grid.json sweep.json --seed 325
```

## Contributing

There aren't really that many rules to contributing here! Just be nice, abide by the GitHub community guidelines, and don't go _too_ off topic. I try to be as welcoming as possible, in the spirit of open source.

I have opened GitHub discussions for this repo so go discuss ideas, share cool results, ask questions, etc there. You can also open an issue or PR if you want to contribute directly! As mentioned previously, AI contributions are welcome. Thank you :)

## Side note about me and my vision

The original project, a miniature proof-of-concept RNN implementation, was something I (a high school dev) worked on throughout July and August of 2026. Given I had barely dabbled in ML prior to that I spent what was effectively 1.5 months trying ideas, making prototypes, and failing. _(I have been very interested in programming for the past decade or so though, so I do kind of have a head start.)_ Throughout that time, and since, I learned a lot about machine learning, however, and I still have a lot to learn (everyone does lol). I'm excited to keep going though! TMT will continue to grow just as I continue to learn.

When I first started to work on TMT I had a core vision (and still do): I want to create a completely open source continual learning language model framework with training 100% through decentralized means, as a replacement for closed source or open weight (but not open source code) LLM training and inference stacks.

Frontier companies are scaling their LLMs rapidly without giving us a single bit of their code, while ignoring alternative architectures and spending all their financial capital on data centers (with not enough work on algorithmic improvements)! All while they fearmonger about the technology to induce more regulation and direct time towards "safety" instead of continual learning. At the same time, experimental ideas (like recurrent models) aren't being scaled up enough to prove that they work, because people just don't have access to the compute that the labs have.

I wanted to change that (and, by extension, the general trajectory of the AI industry), which is why TMT exists, to hopefully allow rapid experimentation and decentralized scaling as soon as the required features are implemented. This would be worked on not just by humans, but also by LLMs that can test new, open source ideas through TMT 24/7. I think the vision is within reach, but getting there will take time and contributions from all of you. Thanks for helping :)
