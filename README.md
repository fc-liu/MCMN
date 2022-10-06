# Pre-training to Match for Unified Low-shot Relation Extraction
Codes for our ACL 2022 paper: <b>Pre-training to Match for Unified Low-shot Relation Extraction.<b> [[Arxiv](https://arxiv.org/abs/2203.12274)]

## Usage
the data is on FewRel official page https://github.com/thunlp/FewRel,
please download and put into ./data directory.

### prerequirement
- Create a Python 3 environment (3.7 or greater), eg using `conda create --name MCMN python=3.9`
- Activate the environment: `conda activate MCMN`
- Install the dependency packages: `pip install -r requirements.txt`
### FLEX Task Setup
- Clone the repository: `git clone git@github.com:allenai/flex.git`
- enter into flex directory: `cd flex`
- Install the package locally with `pip install -e .`

* For detail usage of Flex, please refer to https://github.com/allenai/flex
### FLEX data preparation
- replace the file `flex/fewshot/challenges/__init__.py` with file `__ROOT__/challenge/__init__.py`(This step removes other unrelated tasks in FLEX and only keeps FewRel tasks.）
- Make dataset:
```bash
python -c "import fewshot; fewshot.make_challenge('flex');"
```

### FLEX supervised only models:
```bash
./test_flex_ft.sh
```

### FLEX pretrain+supervised Test
```bash
./test_flex_pt_ft.sh
```

### FewRel None-Of-The-Above Train and Test
#### For Supervise only model:
- na rate 0.15: `./test_nota_0.15_ft.sh`
- na rate 0.5: `./test_nota_0.5_ft.sh`
#### For Pretrain + Supervise model:
- na rate 0.15: `./test_nota_0.15_pt_ft.sh`
- na rate 0.5: `./test_nota_0.5_pt_ft.sh`

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.
