# Hallucination Progression MAP Network
**William Palmer<sup>1</sup>, Albert Powers<sup>2</sup>, Tyrone D. Cannon<sup>1, 2</sup>, Ilker Yildirim<sup>1</sup>**\
<sup>1</sup>Yale University Department of Psychology\
<sup>2</sup>Yale University Department of Psychiatry

## Summary
**Aim**: create a computational model of hallucination progression incorporating both low-level perceptual distortions and high-level scene expectations using a Forward Graphics Engine and Inverse Maximum A Posteriori (MAP) Graphics network based on AlexNet. 

**CPC2024 Poster**: https://drive.google.com/file/d/1S41mHeREbUjruriSWvz5Tmo6sUc3Qdv1/view

## Image Composition
* Agents/Objects
    - Items: none, cap, camera, boot, bird, cat, dog, baby, woman, man
    - `agents.tar.gz`: https://drive.google.com/file/d/1_xaf_kcyGgTsM95XCGuVhy_BcS72rnCW/view?usp=sharing
    - Source: https://free3d.com/3d-models/
* Scenes
    - Items: bathroom, bedroom, dining room, grey-white room, kitchen, living room, staircase, study, tea room
    - `scenes.tar.gz`: https://drive.google.com/file/d/1IZ3sua18nlG2V3F6SnIvSmbe2517l0mH/view?usp=sharing
    - Source: https://mitsuba.readthedocs.io/en/latest/src/gallery.html

## Dependencies
### Server
* Operating System: Red Hat Enterprise Linux 8.8 (Ootpa)
* Architecture: x86-64
* Workload Manages: Slurm 23.02.7
* GPU: NVIDIA A100

### Languages
#### Python
* numpy
* pandas
* matplotlib
* mitsuba
* pytorch
* sklearn
* skimage
* scipy

#### Julia
* Distributions
* ProgressMeter
* Gen
* Plots
* DataFrames
* CSV
* PyCall
* Conda
* ArgParse

## Building conda environment
1. `conda create -n "generative" python=3.11.5`
2. `conda activate generative`
3. `pip install mitsuba`
4. `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
5. `conda install scikit-learn`
6. `conda install scikit-image`
7. `conda install pandas`
8. `python -m pip install -U matplotlib`
* *Note*: There is a `environment.yaml` file.

## Main Scripts
### Forward Graphics Engine (Generative Function)
`bin/forward_graphics_engine/gen_train.jl`
* creates latent variables of scene and agents using dirichlet-categorical model ($p(y_{agent}|\alpha) = p(y_{agent}|\theta)p(\theta|\alpha)$, where $p(y_{agent}|\theta)\sim Cat(\theta)$ and $p(\theta|\alpha) \sim Dir(\alpha)$) for determining agent ($\alpha = \alpha_{previous} + n_{agent}$, where $\alpha_{previous}$ is the previous dirichlet prior and $n_{agents}$ is the predicted agent count from test images) and position using a uniform distribution

* inputs:
    - `iter`: (str) name of iteration
    - `dataset`: (str) name of dataset
    - `n_train`: (int) number of training images
    - `alpha`: (str) alpha for dirichlet distribution
    - `count`: (str) counts of previously predicted agents
* outputs:
    - labels: `images/${iter}/${dataset}/labels.csv`
    - alpha: `images/${iter}/${dataset}/${dataset}_alpha.txt`
* example: `julia gen_train.jl --iter=A --dataset=A1_CE01500_00_2 --n_train=9000 --alpha "[10000, 10000, 10000, 10000, 10000, 10000]" --count "[198, 204, 185, 192, 221, 202]"`

`bin/forward_graphics_engine/gen_test.jl`
* creates test latent variables for scene, agent, and position evenly spaced (100 each)
* inputs: None
* outputs: 
    - labels (only for "clean"): `images/test/labels.csv`
        * all images contain the same latent variables
* example: `julia gen_test.jl`

### Inverse MAP Graphics network

`bin/inverse_MAP_network/split_data.py`
* splits generated dataset into training and validation sets (4:1 ratio)
* input:
    - `iter`: (str) name of iteration
    - `dataset`: (str) name of dataset
* output:
    - split directories: `images/${iter}/${dataset}/{train, valid}/{scene}_{agent}_{number}.png`
    - split labels: `images/${iter}/${dataset}/{train, valid}/labels.csv`
* example: `python split_data.py --iter=A --dataset=A1_CE01500_00_2`

`bin/inverse_MAP_network/train_MAPnet.py`
* trains MAP network's (AlexNet) classifier and confidence unit using Adam with a batch size of 20
* input:
    - `iter`: (str) name of iteration
    - `dataset`: (str) name of dataset
    - `model`: (str) name of previously trained model or "new" if no previous model
    - `lr`: (float) learning rate for Adam
    - `n_epochs`: (int) number of training epochs
    - `n_scenes`: (int) number of rooms
    - `n_agents`: (int) number of agents/objects
* output: 
    - training and validation loss: `images/${iter}/${dataset}/${dataset}_training_loss.csv`
    - saved network classifier units: `images/${iter}/${dataset}/${dataset}_classif.pt`
    - saved network confidence unit: `images/${iter}/${dataset}/${dataset}_conf.pt`
* example: `python train_MAPnet.py --iter=A --dataset=A1_CE01500_00_2 --model=A1_CE01500_00_1 --lr=0.0001 --n_epoch=75 --n_scenes=9 --n_agents=10`

`bin/inverse_MAP_network/test_MAPnet.py`
* tests specified MAP network on specified test set
* input:
    - `iter`: (str) name of iteration
    - `model`: (str) name of trained model
    - `n_scenes`: (int) number of rooms
    - `n_agents`: (int) number of agents/objects
    - `test_type`: (str) type of distortion (opts: "clean", "noisy", "blurred", "color", "cedge", "edge", "complex", "mixed")
    - `dval`: (float) intensity of distortion
    - `conf`: (float) cutoff for confidence count
* output:
    - model predictions: `images/${iter}/${model}/${model}_${test_type}_pred.csv`
    - agent prediction count: `images/${iter}/${model}/${model}_${test_type}_pred_nagent.txt`
* example: `python test_MAPnet.py --iter=A --model=A1_CE01500_00_2 --n_scenes=9 --n_agents=10 --test_type=cedge --dval=0.05 --conf=0.0`

`bin/inverse_MAP_network/eval_MAPnet.py`
* evaluates specified MAP network on specified test set predictions
* input:
    - `iter`: (str) name of iteration
    - `model`: (str) name of trained model
    - `test_type`: (str) type of distortion (opts: "clean", "noisy", "blurred", "color", "cedge", "edge", "complex", "mixed")
* output:
    - model scene errors: `images/${iter}/${model}/${model}_${test_type}_scene_error.csv`
    - model agent errors: `images/${iter}/${model}/${model}_${test_type}_agent_error.csv`
    - model halluciantion rate: `images/${iter}/${model}/${model}_${test_type}_halluc_rate.txt`
* example: `python eval_MAPnet.py --iter=A --model=A1_CE01500_00_2 --test_type=cedge`

### Other
*Note: There were issues creating all images in a Julia or python script [Weird variable leak issue](https://github.com/mitsuba-renderer/drjit/issues/87)*
`bin/utils/gen_images.py`
* inputs:
    - `model`: (str) name of the model (essentially place to generate images)
    - `row`: (int) row of the dataframe to create
    - `test_type`: (string) type of distortion (opts: "clean", "noisy", "blurred")
    - `dval`: (float) intensity of distortion
    - `spp`: (int) sample per pixel for mitsuba
* output:
    - image: `images/${iter}/${model}/${room}_${agent}_${row+1}.png`
* example: `python gen_images.py --model=A/A1_CE01500_00_2 --row=0 --test_type=clean --dval=0.05 --spp=512`

## Driver Scripts
### Setup
0. Download and unzip the `scene.tar.gz` and `agent.tar.gz` files from Google Drive links above.
1. `bin/forward_graphics_engine/setup_julia.jl`
    * Creates Julia environment with required packages
    * input:
        - None
    * output:
        - environment directory: `forward_graphics_engine/halluc_prog`
    * example: `julia setup_julia.jl`
2. `bin/inverse_MAP_network/setup_MAPnet.py`
    * instantiates MAP network (AlexNet) and freezes all convolutional layers
    * input:
        - None
    * output:
        - saved network: `lib/adapted_AlexNet.pt`
    * example: `python setup_MAPnet.py`
3. `bin/create_models.py`
    * creates required files for running models
    * input:
        - model specifications: `lib/model_specifications.csv`
            - sample of required information:

        | model         | family | age   | expectation | distortion | dval | progression | confidence | lr    |
        |---------------|--------|-------|-------------|------------|------|-------------|------------|-------|
        | A1_CL00050_00 | noconf | 10000 | 0           | clean      | 0.05 | 0           | 0          | 0.001 |
        | A1_BL00200_00 | noconf | 10000 | 0           | blurred    | 0.20 | 0           | 0          | 0.001 |
    * output:
        - model confidence threshold over time: `lib/${model}/conf.txt`
        - model distortion value over time: `lib/${model}/dval.txt`
        - model learning rate over time: `lib/${model}/lr.txt`
        - model number of training epochs over time: `lib/${model}/n_epoch.txt`
        - model number of training images over time: `lib/${model}/n_train.txt`
        - model parameters (name, distortion, and expectation): `bin/model_params.csv`
        - slurm script for submitting models: `bin/submit_models.sh`
            * *Note*: It is assumes that the first iteration run is `A` and that all models should be submitted to run for 3 time points (`0 2`), which takes about 2 days. Change these values in the final lines when running different iterations or time points.
    * example: `python create_models.py`
4. `bin/run_baseline_example.sh`
    * Runs baseline model which serves the bases for all subsequent models at time 0 (this initial model takes the longest because it is trained on 45000 images instead of 9000 images)
    * input:
        - None
    * output:
        - baseline alpha values: `images/${iter}/baseline_alpha.txt`
        - baseline classifier network: `images/${iter}/baseline_classif.pt`
        - baseline confidence unit: `images/${iter}/baseline_conf.pt`
        - baseline training loss: `images/${iter}/baseline_training_loss.csv`
        - baseline generated images: `images/${iter}/labels.csv`
        - baseline training images: `images/${iter}/train/labels.csv`
        - baseline validation images: `images/${iter}/valid/labels.csv`
    * example: `sbatch run_baseline_example.sh`

### Run Models
`sbatch submit_models.sh`
* submits models to Slurm scheduler and runs following two scripts
1. `run_models.sh`
    * call `model.sh` with specified inputs to run models
    * input:
        - `iter`: (str) name of iteration
        - `begin`: (int) initial time point (zero indexed)
        - `end`: (int) end time point
    * output:
        - see below
    * example: `./run_models.sh A 0 2`
2. `model.sh`
    * runs model with specified number of iteration by calling the above scripts in "Main Scripts"
    * input
        - `iter`: (str) name of iteration
        - `model`: (str) name of model family
        - `start_iter`: (int) iteration to start run of model
        - `end_iter`: (int) number of iterations to run of the model
        - `init_alpha`: (int) initial alpha values for dirichlet distribution
        - `test_type`: (str) type of distortion (opts: "clean", "noisy", "blurred", "color", "cedge", "edge", "complex", "mixed")
        - `expect`: (bool: 0, 1) expectation updating is performed over time points
    * output
        - outputs of all above scripts in model directories: `images/${iter}/${model}_${start_iter}, ${model}_${start_iter+1},..., ${model}_${end_iter}}`
    * example: `./model.sh A1_NO00110_00 0 4 "[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]" noisy 1`
