# Hallucination Progression MAP Graphics Network
**William Palmer<sup>1</sup>, Albert Powers<sup>2</sup>, Tyrone D. Cannon<sup>1, 2</sup>, Ilker Yildirim<sup>1</sup>**\
<sup>1</sup>Yale University Department of Psychology\
<sup>2</sup>Yale University Department of Psychiatry

## Summary
Create a computational model of hallucination progression incorporating both low-level perceptual distortions and high-level scene expectations using a Forward Graphics Engine and Inverse Maximum A Posteriori (MAP) Graphics network based on AlexNet

## Image Composition
* Agents
    - https://free3d.com/3d-models/
    - man, woman, baby, dog, cat, hamster, snake, bird, boat, airplane, helmet, cap, bottle
* Scenes
    - https://mitsuba.readthedocs.io/en/latest/src/gallery.html
    - bedroom, living-room, bathroom

## Scripts
### Dependencies
#### Python
* numpy
* pandas
* matplotlib
* mitsuba
* pytorch
* sklearn
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
6. `conda install pandas`
7. `python -m pip install -U matplotlib`

## Scripts
### Forward Graphics Engine (Generative Function)
`gen_train.jl`
* creates latent variables of scene and agents using dirichlet-categorical model ($p(y_{agent}|\alpha) = p(y_{agent}|\theta)p(\theta|\alpha)$, where $p(y_{agent}|\theta)\sim Cat(\theta)$ and $p(\theta|\alpha) \sim Dir(\alpha)$) for determining agent ($\alpha = \alpha_{previous} + n_{agent}$, where $\alpha_{previous}$ is the previous dirichlet prior and $n_{agents}$ is the predicted agent count from test images) and position using a uniform distribution

* inputs:
    - `dataset`: (str) name of dataset
    - `alpha`: (str) alpha for dirichlet distribution
    - `count`: (str) counts of previously predicted agents
* outputs:
    - labels: `images/{dataset}/labels.csv`
    - alpha: `images/{dataset}/{dataset}_alpha.txt`
* example: `julia gen_train.jl --dataset full_1 --alpha "[10000, 10000, 10000, 10000, 10000, 10000]" --count "[198, 204, 185, 192, 221, 202]"`

`gen_test.jl`
* creates test latent variables for scene, agent, and position evenly spaced (100 each)
* inputs: None
* outputs: 
    - labels (only for "clean"): `images/test/labels.csv`
        * all images contain the same latent variables
* example: `julia --test_type noisy --dval 0.1`

### Other
*Note: There were issues creating all images in a Julia or python script [Weird variable leak issue](https://github.com/mitsuba-renderer/drjit/issues/87)*
`utils/gen_images.py`
* inputs:
    - `model`: (string) name of the model (essentially place to generate images)
    - `row`: (int) row of the dataframe to create
    - `test_type`: (string) type of distortion (opts: "clean", "noisy", "blurred")
    - `dval`: (float) intensity of distortion (converted to int for "blurred")
    - `spp`: (int) sample per pixel for mitsuba

`utils/gen_utils.py`
* mitsuba utility functions used in `gen_images.py` to generate images

`utils/dirichlet.jl`
* defines Gen dirichlet distribution for `gen_{train, test}.jl`
* equation: $Dir(\alpha) = f(x_{1},...,x_{K}; \alpha_{1},...,\alpha_{K}) = \frac{1}{B(\alpha)}\prod^{K}_{i=1}x^{\alpha_{i}-1}_{i}$
* $\alpha$ parameter acts as a pseudo-count to generate experience based $\theta$ for $Cat(\theta)$

``

### Inverse MAP Graphics network

`split_data.py`
* splits generated dataset into training and validation sets (4:1 ratio)
* input:
    - `dataset`: (string) name of dataset
* output:
    - split directories: `images/uniform/{train, valid}/{scene}_{agent}_{number}.png`
    - split labels: `images/uniform/{train, valid}/labels.csv`
* example: `python split_data.py full_1`

`train_MAPnet.py`
* trains MAP network (AlexNet) after freezing all convolutional layers using Adam with a batch size of 20
* input:
    - `dataset`: (string) name of dataset
    - `model`: (string) name of previously trained model or "new" if no previous model
    - `lr`: (float) learning rate for Adam
    - `n_epochs`: (int) number of training epochs
* output: 
    - training and validation loss: `images/{dataset}/{dataset}_training_loss.csv`
    - saved network: `images/{dataset}/{dataset}.pt`
* example: `python train_MAPnet.py --dataset full_1 --model baseline_0 --lr 0.0001 --n_epoch 75`

`test_MAPnet.py`
* tests specified MAP network on specified test set
* input:
    - `model`: (string) name of trained model
    - `test_type`: (string) type of distortion (opts: "clean", "noisy", "blurred")
    - `dval`: (float) intensity of distortion (converted to int for "blurred")
* output:
    - model predictions: `images/{model}/{model}_{test_type}_pred.csv`
    - agent prediction count: `images/{model}/{model}_{test_type}_pred_nagent.txt`
* example: `python test_MAPnet.py --model full_1 --test_type noisy --dval 0.1`

`eval_MAPnet.py`
* evaluates specified MAP network on specified test set predictions
* input:
    - `model`: (string) name of trained model
    - `test_type`: (string) type of distortion (opts: "clean", "noisy", "blurred")
* output:
    - model scene errors: `images/{model}/{model}_{test_type}_scene_error.csv`
    - model agent errors: `images/{model}/{model}_{test_type}_agent_error.csv`
    - model halluciantion rate: `images/{model}/{model}_{test_type}_halluc_rate.txt`
* example: `python eval_MAPnet.py --model full_1 --test_type noisy`
`plot_trajectory.py`
* plots error trajectories across specified models
* input:
    - `model`: (str) name of model family
    - `count`: (int) number of models in family
    - `test_type`: (str) type of distortion (opts: "clean", "noisy", "blurred")
* output:
    - alpha plot: `results/{model}/{model}_{test_type}_alpha.png` 
    - scene error plot: `results/{model}/{model}_{test_type}_scene_error.png` 
    - agent error plot: `results/{model}/{model}_{test_type}_agent_error.png`
    - hallucination rate plot: `results/{model}/{model}_{test_type}_halluc_rate.png`
* example: `python plot_trajectory --model full_model --count 5 --test_type noisy`
## Full Model (Driver script)
`model.sh`
* runs model with specified number of iteration by calling the above scripts
* input
    - `model`: (str) name of model family
    - `start_iter`: (int) iteration to start run of model
    - `end_iter`: (int) number of iterations to run of the model
    - `init_alpha`: (int) initial alpha values for dirichlet distribution
    - `test_type`: (str) type of distortion (opts: "clean", "noisy", "blurred")
    - `lib/model/dval.txt`: textfile with (float) intensities of distortion (converted to int for "blurred") for each model iteration
    - `lib/model/lr.txt`: textfile with (float) learning rates for Adam for each model iteration
    - `lib/model/n_epoch.txt`: textfile with (int) number of training epochs for each model iteration
* output
    - outputs of all above scripts in model directories: `images/{baseline_0, {model}_{i},..., {model}_{iter}}`
* example: `./model.sh full 0 6 "[10000, 10000, 10000, 10000, 10000, 10000]" noisy`


## Running Scripts
### Create baseline model
1. Generate shared baseline dataset: ```julia gen_train.jl --dataset "baseline" --alpha "[10000, 10000, 10000, 10000, 10000, 10000]" --count "[0, 0, 0, 0, 0, 0]"```
2. Split baseline dataset for training: `python split_data.py --dataset="baseline"`
3. Train baseline model: `python train_MAPnet.py --dataset="baseline" --model="new" --lr=0.0001 --n_epoch=75`
4. Generate test and/or distorted dataset: `julia gen_test.jl` or `julia gen_distorted.jl --distortion "{blurred, noisy}" --dval {distortion_intensity}`
5. Test baseline model: `python test_MAPnet.py --model="baseline" --test_type={clean, blurred, noisy, color, edge, complex, mixed} --dval={distortion intensity}`

### Full (Expectation + Distortion) model
`./run_full_model.sh {num_iter} {test_type} {dval} {lr} {n_epoch}`
### Expectation-only model
1. Generate new clean training dataset based on predictions: ```alpha=`cat <alpha_file>`; count=`cat <count_file>`; julia gen_train.jl --dataset "expectation_<step>" --alpha "$alpha" --count "$count"```
2. Split new clean training dataset: `python split_data.py --dataset="expectation_<step>"`
3. Train new model: `python train_MAPnet.py --dataset="expectation"_<step>" --model={"baseline", "expectation_<step-1>"} --lr=0.0001 --n_epoch=75`
4. Test new model: `python test_MAPnet.py --model="expectation_<step> --test_type="clean"`
### Distortion-only model
