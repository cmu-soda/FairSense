# FairSense: Long-Term Fairness Analysis of ML-Enabled Systems

**Authors** Yining She, Sumon Biswas, Christian Kästner, Eunsuk Kang

This artifact contains the source code for the ICSE 2025 paper: "FairSense: Long-Term Fairness Analysis of ML-Enabled Systems". It is composed by two parts: the simulation source code and the sensitivity analysis source code. 
The experimental results of three case studies presented in the paper can be reproduced using this artifact.

## Provenance

**Paper PDF:** https://arxiv.org/abs/2501.01665

**Supplementary Material:** Located in `./SupplementaryMaterial/`

![The overview of this work](/overview.jpg)

## Setup
### Environment Preparation
To run the artifact, we need to install Python 3 and R.

**Python:** Tested on Version 3.7 (dependencies listed in `./requirements.txt`)

- Create Conda environment: 
    ```
    conda create -n fairsense python=3.7
    conda activate fairsense
    ``` 
    (If you are new to Conda, refer to [Conda Guide](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html))
- Install dependencies: 
    ```
    pip install -r requirements.txt
    ```
- *Note: The Python library `rpy2` in `requirements.txt` requires R to be installed on your system before installation. However, if you do not intend to run the simulation for the Predictive Policing case study, you can skip installing R and rpy2.*

**R:** Tested on Version 4.2.3 (required for Predictive Policing case study; dependencies listed in `./r_libraries.txt`)
- *Note: If you meet issues when installing certain libraries, please refer to the relevant library documentation, e.g. library [sf](https://github.com/r-spatial/sf?tab=readme-ov-file#installing).*


### Public Datasets
The artifact contains all three case studies in the paper: **Loan lending**, **Opioid risk prediction**, and **Predictive policing**. 

- **Loan lending:** Dataset available from FairMLBook (Barocas, Hardt, and Narayanan, 2018) https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore/data. Save it in `./Simulation/LoanLending/data/`.

- **Opioid risk prediction:** Dataset from MIMIC-IV https://physionet.org/content/mimiciv/2.2/. Save it in `./Simulation/OpioidRisk/mimic-iv-2.2/`. (Restricted-access resource, please follow its official guideline to download)

- **Predictive policing:** Provided in `./Simulation/PoliceAllocation/data/`. We synthezided it following the first few steps in the [previous work](https://github.com/nakpinar/diff-crime-reporting).

## Usage
FairSense is a simulation-based framework that can detect and analyze long-term unfairness in ML-enabled systems under the existence of feedback loop. It contains two parts:
1. **Monte-Carlo simulation** to enumerates evolution trace for each system configuration. 
2. **Sensitivity analysis** on the space of possible configurations to understand the impact of design options and environmental factors on the long-term fairness of the system.

### Part 1: Simulation source code
The directory `./Simulation/` contains the code for simulating the evolution of feedback loop model for 3 case study systems. Each case study has a dedicated subfolder:

- Loan lending: `./Simulation/LoanLending/`
- Opioid risk prediction: `./Simulation/OpioidRisk/`
- Predictive policing: `./Simulation/PoliceAllocation/`

#### Steps to Run

1. **Simulation**
- Run `CASESTUDY_montecarlo.ipynb` in the corresponding subfolder (e.g., `loan_lending_montecarlo.ipynb` for Loan Lending) to collect simulation traces.
- The outputs are complete logs of all simulation traces and are stored in `./Simulation/CASESTUDY/simulation_results/`.

2. **Post-processing**
- Process simulation traces using `CASESTUDY_simulation_data_postprocess.ipynb` to prepare data for  sensitivity analysis.
- The outputs are csv tables containing each simulated configuration's parameters, long-term fairness scores and utility scores. They are stored in `./Simulation/CASESTUDY/simulation_results/` as well.

#### More Information
##### Loan Lending
Three files of the predictive models (`distribution_to_loans_outcomes.py`, `fico.py`, `solve_credit.py`) are adapted from the [artifact](https://github.com/lydiatliu/delayedimpact/) of "Delayed Impact of Fair Machine Learning:.

##### Opioid Risk
`/mimic-preprocess/` contains the code to preprocess the MIMIC-IV dataset. Please download the dataset first, then run `selectPopulationPrescribed.ipynb` to selects the target patients, and finally run `selectPopulationPrescribed.ipynb` to divide the dataset for training and simulation. The preprocessed dataset will be stored in `/mimic_data_after_preprocess/`.

`/mimic-model/` contains the code to train the XGBoost model and MLP model. The subfolder `/mimic-model/models/` contains the trained models.

##### Predictive Policing
`/data/`, `/metadata/`, `/output/`, and `/utils/` are adapted from the [artifact](https://github.com/nakpinar/diff-crime-reporting) of "The effect of differential victim crime reporting on predictive policing systems".


### Part 2: Sensitivity analysis source code
The directory `./SensitivityAnalysis/` contains the programs for sensitivity analysis of simulation traces. Each case study has its subfolder.

The analysis requires postprocessed simulation results. Our experiments' results are already provided in the folders.

To get the analysis result, please run `CASESTUDY_regression.ipynb` in the subfolder for each case study (e.g., `loan_lending_regression.ipynb` for Loan Lending). The script will:
1. Fit regression model, and compute sum of squares and $\eta^2$ for each configuration parameter and interaction term.
2. Identify pareto-optimal configurations and visualize the trade-offs.
3. Compute ranking similarity between the baseline ranks and the ranks found using sampling heuristic.
