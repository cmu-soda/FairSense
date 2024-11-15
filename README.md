# FairSense

This repo contains the the source code for the paper: "FairSense: Long-Term Fairness Analysis of ML-Enabled Systems". This paper is accepted by ICSE 2025.

### Environment Preparation
Most programs are in python while in predictive policing case study the python main function calls some R functions.

The R version we used is 4.2.3

The python version we used is 3.7. The package environment requirement is in `./requirements.txt`. You can install it using command `pip install requirements.txt` under the repo root folder.

### Simulation source code
The 3 case studies described in the paper are Loan lending, Opioid risk prediction, and Predictive policing. The simulation code for them are in 

1.  Loan lending: `./LoanLending`
2.  Opioid risk prediction: `./OpioidRisk`
3.  Predictive policing: `./PoliceAllocation`
   
For each case study, there are two jupyter notebooks in its corresponding folder.

To run the simulation and collect traces, run the jupyter notebook `CASESTUDY_montecarlo.ipynb`. 
Take loan lending as an example. After navigating to `./LoanLending` directory, please run the `loan_lending_montecarlo.ipynb`.

After running simulation, run `CASESTUDY_simulation_data_postprocess.ipynb` to postprocess the collected simulation traces. This step is to prepare data for sensitivity analysis. For loan lending case study, please run `loan_lending_simulation_data_postprocess.ipynb`.


### Sensitivity analysis source code
The directory `./SensitivityAnalysis` contains the programs for analyzing long-term fairness. Each case study has one corresponding folder for analysis under `./SensitivityAnalysis` directory.

To get the analysis result, please run the jupyter notebook in the subfolder for each case study. The results shown in the paper can be found in these notebooks.
