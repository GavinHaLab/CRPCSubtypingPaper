# ctdPheno

ctdPheno is a probabilistic approach for classifying tumor phenotypes from circulating tumor DNA (ctDNA). 

Publication/reference: awaiting DOI...

## Description
The key features of ctdPheno is that it accounts for the ctDNA fraction (i.e. tumor fraction estimated by [ichorCNA](https://github.com/GavinHaLab/ichorCNA)) to adjust the expected signals for the tumor phenotype and the non-tumor cfDNA component.

ctdPheno uses cfDNA fragment features extracted from nucleosome coverage profiles computed by [Griffin](https://github.com/GavinHaLab/Griffin). 

For castration-resistant prostate cancer (CRPC), these Griffin features were extracted for adenocarcinoma prostate cancer (ARPC) and neuroendocrine prostate cancer (NEPC). We used patient-derived xenograft plasma ctDNA because it provides pure human ctDNA signals from which ctdPheno can incorporate the tumor fraction to adjust the expected mixtures for tumor and non-tumor cfDNA components.

## Usage and functions
Functions for using the ctdPheno model: inputs include a feature matrix in pickle format for both the reference data
(to inform the model, e.g. LuCaP features and healthy normal features) and the samples of interest.

df_diff() is first run on the reference data to find features differentially regulated between the two subtypes of interest;
metric_dict() then takes the differential dataframe and produces a dicitonary of prior estimates for use in the model. Finally,
beta_descent() uses the priors from metric_dict to generate predictions based on the sample dataframe, which are output to .tsv.


---------------------------------------------------------------  
Copyright (c) 2022 Fred Hutchinson Cancer Research Center  
All rights reserved.  

This program is free software: you can redistribute it and/or modify it under the terms of the BSD-3-Clause-Clear license. No licenses are granted to any patent rights of the Fred Hutchinson Cancer Research Center.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause-Clear license for more details.  

You should have received a copy of the G BSD-3-Clause-Clear license along with this program. If not, see https://spdx.org/licenses/BSD-3-Clause-Clear.html. 
