# ctdPheno (probabilistic, informed Gaussian Mixture Model)

Functions for using the ctdPheno model: inputs include a feature matrix in pickle format for both the reference data
(to inform the model, e.g. LuCaP features and healthy normal features) and the samples of interest.

df_diff() is first run on the reference data to find features differentially regulated between the two subtypes of interest;
metric_dict() then takes the differential dataframe and produces a dicitonary of prior estimates for use in the model. Finally,
beta_descent() uses the priors from metric_dict to generate predictions based on the sample dataframe, which are output to .tsv.
