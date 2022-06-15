# Generates fragment size based features for BAM sample files in regions of interest

Input consists of a sample name, BAM file, BED-style annotation with regions of interest, output directory, list of regions to generate
plots for, minimum mapping quality (defauly 20), fragment sizes to consider (default 15-500), and cpus for parallel processing.

Outputs a feature matrix with the short-long ratio, Shannon entropy, coefficient of variation (CV), mean length, and median absolute
deviation (MAD) for the fragment size distirbution in each region.
 
