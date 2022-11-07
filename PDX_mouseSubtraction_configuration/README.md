### Contains config.yaml for use in LuCaP mouse subtraction (https://github.com/GavinHaLab/PDX_mouseSubtraction)

human reference genome used may be obtained at: http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
mouse reference genome used may be obtained at: http://igenomes.illumina.com.s3-website-us-east-1.amazonaws.com/Mus_musculus/NCBI/GRCm38/Mus_musculus_NCBI_GRCm38.tar.gz


## How to run pipeline

First, generate concatRef genome using concatenate_reference.py

  Additional info and details about the ConcatRef is in ConcatRef_README.txt

Second, run subtract_mouse_and_realign.snakefile to unmap, subtract mouse, and realign the results (Separate realignment step no longer required)

    To run this step, there are 4 parameters that should be adjusted in the config:
    	1. input_reference_genome - if your input is a cram file, the reference genome for this file is required. If you have a bam file use 'null'
    	2. ConcatRef_genome - the concatenated mouse+human genome produced by concatenate_reference.py
    	3. human_reference_genome - the version of the human reference genome that will be used for the final realignment (recommended, but not required, to be the same genome version as the human part of the concatRef)
    	4. tag - the suffix that is used to denote mouse chromosomes in the concatRef. Any read mapping to a chromosome containing this tag in the name will be removed.
