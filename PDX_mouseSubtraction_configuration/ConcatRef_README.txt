This is one of the pipelines described in the paper:
Jo, S.-Y., Kim, E. & Kim, S. Impact of mouse contamination in genomic profiling of patient-derived models and best practice for robust analysis. Genome Biol. 20, 231 (2019).

1) concatenate_reference.py
This creates the concatRef Fasta file.
python3 concatenate_reference.py --help
usage: concatenate_reference.py [-h] [--humanRef HUMAN_REF_PATH]
                                [--mouseRef MOUSE_REF_PATH]
                                [--concatRef OUT_FILE] [--tag TAG]

optional arguments:
  -h, --help            show this help message and exit
  --humanRef HUMAN_REF_PATH
                        Path to human reference genome
  --mouseRef MOUSE_REF_PATH
                        Path to mouse reference genome
  --concatRef OUT_FILE  Output concated reference file name
  --tag TAG             tag info to rename mouse contig

After creating the new fasta, you also need to create a new BWA index:
e.g.
ml BWA/0.7.17-foss-2016b
bwa index GRCh38_plus_UCSC_mm10.fa

2) mod_pipe_ConcatRef.sh
This is a copy of pipe_ConcatRef.sh that modified to work with our data and with hg38
(https://github.com/Yonsei-TGIL/BestPractice_for_PDMseq/blob/master/pipe_ConcatRef.sh)
