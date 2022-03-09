#!/usr/bin/env python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 08/02/2021

# Generate ROI-level fragmentation based features from bam files

import os
import sys
import pysam
import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import Pool
from functools import partial


def frag_ratio(frag_lengths):  # compute the ratio of short to long fragments
    short_frags = len([x for x in frag_lengths if x <= 120])
    long_frags = len([x for x in frag_lengths if 140 <= x <= 250])
    if short_frags > 0 and long_frags > 0:
        ratio = short_frags / long_frags
        return ratio
    else:
        return np.nan


def normalized_entropy(frag_lengths, bins):
    histogram = np.histogram(frag_lengths, bins=bins)[0]
    pdf = histogram / sum(histogram)
    pdf = [p for p in pdf if p != 0.0]
    moments = [p * np.log(p) / np.log(len(pdf)) for p in pdf]
    return - sum(moments)


def frag_info(bed_region, params):
    bam_path, sample, out_direct, frag_range, to_plot, map_q = params
    bam = pysam.AlignmentFile(bam_path, 'rb')
    bed_tokens = bed_region.strip().split('\t')
    start_pos = int(bed_tokens[1])
    stop_pos = int(bed_tokens[2])
    site = str(bed_tokens[3])
    fragment_lengths = []
    segment_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
    for read in segment_reads:
        fragment_length = read.template_length
        if frag_range[0] <= np.abs(fragment_length) <= frag_range[1] and read.is_paired and read. \
                mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
            fragment_lengths.append(abs(read.template_length))
    if len(fragment_lengths) > 10:
        ratio = frag_ratio(fragment_lengths)
        bin_seq = list(range(frag_range[0], frag_range[1] + 1, 10))
        ent = normalized_entropy(fragment_lengths, bin_seq)
        std = np.std(fragment_lengths)
        mean = np.mean(fragment_lengths)
        cv = std / mean
        mad = np.median(np.absolute(fragment_lengths - np.median(fragment_lengths)))
        if site in to_plot:  # only plot genes of interest
            y, x, _ = plt.hist(fragment_lengths, bins=bin_seq)
            plt.xlabel('fragment length')
            plt.ylabel('counts')
            plt.title('histogram of fragment lengths for ' + sample + ' ' + bed_tokens[3])
            plt.savefig(out_direct + '/' + bed_tokens[3] + '_' + sample + '.pdf')
            plt.close()
        return site, ratio, ent, cv, mean, mad
    else:
        return site, np.nan, np.nan, np.nan, np.nan, np.nan


def main():
    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### GenerateFragmentFeatures.py ### Generate fragment features in'
                                                 'regions of interest for a bam file - see code for features.')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='bam file', required=True)
    parser.add_argument('-a', '--annotation', help='bed file with regions of interest', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-p', '--plot_list', help='list of genes/regions to generate plots for', required=True)
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range to use (bp)', nargs=2, type=int, default=(15, 500))
    parser.add_argument('-c', '--cpus', help='cpu available for parallel processing', type=int, required=True)
    args = parser.parse_args()

    print('Loading input files . . .')

    sample_name = args.sample_name
    bam_path = args.input
    bed_path = args.annotation
    results_dir = args.results_dir
    plot_list = args.plot_list
    map_q = args.map_quality
    size_range = args.size_range
    cpus = args.cpus

    print('\narguments provided:')
    print('\tsample_name = "' + sample_name + '"')
    print('\tbam_path = "' + bam_path + '"')
    print('\tsites_bed = "' + bed_path + '"')
    print('\tplot_list = "' + plot_list + '"')
    print('\tresults_dir = "' + results_dir + '"')
    print('\tsize_range =', size_range)
    print('\tmap_q =', map_q)
    print('\tCPUs =', cpus)
    print('\n')
    sys.stdout.flush()

    to_plot = pd.read_table(plot_list, header=None)[0].tolist()
    sites = [region for region in open(bed_path, 'r')]
    random.shuffle(sites)
    params = [bam_path, sample_name, results_dir, size_range, to_plot, map_q]

    print('Running frag_info on ' + str(len(sites)) + ' regions . . .')

    with Pool(cpus) as pool:
        results = list(pool.imap_unordered(partial(frag_info, params=params), sites, len(sites) // cpus))

    print('Merging results . . .')

    fm = {sample_name: {'Sample': sample_name}}
    for result in results:
        fm[sample_name][result[0] + '_short-long-ratio'] = result[1]
        fm[sample_name][result[0] + '_shannon-entropy'] = result[2]
        fm[sample_name][result[0] + '_frag-cv'] = result[3]
        fm[sample_name][result[0] + '_frag-mean'] = result[4]
        fm[sample_name][result[0] + '_frag-MAD'] = result[5]
    df = pd.DataFrame(fm).transpose()

    out_file = results_dir + '/' + sample_name + '_FragmentFM.tsv'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    df.to_csv(out_file, sep='\t')

    print('Finished.')


if __name__ == "__main__":
    main()
