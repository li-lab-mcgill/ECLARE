## import modules needed for merge_peaks function
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, oaconvolve
from scipy.signal.windows import gaussian
#from .utils import stats

## import other libraries
import os
from glob import glob
from pybedtools import BedTool
from muon import read_10x_h5, MuData
import h5py
from scipy.sparse import csr_matrix
import anndata

import gc
import socket
hostname = socket.gethostname()

if 'narval' in hostname:
    os.environ['machine'] = 'narval'
    datapath = os.environ['SCRATCH']

elif 'Dylan' in hostname:
    os.environ['machine'] = 'local'
    datapath = "/Users/dmannk/cisformer/workspace/PD_Adams_et_al/"

def save_fragments_file_to_bed(fragments_file):
    # Load the ATAC fragments tsv.gz file
    columns = ['chromosome', 'start', 'end']
    atac_fragments = pd.read_csv(fragments_file, sep='\t', header=None, names=columns)

    # Ensure the dataframe has at least three columns (chromosome, start, end)
    if atac_fragments.shape[1] >= 3:
        # Select the first three columns and save as BED format
        atac_fragments_bed = atac_fragments.iloc[:, [0, 1, 2]]
        
        # Optionally, you can add a name or score column if needed (required for MACS2 narrowPeak)
        atac_fragments_bed['name'] = 'peak'  # A placeholder name for each peak
        atac_fragments_bed['score'] = 0      # A placeholder score

        # Save as .bed file
        output_bed = fragments_file.replace('.tsv.gz', '.bed.gz')
        atac_fragments_bed.to_csv(output_bed, sep='\t', header=False, index=False, compression='gzip')
        return output_bed
    else:
        raise ValueError("The ATAC fragment file doesn't contain enough columns (expecting at least 3)")

def macs2_peak_calling_from_bed(bed_file):

    output_dir = bed_file.replace('.tsv.gz', '_macs2')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define output file prefix
    output_prefix = os.path.join(output_dir, "atac_peaks")

    # Run MACS2 with options suited for ATAC-seq
    os.system(f"macs2 callpeak -t {bed_file} -f BED -n {output_prefix} --outdir {output_dir} --nomodel --shift -100 --extsize 200 -q 0.01")

    return output_dir

def merge_peaks(beds, chrom_sizes, fileFormat="narrowPeak", inferCenter=False, forceUnstranded=False, 
                sigma="auto", perPeakDensity=False, minOverlap=2, output_bedgraph=None):
    """
    From: https://github.com/pdelangen/Muffin/blob/main/muffin/peakMerge.py

    Read peak called files, generate consensuses and the matrix.

    Parameters
    ----------
    beds: list of pandas dataframes
        Each dataframe should be formatted in bed format. Summit is assumed to
        be 7th column (=thickStart) for bed format, and 9th column for
        narrowPeak format.

    chrom_sizes: str or dict
        Path to tab separated (chromosome, length) annotation file. If a dict,
        must be of the form {"ChrName": "chrSize"}.

    fileFormat: "bed" or "narrowPeak"
        Format of the files being read. Bed file format assumes the max signal
        position to be at the 6th column (0-based) in absolute coordinates. The
        narrowPeak format assumes the max signal position to be the 9th column
        with this position being relative to the start position.

    inferCenter: boolean (optional, default False)
        If set to true will use the position halfway between start and end
        positions. Enable this only if the summit position is missing. Can also
        be suitable for broad peaks as the summit position can be unreliable.

    forceUnstranded: Boolean (optional, default False)
        If set to true, assumes all peaks are not strand-specific even if strand
        specific information was found.

    sigma: float or "auto" (optional, default "auto")
        Size of the gaussian filter (lower values = more separation). Only
        effective if perPeakDensity is set to False. "auto" automatically
        selects the filter width at (average peak size)/8. 
    
    perPeakDensity: Boolean (optional, default False)
        Recommended for broad peaks. If set to false will perform a gaussian
        filter along the genome (faster), assuming all peaks have roughly the
        same size. If set to true will create the density curve per peak based
        on each peak individual size. This is much more slower than the filter
        method. May be useful if peaks are expected to have very different
        sizes. Can also be faster when the number of peaks is small.
    
    minOverlap: integer (optional, default 2)
        Minimum number of peaks required at a consensus. 2 Indicates that a peak
        must be replicated at least once.
    """
    alltabs = []
    if type(beds) is not list:
        beds = [beds]
    for tab in beds:
        fmt = fileFormat
        if not fmt in ["bed", "narrowPeak"]:
            raise TypeError(f"Unknown file format : {fmt}")
        # Read bed format
        if fmt == "bed":
            if inferCenter:
                usedCols = [0,1,2,5]
            else:
                usedCols = [0,1,2,5,6]
            tab = tab.iloc[:, usedCols].copy()
            tab[5000] = 1
            tab.columns = np.arange(len(tab.columns))
            tab[0] = tab[0].astype("str", copy=False)
            tab[3].fillna(value=".", inplace=True)
            if inferCenter:
                tab[5] = tab[4]
                tab[4] = ((tab[1]+tab[2])*0.5).astype(int)
            tab[5] = [1]*len(tab)
            alltabs.append(tab)
        elif fmt == "narrowPeak":
            if inferCenter:
                usedCols = [0,1,2,5]
            else:
                usedCols = [0,1,2,5,9]
            tab = tab.iloc[:, usedCols].copy()
            tab[5000] = 1
            tab.columns = np.arange(len(tab.columns))
            tab[0] = tab[0].astype("str", copy=False)
            tab[3].fillna(value=".", inplace=True)
            if inferCenter:
                tab[5] = tab[4]
                tab[4] = ((tab[1]+tab[2])*0.5).astype(int, copy=False)
            else:
                tab[4] = (tab[1] + tab[4]).astype(int, copy=False)
            alltabs.append(tab)
    if type(chrom_sizes) is str:
        chrom_sizes = pd.read_csv(chrom_sizes, sep="\t", header=None, index_col=0).iloc[:,0].to_dict()  
    # Concatenate files
    df = pd.concat(alltabs)
    numElements = len(df)
    avgPeakSize = np.median(df[2] - df[1])
    # Check strandedness
    if forceUnstranded == True:
        df[3] = "."
        strandCount = 1
    else:
        # Check if there is only stranded or non-stranded elements
        strandValues = np.unique(df[3])
        strandCount = len(strandValues)
        if strandCount > 2:
            raise ValueError("More than two strand directions !")
        elif strandCount == 2 and "." in strandValues:
            raise ValueError("Unstranded and stranded values !")
    # Split per strand
    df = dict([(k, x) for k, x in df.groupby(3)])
    ########### Peak separation step ########### 
    # Compute sigma if automatic setting
    if sigma == "auto":   
        sigma = avgPeakSize/4
    else:
        sigma = float(sigma)
    if perPeakDensity:
        sigma = 0.25
    windowSize = int(8*sigma)+1
    sepPerStrand = {}
    sepIdxPerStrand = {}
    
    
    # Iterate for each strand
    consensuses = []
    j = 0
    for s in df.keys():
        # Split peaks per chromosome
        df[s].sort_values(by=[0, 4], inplace=True)
        posPerChr = dict([(k, x.values[:, [1,2,4]].astype(int)) for k, x in df[s].groupby(0)])
        infoPerChr = dict([(k, x.values) for k, x in df[s].groupby(0)])
        # Iterate over all chromosomes
        sepPerStrand[s] = {}
        sepIdxPerStrand[s] = {}
        if output_bedgraph is not None:
            f_bedgraph = open(output_bedgraph+f"{s}.wig", "w")
        for chrName in posPerChr.keys():
            # Place peak on the genomic array
            try:
                currentLen = chrom_sizes[str(chrName)]
            except KeyError:
                print(f"Warning: chromosome {str(chrName)} is not in genome annotation and will be removed")
                continue
            array = np.zeros(currentLen, dtype="float32")
            peakIdx = posPerChr[chrName]
            np.add.at(array, peakIdx[:, 2],1)
            if not perPeakDensity:
                # Smooth peak density
                smoothed = oaconvolve(array, gaussian(windowSize, sigma), "same")
                separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
            else:
                smoothed = np.zeros(currentLen, dtype="float32")
                for i in range(len(peakIdx)):
                    peakSigma = (peakIdx[i, 1] - peakIdx[i, 0])*sigma
                    windowSize = int(8*peakSigma)+1
                    center = (peakIdx[i, 1] + peakIdx[i, 0])*0.5
                    start = max(center - int(windowSize/2), 0)
                    end = min(center + int(windowSize/2) + 1, currentLen)
                    window = gaussian(end-start, peakSigma)
                    smoothed[start:end] += window/window.sum()
            # Split consensuses
            separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
            if output_bedgraph:
                sampling_interval = 5
                f_bedgraph.write(f"fixedStep chrom={chrName} start=1 step={sampling_interval}\n")
                positions = np.arange(0, len(smoothed), sampling_interval)[:-1]
                to_write = np.around(smoothed[positions+int(1+sampling_interval/2)],3)
                to_write = "\n".join(to_write.astype(str))
                f_bedgraph.write(to_write+"\n")
            separators = separators[np.where(np.ediff1d(separators) != 1)[0]+1]    # Removes consecutive separators (because less-equal comparison)
            separators = np.insert(separators, [0,len(separators)], [0, currentLen])        # Add start and end points
            # Assign peaks to separators
            # Not the most optimized but fast enough
            separators[-1]+=1
            smallest_bin = np.digitize(peakIdx[:,0], separators)
            largest_bin = np.digitize(peakIdx[:,1], separators)
            bin_to_segments = dict()
            for seg_id, (smallest, largest) in enumerate(zip(smallest_bin, largest_bin)):
                seg_start, seg_end = peakIdx[seg_id, :2]
                seg_length = seg_end - seg_start
                for bin_id in range(smallest, largest+1):
                    bin_start = separators[bin_id-1]  # np.digitize's output is 1-indexed
                    bin_end = separators[bin_id]
                    bin_length = bin_end - bin_start

                    overlap_start = max(bin_start, seg_start)
                    overlap_end = min(bin_end, seg_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    if overlap_length / bin_length > 0.5 or overlap_length / seg_length > 0.5:
                        if bin_id in bin_to_segments:
                            bin_to_segments[bin_id].append(seg_id)
                        else:
                            bin_to_segments[bin_id] = [seg_id]
            # Format consensus peaks
            for k in bin_to_segments.keys():
                currentConsensus = infoPerChr[chrName][bin_to_segments[k]]
                # Exclude consensuses that are too small
                if len(currentConsensus) < minOverlap:
                    continue
                currentSep = separators[k-1:k+1]
                # Setup consensuses coordinates
                consensusStart = max(np.min(currentConsensus[:,1]), currentSep[0])
                consensusEnd = min(np.max(currentConsensus[:,2]), currentSep[1])
                # Discard abnormally small consensus peaks
                if consensusEnd-consensusStart < avgPeakSize*0.125:
                    continue
                inSep = (currentConsensus[:,4] > currentSep[0]) & (currentConsensus[:,4] < currentSep[1]) 
                if inSep.sum() >= 1:
                    consensusCenter = int(np.mean(currentConsensus[inSep,4]))
                else:
                    consensusCenter = int(consensusStart*0.5+consensusEnd*0.5)
                # Mean value of present features
                meanScore = len(currentConsensus)
                # Add consensus to the genomic locations
                data = [chrName, consensusStart, consensusEnd, j, 
                        meanScore, s, consensusCenter, consensusCenter + 1]
                consensuses.append(data)
                j += 1
        if output_bedgraph:
            f_bedgraph.close()

    return pd.DataFrame(consensuses)

def create_count_matrix(fragments_bed, consensus_peaks_bed, consensus_peaks, atac_barcodes):
    # Intersect fragments with peaks to find overlaps. Ensure to include all consensus peaks, even if they don't have any fragments.
    overlap = consensus_peaks_bed.intersect(fragments_bed, wa=True, wb=True, loj=True).to_dataframe(
        names=['peak_chrom', 'peak_start', 'peak_end', 'peak_index', 'peak_score', 
            'strand', 'peak_summit_start', 'peak_summit_end', 
            'frag_chrom', 'frag_start', 'frag_end', 'barcode', 'frag_score'])

    # Find barcodes that are not in atac_barcodes
    #missing_barcodes = set(overlap['barcode'].unique()) - set(atac_barcodes)

    # Fill NaN values for fragment columns
    #overlap['frag_chrom'].fillna('no_fragment', inplace=True)
    #overlap['barcode'].fillna('no_fragment', inplace=True)
    #overlap['frag_score'].fillna(0, inplace=True)

    n_unique_barcodes_pre = overlap['barcode'].nunique()

    # Retain overlap arising from filtered ATAC barcodes
    atac_barcodes_set = set(atac_barcodes)
    overlap = overlap[overlap['barcode'].isin(atac_barcodes_set)]
    print(f"\t\t - Retained {overlap['barcode'].nunique()} out of {n_unique_barcodes_pre} unique barcodes.", flush=True)

    # Create a unique list of barcodes (cells) and peaks
    #unique_barcodes = overlap['barcode'].unique()
    #unique_peaks = overlap[['peak_chrom', 'peak_start', 'peak_end']].drop_duplicates().reset_index(drop=True)

    # Map barcodes (cells) and peaks to indices
    barcode_to_index = {barcode: idx for idx, barcode in enumerate(atac_barcodes)}
    peak_to_index = {(row.chrom, row.start, row.end): idx for idx, row in consensus_peaks.iterrows()}  # long

    # Extract row indices, column indices for constructing the COO matrix
    overlap_index = pd.MultiIndex.from_frame(overlap[['peak_chrom', 'peak_start', 'peak_end']])  # Create a MultiIndex for efficient lookup
    col_indices = overlap_index.map(peak_to_index).values
    row_indices = overlap['barcode'].map(barcode_to_index).values
    #col_indices = overlap.apply(lambda x: peak_to_index[(x['peak_chrom'], x['peak_start'], x['peak_end'])], axis=1).values

    # Initialize an array to store counts
    count_matrix = np.zeros((len(atac_barcodes), len(consensus_peaks)), dtype=np.int32)

    # Accumulate counts using numpy's add.at
    np.add.at(count_matrix, (row_indices, col_indices), 1)

    # Convert to a sparse matrix (CSR format)
    count_matrix = csr_matrix(count_matrix)

    return count_matrix


if __name__ == "__main__":

    print("- Getting chromsizes", flush=True)

    ## get annotations data: wget https://zenodo.org/records/10708208/files/genome_annot.zip
    path_chromsizes = os.path.join(datapath, "hg38.chrom.sizes.sorted")
    chromSizes = pd.read_csv(path_chromsizes, sep="\t", header=None, index_col=0).iloc[:,0].to_dict()

    print("- Getting paths to fragments files and consensus peaks (if exists)", flush=True)
    fragments_paths = glob(os.path.join(datapath, "PD_all_files", "*fragments.tsv.gz"))
    consensus_peaks_path = os.path.join(datapath, "consensus_peaks.bed")

    if not os.path.exists(consensus_peaks_path):

        print(  "- Processing fragments files to obtain consensus peaks", flush=True)

        narrowPeak_paths = []

        for fragments_path in fragments_paths:  # could be parallelized

            print(f"\t - Processing {fragments_path}", flush=True)

            print("\t\t - Running MACS2 peak calling \n", flush=True)
            output_macs2_path = macs2_peak_calling_from_bed(fragments_path)

            print("\t\t - Getting narrowPeak path \n", flush=True)
            narrowPeak_path = glob(os.path.join(output_macs2_path, "*.narrowPeak"))

            if len(narrowPeak_path) == 0:
                print("\t\t - No peaks found. Skipping.", flush=True)
                continue
            elif len(narrowPeak_path) > 0:
                narrowPeak_path = narrowPeak_path[0]
                narrowPeak_paths.append(narrowPeak_path)

            print("\n", flush=True)

        print("- Creating consensus peaks", flush=True)
        
        beds = [BedTool(narrowPeak_path).to_dataframe() for narrowPeak_path in narrowPeak_paths]
        consensus_peaks = merge_peaks(beds, chromSizes)
        consensus_peaks_bed = BedTool.from_dataframe(consensus_peaks)
        consensus_peaks.to_csv(os.path.join(datapath, "consensus_peaks.bed"), sep="\t", header=False, index=False)

    elif os.path.exists(consensus_peaks_path):
        print("- Loading consensus peaks", flush=True)
        consensus_peaks_bed = BedTool(os.path.join(datapath, "consensus_peaks.bed"))

    print("- Creating new count matrices based on consensus peaks.", flush=True)

    consensus_peaks_bed = BedTool(os.path.join(datapath, "consensus_peaks.bed"))
    consensus_peaks = consensus_peaks_bed.to_dataframe()
    all_atac_consensus = []

    for fragments_path in fragments_paths:  # could be parallelized

        print(f"\t - Processing {fragments_path}", flush=True)
            
        ## Get BedTool object from fragments file
        fragments_bed = BedTool(fragments_path)

        ## Load ATAC barcodes
        h5_path = fragments_path.replace('atac_fragments.tsv.gz', 'filtered_feature_bc_matrix.h5')
        with h5py.File(h5_path, 'r') as f:
            atac_barcodes = f['matrix']['barcodes'][:].astype(str)
        #data = read_10x_h5(h5_path)
        #atac_barcodes = data['atac'].obs_names
        #rna = data['rna']

        print(f"\t\t - Creating count matrix", flush=True)
        #fragments_bed = fragments_bed.filter(lambda frags: frags['name'] in atac_barcodes), doesn't really help
        count_matrix = create_count_matrix(fragments_bed, consensus_peaks_bed, consensus_peaks, atac_barcodes)

        print(f"\t\t - Creating ATAC annadata & mudata from consensus peaks count matrix", flush=True)
        atac_consensus = anndata.AnnData(count_matrix, obs=pd.DataFrame(index=atac_barcodes), var=pd.DataFrame(index=[peak.to_string() for peak in consensus_peaks_bed]))
        atac_consensus = atac_consensus[atac_consensus.obs_names.argsort()]

        all_atac_consensus.append(atac_consensus.copy())

        del count_matrix
        gc.collect()

    print("- Concatenating all ATAC consensus matrices", flush=True)
    all_atac_consensus = anndata.AnnData.concatenate(*all_atac_consensus, batch_key='sample', batch_categories=[os.path.basename(fragments_path).split('_')[0] for fragments_path in fragments_paths])

    print("- Saving ATAC consensus matrix", flush=True)
    all_atac_consensus.write(os.path.join(datapath, "atac_consensus.h5ad"))

    if os.environ['machine'] = 'narval':
        print("- Moving ATAC consensus matrix to project", flush=True)
        os.system(f"mv {os.path.join(datapath, 'atac_consensus.h5ad')} /home/dmannk/projects/def-liyue/dmannk/data/PD_Adams_et_al")

    print("Done!")