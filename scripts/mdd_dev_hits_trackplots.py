import os
import pandas as pd
import scglue
import anndata

import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

egr1_hits_dir = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.csv')
egr1_hits_df = pd.read_csv(egr1_hits_dir)

tg = egr1_hits_df['TG']

# Get gene positions using gencode annotation
gene_ad = anndata.AnnData(var=pd.DataFrame(index=tg.unique()))
scglue.data.get_gene_annotation(
    gene_ad, 
    gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), 
    gtf_by='gene_name'
)
# Filter out genes without positions
gene_ad = gene_ad[:, gene_ad.var['chrom'].notna()]

# Extract gene positions dataframe
gene_positions = gene_ad.var[['chrom', 'chromStart', 'chromEnd', 'strand', 'name']].copy()
print(f"Found positions for {len(gene_positions)} out of {len(tg.unique())} unique genes")
print(f"\nGene positions:\n{gene_positions.head()}")

## Merge gene positions with egr1_hits_df
egr1_hits_df = egr1_hits_df.merge(gene_positions, left_on='TG', right_index=True, how='left', suffixes=('_peak', '_gene'))
print(f"Merged {len(egr1_hits_df)} rows into egr1_hits_df")
print(f"\nFirst few rows of egr1_hits_df:\n{egr1_hits_df.head()}")

## add -log10(pvalue) to egr1_hits_df
import numpy as np
egr1_hits_df = egr1_hits_df.assign(log10_pvalue=-np.log10(egr1_hits_df['pvalue']))
egr1_hits_df['product'] = egr1_hits_df[['LR', 'log2FoldChange', 'log10_pvalue']].prod(axis=1).abs()

## save hits dataframe to file to be ingested by pyGenomeTracks
egr1_links_df = egr1_hits_df[['chrom_gene', 'chromStart_gene', 'chromEnd_gene', 'chrom_peak', 'chromStart_peak', 'chromEnd_peak', 'product']]
egr1_links_df['chromEnd_gene'] = egr1_links_df['chromStart_gene'] + 1 # make gene position 1 bp, or else have errors with plotting multiple arcs falling inside the gene body
egr1_links_df_path = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.links')
egr1_links_df.to_csv(egr1_links_df_path, index=False, sep='\t', header=False)

def open_region_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

## Create config.ini file
def get_config_ini_content(links_file, celltype, peak_max_value=2):

    # Start with the gene-enhancer links section
    content = f'''
    [gene–enhancer links]
    file = {links_file}
    file_type = links
    links_type = arcs
    color = black
    height = 5
    use_middle = true

    '''
    
    # Check if celltype is a list
    if isinstance(celltype, list):
        # Loop through all celltypes
        for ct in celltype:
            # Add case track
            content += f'''
    [narrow female {ct} case]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/mdd_atac_broad_sub_14814.female_{ct}_Case.narrowPeak
    height = 4
    max_value = {peak_max_value}
    line_width = 0.1
    title = female {ct} case
    show_labels = false
    color = red
    type = peak
    width_adjust = 1.5

    [narrow female {ct} control]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/mdd_atac_broad_sub_14814.female_{ct}_Control.narrowPeak
    height = 4
    max_value = {peak_max_value}
    line_width = 0.1
    title = female {ct} control
    show_labels = false
    color = blue
    overlay_previous = no
    type = peak
    width_adjust = 1.5

    '''
    else:
        # Single celltype (original behavior)
        content += f'''
    [narrow female {celltype} case]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/mdd_atac_broad_sub_14814.female_{celltype}_Case.narrowPeak
    height = 4
    max_value = 1
    line_width = 0.1
    title = female {celltype} case
    show_labels = false
    color = red
    type = peak
    width_adjust = 1.5

    [narrow female {celltype} control]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/mdd_atac_broad_sub_14814.female_{celltype}_Control.narrowPeak
    height = 4
    max_value = 1
    line_width = 0.1
    title = female {celltype} control
    show_labels = false
    color = blue
    overlay_previous = no
    type = peak
    width_adjust = 1.5

    '''
    
    # Add the final sections (spacer, Genes, x-axis)
    content += f'''
    [spacer]
    height = 1

    [Genes]
    file = {os.environ['DATAPATH']}/gencode.v48.annotation.gtf.gz
    title = Genes
    prefered_name = gene_name
    height = 5
    merge_transcripts = True
    labels = True
    max_labels = 200
    all_labels_inside = False
    style = UCSC
    file_type = gtf
    fontsize = 10
    display = stacked

    [spacer]
    height = 1

    [x-axis]
    fontsize = 12

    '''
    return content

config_ini_content = get_config_ini_content(links_file=egr1_links_df_path, celltype=['EN-fetal-late'], peak_max_value=1)
with open(os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'config.ini'), 'w') as f:
    f.write(config_ini_content)

## create trackplot
output_path = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.png')
subprocess.run([
    'pyGenomeTracks', '--tracks', os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'config.ini'),
    '--region', 'chr17:77850000-78500000',
    '--width', '20',
    '-o', output_path,
    '--dpi', '150'
    ])
print(f"Trackplot saved to {output_path}")

## show trackplot
image_path = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.png')
open_region_image(image_path)

def case_control_analysis():
    '''
    Analyze case-control differences in EGR1 hits
    '''
    case_df = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'pseudobulked_narrowPeaks', f'mdd_atac_broad_sub_14814.female_EN_Case.narrowPeak'), sep='\t', header=None)
    control_df = pd.read_csv(os.path.join(os.environ['DATAPATH'], 'mdd_data', 'pseudobulked_narrowPeaks', f'mdd_atac_broad_sub_14814.female_EN_Control.narrowPeak'), sep='\t', header=None)

    case_df.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    control_df.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']

    ## find interval with maximum case-control difference
    case_control_df = pd.merge(case_df, control_df, left_index=True, right_index=True, how='inner', suffixes=('_case', '_control'))
    case_control_df = case_control_df.loc[case_control_df['signalValue_case'].ge(0) & case_control_df['signalValue_control'].ge(0)]
    delta = np.abs(case_control_df['signalValue_case'] - case_control_df['signalValue_control'])
    delta_max = delta.argmax()

    ## find EGR1 hits in the interval
    intervals = case_control_df[['chrom_case', 'chromStart_case', 'chromEnd_case']]
    intervals['chromStart_case'] = intervals['chromStart_case'] + 1
    intervals['interval'] = intervals.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1)
    egr1_hits_df = egr1_hits_df.assign(egr1_hit=egr1_hits_df['enhancer'].isin(intervals['interval']))
    return egr1_hits_df

