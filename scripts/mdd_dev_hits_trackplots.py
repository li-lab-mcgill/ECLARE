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

## 
egr1_links_df = egr1_hits_df[['chrom_gene', 'chromStart_gene', 'chromEnd_gene', 'chrom_peak', 'chromStart_peak', 'chromEnd_peak', 'product']]
egr1_links_df_path = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.links')
egr1_links_df.to_csv(egr1_links_df_path, index=False, sep='\t')

## Create config.ini file
def get_config_ini_content(links_file):
    content = f'''
    [gene–enhancer links]
    file = {links_file}
    file_type = links
    links_type = arcs
    color = black
    height = 3

    [narrow female ExN case]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/ArchR_Subcluster.female_ExN_case.narrowPeak
    height = 4
    max_value = 5
    line_width = 0.1
    title = female ExN case
    show_labels = false
    color = red

    [narrow female ExN control]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/ArchR_Subcluster.female_ExN_control.narrowPeak
    height = 4
    max_value = 5
    line_width = 0.1
    title = female ExN control
    show_labels = false
    color = blue
    overlay_previous = no

    [narrow female End control]
    file = {os.environ['DATAPATH']}/mdd_data/pseudobulked_narrowPeaks/ArchR_Subcluster.female_End_control.narrowPeak
    height = 4
    max_value = 5
    line_width = 0.1
    title = female End control
    show_labels = false
    color = black
    overlay_previous = no

    [spacer]
    height = 1

    [Genes]
    file = {os.environ['DATAPATH']}/gencode.v48.annotation.gtf.gz
    title = Genes
    prefered_name = gene_name
    height = 8
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

config_ini_content = get_config_ini_content(links_file=egr1_links_df_path)
with open(os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'config.ini'), 'w') as f:
    f.write(config_ini_content)

## create trackplot
subprocess.run([
    'pyGenomeTracks', '--tracks', os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'config.ini'),
    '--region', 'chr3:50200000-50340000',
    '-o', os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.png')
    ])

## show trackplot
def open_region_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

image_path = os.path.join(os.environ['OUTPATH'], 'mdd_developmental_analysis', 'egr1_scompreg_hits_grn_all.png')
open_region_image(image_path)
