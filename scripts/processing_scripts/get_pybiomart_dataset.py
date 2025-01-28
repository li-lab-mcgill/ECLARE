from pybiomart import Dataset as biomart_Dataset

if __name__ == '__main__':

    print('*** Warning ***: This script requires an internet connection to access the Ensembl Biomart database.')

    dataset = biomart_Dataset(name='hsapiens_gene_ensembl', host='http://grch37.ensembl.org')
    results = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype', 'chromosome_name', 'start_position', 'end_position', 'transcription_start_site'])

    savepath = '/home/dmannk/projects/def-liyue/dmannk/data'
    results.to_csv(savepath + '/ensembl_gene_positions.csv', index=False)