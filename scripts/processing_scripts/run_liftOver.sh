#! /bin/bash

CHAIN_FILE_NAME=$1  # e.g. mm10ToHg38.over.chain.gz
DATASET_NAME=$2  # e.g. 10x mouse brain

# Create bed intervals for liftOver
#python create_bed_intervals_for_liftOver.py ${DATASET_NAME}

# Search for bed file
INPUT_BED=$(find ${DATAPATH}/${DATASET_NAME} -name "*.bed")

# Paths
LIFTOVER_PATH=~/bin/liftOver
CHAIN_FILE=${DATAPATH}/${CHAIN_FILE_NAME}

OUTPUT_BED=${DATAPATH}/${DATASET_NAME}_peak_beds_lifted_${CHAIN_FILE_NAME%%.*}.bed
UNMAPPED_BED=${DATAPATH}/${DATASET_NAME}_peak_beds_unmapped_${CHAIN_FILE_NAME%%.*}.bed

# Run liftOver
$LIFTOVER_PATH -minMatch=0.9 $INPUT_BED $CHAIN_FILE $OUTPUT_BED $UNMAPPED_BED

# Check result
echo "LiftOver complete. Mapped: $OUTPUT_BED, Unmapped: $UNMAPPED_BED"

#***********************************************************************
# WARNING: liftOver was only designed to work between different
#          assemblies of the same organism. It may not do what you want
#          if you are lifting between different organisms. If there has
#          been a rearrangement in one of the species, the size of the
#          region being mapped may change dramatically after mapping.
#***********************************************************************