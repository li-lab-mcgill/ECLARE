% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generics.R, R/viz.R
\name{GeneCCHeatmap}
\alias{GeneCCHeatmap}
\alias{GeneCCHeatmap.Seurat}
\alias{GeneCCHeatmap.ArchRProject}
\title{GeneCCHeatmap function}
\usage{
GeneCCHeatmap(
  object,
  CCsToPlot = 1:3,
  RNA.assay = "GeneExpressionMatrix",
  colorbar.group = "celltype",
  MOJITOO.reduction = "MOJITOO",
  filter.mito = T,
  filter.ribo = T,
  topN = 10,
  raster = T,
  cols = (ggsci::pal_igv())(51),
  rerun = T,
  ...
)

\method{GeneCCHeatmap}{Seurat}(
  object,
  CCsToPlot = 1:3,
  RNA.assay = "RNA",
  colorbar.group = "celltype",
  MOJITOO.reduction = "MOJITOO",
  filter.mito = T,
  filter.ribo = T,
  topN = 10,
  raster = T,
  cols = (ggsci::pal_igv())(51),
  rerun = T,
  ...
)

\method{GeneCCHeatmap}{ArchRProject}(
  object,
  CCsToPlot = 1:3,
  RNA.assay = "GeneExpressionMatrix",
  colorbar.group = "celltype",
  MOJITOO.reduction = "MOJITOO",
  filter.mito = T,
  filter.ribo = T,
  topN = 10,
  raster = T,
  cols = (ggsci::pal_igv())(51),
  rerun = T,
  ...
)
}
\arguments{
\item{object}{ArchR Object}

\item{reduction.list}{reduction list}

\item{dims.list}{dims list}

\item{reduction.name}{reduction name}
}
\description{
Top positive&negative gene expression of a CC heatmap

Top positive&negative gene expression of a CC heatmap

Top positive&negative gene expression of a CC heatmap
}
\examples{
GeneCCHeatmap()
GeneCCHeatmap()
GeneCCHeatmap()
}
\keyword{GeneCCHeatmap}
