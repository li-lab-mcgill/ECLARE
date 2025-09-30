human_dendrogram <- readRDS('dend.json')

sim_matrix <- cophenetic(human_dendrogram)

heatmap(sim_matrix,
        Rowv = NA,
        Colv = NA,
        col = heat.colors(256),
        scale = "none",
        main = "Similarity Matrix Heatmap")

hc <- hclust(sim_matrix, method = "average")
tail(hc$height)

labels_vec <- labels(human_dendrogram)
