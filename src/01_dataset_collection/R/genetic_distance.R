# mini script to calculate genetic distance between dataset barcodes

suppressMessages(if (!require("pacman")) install.packages("pacman"))
pacman::p_load("optparse")
pacman::p_load("seqinr")
pacman::p_load("bit64")

option_list = list(
  make_option(c("-a", "--alignment"), type="character", help="alignment file", metavar="character"),
  make_option(c("-o", "--output"), type="character", help="output file", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

aln <- read.alignment(opt$alignment, 'fasta')
dist_matrix <- as.matrix(dist.alignment(aln, matrix="identity"))
write.table(dist_matrix, file=opt$out, sep="\t")

