pacman::p_load("caret")
pacman::p_load("cowplot")
pacman::p_load("dplyr")
pacman::p_load("ggnewscale")
pacman::p_load("colorspace")
pacman::p_load("ggnewscale")
pacman::p_load("ggplot2")
pacman::p_load("ggpubr")
pacman::p_load("jpeg")
pacman::p_load("magick")
pacman::p_load("Metrics")
pacman::p_load("patchwork")
pacman::p_load("png")
pacman::p_load("RColorBrewer")
pacman::p_load("rstatix")
pacman::p_load("RVAideMemoire")
pacman::p_load("stringr")
pacman::p_load("svglite")
pacman::p_load("tidyr")
pacman::p_load("tidyverse")
pacman::p_load("wesanderson")

base_dir = ''
cm_palette <- brewer.pal(3, "YlGnBu")
cm_palette2 <- brewer.pal(3, "PRGn")
cm_palette3 <- brewer.pal(3, "PuBuGn")
cm_palette4 <- colorRampPalette(c(brewer.pal(3, "Blues")[2], brewer.pal(3, "Reds")[2]))(3)

# load LOOCV results
load_data <- function(job_id) {
  if (job_id == 'Asteraceae' | job_id == 'Poaceae') {
    marker = 'rbcLa'
  } else {
    marker = 'COI-5P'
  }
  
  class_map = read.table(str_interp('${base_dir}/results/${job_id}/class_map.tsv'), sep='\t', header=TRUE)
  records = read.table(str_interp('${base_dir}/data/${job_id}/records/merged/${job_id}_${marker}.tsv'), sep='\t', header=TRUE)
  
  # take the best fusion approach (or the first of them)
  overall_results <- read.table(str_interp('${base_dir}/results/loocv/${job_id}/results_total.tsv'), sep='\t', header=TRUE)
  overall_results <- overall_results[(overall_results$barcode_preprocessing == 'aligned_barcode') & (overall_results$barcode_encoding == 'one_hot_bar') & (overall_results$round_2 != 'BLAST'),]
  
  overall_results <- add_additional_ds_info(overall_results, records, class_map)
  return(overall_results)
}

# add ground truth and prediction species and genus names and add additional counts
add_additional_ds_info <- function(overall_results, records, class_map) {
  # add species
  overall_results <- merge(x = overall_results, y = class_map, by.x = "Y_gt", by.y = 'idx', suffixes = c('', '_gt'), all.y = FALSE)
  overall_results <- merge(x = overall_results, y = class_map, by.x = "Y_pred", by.y = 'idx', suffixes = c('', '_pred'), all.y = FALSE)
  
  # add genera
  overall_results <- overall_results %>% separate(species_name, c('genus_name', NA), remove=FALSE, sep='_')
  overall_results <- overall_results %>% separate(species_name_pred, c('genus_name_pred', NA), remove=FALSE, sep='_')
  
  # collect number of species in genera
  overall_results <- overall_results %>%
    group_by(genus_name) %>%
    mutate(species_in_genus = n_distinct(species_name))
  
  # collect number of samples in genera
  records <- records %>%
    add_count(genus_name, name = 'sample_cnt_genus')
  records_sample_cnt <- records[,c('genus_name', 'sample_cnt_genus')] %>%
    distinct(genus_name, .keep_all = TRUE)
  
  overall_results <- merge(x = overall_results, y = records_sample_cnt, by = 'genus_name', all.y = FALSE, all.x=TRUE)
  return(overall_results)
}

# generate confusion matrices
calculcate_confusion <- function(overall_results) {
  ### convert to factors
  overall_results$genus_name <- factor(overall_results$genus_name, levels = rev(str_sort(unique(overall_results$genus_name))), ordered=TRUE)
  overall_results$genus_name_pred <- factor(overall_results$genus_name_pred, levels = rev(str_sort(unique(overall_results$genus_name))), ordered=TRUE)
  
  # calculcate confusion matrix per modality
  # barcodes
  barcode_only <- overall_results[overall_results$round_2 == 'bar',]
  gcm <- confusionMatrix(data=barcode_only[barcode_only$Y_pred != barcode_only$Y_gt,]$genus_name_pred, reference = barcode_only[barcode_only$Y_pred != barcode_only$Y_gt,]$genus_name)
  gcm_table_bar <- as.data.frame(gcm$table)
  gcm_table_bar <- merge(gcm_table_bar, distinct(barcode_only[,c('genus_name', 'sample_cnt_genus', 'species_in_genus')]), 
                         by.x="Reference", by.y="genus_name",
                         all.x=TRUE)
  gcm_table_bar$rel_freq <- gcm_table_bar$Freq / gcm_table_bar$sample_cnt_genus
  
  # images
  image_only <- overall_results[overall_results$round_2 == 'img',]
  gcm <- confusionMatrix(data=image_only[image_only$Y_pred != image_only$Y_gt,]$genus_name_pred, reference = image_only[image_only$Y_pred != image_only$Y_gt,]$genus_name)
  gcm_table_img <- as.data.frame(gcm$table)
  gcm_table_img <- merge(gcm_table_img, distinct(image_only[,c('genus_name', 'sample_cnt_genus', 'species_in_genus')]), 
                         by.x="Reference", by.y="genus_name",
                         all.x=TRUE)
  gcm_table_img$rel_freq <- gcm_table_img$Freq / gcm_table_img$sample_cnt_genus
  
  # fused
  fused <- overall_results[(overall_results$round_2 != 'img') & (overall_results$round_2 != 'bar'),]
  fused <- fused[fused$round_2 == fused[1, 'round_2'],]
  gcm <- confusionMatrix(data=fused[fused$Y_pred != fused$Y_gt,]$genus_name_pred, reference = fused[fused$Y_pred != fused$Y_gt,]$genus_name)
  gcm_table_fused <- as.data.frame(gcm$table)
  gcm_table_fused <- merge(gcm_table_fused, distinct(fused[,c('genus_name', 'sample_cnt_genus', 'species_in_genus')]), 
                           by.x="Reference", by.y="genus_name",
                           all.x=TRUE)
  gcm_table_fused$rel_freq <- gcm_table_fused$Freq / gcm_table_fused$sample_cnt_genus
  
  # merge
  gcm_table_bar$data <- 'DNA'
  gcm_table_img$data <- 'images'
  gcm_table_fused$data <- 'multimodal'
  
  gcm_table <- rbind(gcm_table_bar, gcm_table_img, gcm_table_fused)
  gcm_table$data <- factor(gcm_table$data, levels = c('DNA', 'images', 'multimodal'), ordered=TRUE)
  
  gcm_table[(gcm_table$Reference == gcm_table$Prediction) & (gcm_table$species_in_genus) == 1, 'rel_freq'] <- NaN
  
  gcm_tables <- calculcate_relative_confusion(gcm_table)
  return(gcm_tables)
}

# calculate relative confusion rates
calculcate_relative_confusion <- function(gcm_table) {
  intra_gcm_table <- gcm_table[gcm_table$Reference == gcm_table$Prediction,]
  intra_gcm_table$comp <- 'intra'
  
  inter_gcm_table <- gcm_table[gcm_table$Reference != gcm_table$Prediction,]
  
  inter_gcm_table <- inter_gcm_table %>%
    group_by(Reference, data) %>%
    mutate(Freq = sum(Freq))
  
  inter_gcm_table$rel_freq <- inter_gcm_table$Freq / inter_gcm_table$sample_cnt_genus
  inter_gcm_table <- inter_gcm_table[!duplicated(inter_gcm_table[c("Reference", "data")]),]
  inter_gcm_table$comp <- 'inter'
  
  combined_gcm_table <- merge(intra_gcm_table[c('Reference', 'Freq', 'sample_cnt_genus', 'species_in_genus', 'data', 'rel_freq')], inter_gcm_table[c('Reference', 'Freq', 'data', 'rel_freq')], by=c('Reference', 'data'), suffixes = c('_intra', '_inter'))
  combined_gcm_table[combined_gcm_table$species_in_genus == 1, 'Freq_intra'] <- 0
  combined_gcm_table$Freq <- combined_gcm_table$Freq_inter + combined_gcm_table$Freq_intra
  combined_gcm_table$rel_freq <- combined_gcm_table$Freq / combined_gcm_table$sample_cnt_genus
  combined_gcm_table$comp <- 'confusion'
  
  # add intra-inter relation with values between -1 and 1; positive values = intrageneric confusion
  bias_gcm_table <- combined_gcm_table
  bias_gcm_table[is.na(bias_gcm_table$rel_freq_intra), 'rel_freq_intra'] = 0.0
  bias_gcm_table$rel_freq <- bias_gcm_table$rel_freq_inter - bias_gcm_table$rel_freq_intra
  bias_gcm_table$comp <- 'bias'
  
  # setup for number of species group
  cnt_gcm_table <- bias_gcm_table
  cnt_gcm_table$comp <- 'species in genus'
  cnt_gcm_table$data <- 'counts'
  cnt_gcm_table <- cnt_gcm_table[!duplicated(cnt_gcm_table$Reference),]
  
  # setup for number of samples in training (LOOCV)
  samples_gcm_table <- cnt_gcm_table
  samples_gcm_table$sample_cnt_genus <- samples_gcm_table$sample_cnt_genus - 1
  samples_gcm_table$comp <- 'train samples'
  samples_gcm_table$data <- 'counts'
  
  # merge all sub datasets
  inter_intra_gcm_table <- rbind(combined_gcm_table, bias_gcm_table, cnt_gcm_table, samples_gcm_table)
  inter_intra_gcm_table[inter_intra_gcm_table$comp != 'confusion', 'rel_freq'] <- NaN
  
  # removes tiles without any confusion to make more room in plots for relevant information
  inter_intra_gcm_table <- inter_intra_gcm_table[inter_intra_gcm_table$Reference %in% intra_gcm_table[!is.na(intra_gcm_table$rel_freq), 'Reference'],]
  bias_gcm_table <- bias_gcm_table[bias_gcm_table$Reference %in%
                                     intra_gcm_table[!is.na(intra_gcm_table$rel_freq), 'Reference'],]
  cnt_gcm_table <- cnt_gcm_table[cnt_gcm_table$Reference %in%
                                   intra_gcm_table[!is.na(intra_gcm_table$rel_freq), 'Reference'],]
  samples_gcm_table <- samples_gcm_table[samples_gcm_table$Reference %in%
                                           intra_gcm_table[!is.na(intra_gcm_table$rel_freq), 'Reference'],]
  
  # change order of groups on x-axis
  inter_intra_gcm_table$comp <- factor(inter_intra_gcm_table$comp,
                                       levels = c("confusion", "bias", "species in genus", "train samples"),
                                       ordered = TRUE)
  inter_intra_gcm_table$data <- factor(inter_intra_gcm_table$data,
                                       levels = c("DNA", "images", "multimodal", "counts"),
                                       ordered = TRUE)
  bias_gcm_table$data <- factor(bias_gcm_table$data,
                                levels = c("DNA", "images", "multimodal", "counts"),
                                ordered = TRUE)
  cnt_gcm_table$data <- factor(cnt_gcm_table$data,
                               levels = c("DNA", "images", "multimodal", "counts"),
                               ordered = TRUE)
  samples_gcm_table$data <- factor(samples_gcm_table$data,
                                   levels = c("DNA", "images", "multimodal", "counts"),
                                   ordered = TRUE)
  
  gcm_tables = list(
    "inter_intra" = inter_intra_gcm_table,
    "bias" = bias_gcm_table,
    "count" = cnt_gcm_table,
    "samples" = samples_gcm_table
  )
  return(gcm_tables)
}

# plot confusion
create_plot <- function(inter_intra_gcm_table, bias_gcm_table, cnt_gcm_table, samples_gcm_table) {
  species_in_genus_range <- max(cnt_gcm_table$species_in_genus) - min(cnt_gcm_table$species_in_genus)
  species_in_genus_breaks <- as.integer(seq(from = min(cnt_gcm_table$species_in_genus), 
                                            to = max(cnt_gcm_table$species_in_genus), 
                                            length.out = 5))
  training_samples_range <- max(samples_gcm_table$sample_cnt_genus) - min(samples_gcm_table$sample_cnt_genus)
  training_samples_breaks <- as.integer(seq(from = min(samples_gcm_table$sample_cnt_genus), 
                                            to = max(samples_gcm_table$sample_cnt_genus), 
                                            length.out = 5))
  
  actual_plot <- ggplot() +
    geom_tile(data = inter_intra_gcm_table, aes(x = comp, y = Reference, fill = rel_freq), colour="white", linewidth=0.1) +
    scale_fill_gradientn(name='Confusion (%)', colors = cm_palette4, limits = 
                           c(0, 1), na.value = "transparent", guide='none') +
    facet_wrap(data~., ncol=4, scales='free_x') +
    labs(x="", y="", title=job_id) +
    theme_minimal() +
    theme(axis.text.x = element_text(size=22, angle = 90, hjust = 1, vjust = 0.5),
          axis.text.y = element_text(size=22),
          legend.position = 'bottom',
          legend.box.margin = margin(0, 0, 0, 0),
          legend.title.align=.5,
          legend.title.position = "left",
          legend.direction = "horizontal",
          legend.box.just = "right",
          legend.box="vertical",
          legend.title=element_text(size=22, face='bold'),
          legend.text=element_text(size=16, face="bold"),
          legend.text.position = "top",
          legend.key.height=grid::unit(60, "pt"),
          legend.key.width=grid::unit(100, "pt"),
          strip.text.x = element_text(size=28, face='bold'),
          plot.title=element_text(hjust=0, size=48, face="bold"),
          plot.margin = unit(c(1,1,1,1), "cm")) +
    scale_x_discrete(labels=c('confusion' = 'confusion', 'bias' = 'bias', 'species in genus' = 'species in\ngenus',
                              'train samples' = 'no. of\ntrain samples')) +
    new_scale_fill() +
    geom_tile(data=bias_gcm_table, aes(x=comp, y=Reference, fill=rel_freq), colour="white", linewidth=0.1) +
    scale_fill_gradientn(name = 'Confusion bias', colors = cm_palette2, 
                         limits = c(-1, 1), na.value = "transparent", guide='none') +
    new_scale_fill() +
    geom_tile(data=cnt_gcm_table, aes(x=comp, y=Reference, fill=species_in_genus), colour="white", linewidth=0.1) +
    scale_fill_gradientn(name='Species in genus', colors = cm_palette3, 
                         limits = c(min(cnt_gcm_table$species_in_genus), max(cnt_gcm_table$species_in_genus)), na.value = "transparent",
                         breaks=species_in_genus_breaks) +
    new_scale_fill() +
    geom_tile(data=samples_gcm_table, aes(x=comp, y=Reference, fill=sample_cnt_genus), colour="white", linewidth=0.1) +
    scale_fill_gradientn(name='Train samples', colors = cm_palette3, 
                         limits = c(min(samples_gcm_table$sample_cnt_genus), max(samples_gcm_table$sample_cnt_genus)), na.value = "transparent",
                         breaks=training_samples_breaks)
  
  # Get legends
  # combi
  p <- ggplot() +
    geom_tile(data = inter_intra_gcm_table, aes(x = comp, y = Reference, fill = rel_freq),
              colour="white", linewidth=0.1) +
    scale_fill_gradientn(name='Confusion (%)', colors = cm_palette4,
                         limits = c(0, 1),
                         na.value = "transparent")
  legend1 <- get_legend(
    # create some space to the left of the legend
    p + theme(legend.box.margin = margin(330, 0, 0, 0),
              legend.spacing.y = unit(1, 'cm'),
              legend.title.align=0,
              legend.box.just = "left",
              legend.title=element_text(size=22, face='bold', margin=margin(b=1, unit = "cm")),
              legend.text=element_text(size=16, face="bold"),
              legend.key.height=grid::unit(100, "pt"),
              legend.key.width=grid::unit(70, "pt"))
  )
  # bias
  p <- ggplot() +
    geom_tile(data = inter_intra_gcm_table, aes(x = comp, y = Reference, fill = rel_freq),
              colour="white", linewidth=0.1) +
    scale_fill_gradientn(name='Confusion bias', colors = cm_palette2,
                         limits = c(-1, 1),
                         na.value = "transparent",
                         breaks=c(-1, 0, 1),
                         labels=c('intra\ngeneric', 'no bias', 'inter\ngeneric'))
  legend2 <- get_legend(
    p + theme(legend.box.margin = margin(330, 0, 0, 0),
              legend.spacing.y = unit(1, 'cm'),
              legend.title.align=0,
              legend.box.just = "left",
              legend.title=element_text(size=22, face='bold', margin=margin(b=1, unit = "cm")),
              legend.text=element_text(size=16, face="bold"),
              legend.key.height=grid::unit(100, "pt"),
              legend.key.width=grid::unit(70, "pt"))
  )
  
  actual_plot <- plot_grid(actual_plot, nrow = 1, ncol = 1) + theme(panel.border = element_rect(colour = "black", fill=NA, size=1))
  
  plot_w_legends = list(
    "plot" = actual_plot,
    "legend1" = legend1,
    "legend2" = legend2
  )
  
  return(plot_w_legends)
}

# main part; iterates over datasets
jobs <- c('Asteraceae', 'Poaceae', 'Coccinellidae', 'Lycaenidae')
gcm_tables_l = vector("list", length = 4)
for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  
  overall_results <- load_data(job_id)
  gcm_tables_l[[i]] <- calculcate_confusion(overall_results)
}

# iteration for plot creation
plots = vector("list", length = 4)
legends = list()

for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  gcm_table <- gcm_tables_l[[i]]
  plot_w_legends <- create_plot(gcm_table$inter_intra, gcm_table$bias, gcm_table$count, gcm_table$samples)
  plots[[i]] <- plot_w_legends$plot
  legends$legend1 <- plot_w_legends$legend1
  legends$legend2 <- plot_w_legends$legend2
}

# merge plots
main_p <- ggarrange(
  plots[[1]], plots[[2]], plots[[3]], plots[[4]],
  nrow = 2,
  ncol = 2
)

legends_grid <- plot_spacer() + legends$legend1 + legends$legend2 + plot_spacer() +
  plot_layout(ncol = 1, heights = c(.8, 1, 1, 1.2))

p <- main_p + legends_grid +
  plot_layout(ncol = 2, width = c(12, 1.2))
