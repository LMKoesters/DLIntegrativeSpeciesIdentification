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

base_dir = '/Users/lkosters/PycharmProjects/2023_barcodejpg_paper/analysis'
pal <- c("#ABDDDE", "#F8AFA8", "#FD6467")

# checks if there are any duplicate sequences
duplicate_status <- function(species_name, species_name_pred, genus_name, distance, duplicate) {
  if (nrow(genetic_distances_zeros$intrageneric[genetic_distances_zeros$intrageneric$species_name == species_name,]) > 0) {
    return('intrageneric duplicate')
  } else if (nrow(genetic_distances_zeros$intergeneric[genetic_distances_zeros$intergeneric$species_name == species_name,]) > 0) {
    return('intergeneric duplicate')
  }
  
  return('no duplicate')
}

# add ground truth and prediction species and genus names
add_information <- function(preds, class_map, preds_basis) {
  # add species
  preds <- merge(x = preds, y = class_map, by.x = "Y_gt", by.y = 'idx', suffixes = c('', '_gt'))
  preds <- merge(x = preds, y = class_map, by.x = "Y_pred", by.y = 'idx', suffixes = c('', '_pred'))
  
  # add genera
  preds <- preds %>% separate(species_name, c('genus_name', NA), remove=FALSE, sep='_')
  preds <- preds %>% separate(species_name_pred, c('genus_name_pred', NA), remove=FALSE, sep='_')
  
  preds <- preds %>%
    add_count(genus_name, name = 'sample_cnt_genus')
  
  preds$data <- preds_basis
  
  return(preds)
}

# calculate the relative confusions
calculate_rel_confusions <- function(gcm_table) {
  gcm_table <- gcm_table %>%
    group_by(species_name, species_name_pred, data) %>%
    mutate(Freq = n()) %>%
    ungroup() %>%
    group_by(genus_name, genus_name_pred, data) %>%
    mutate(genus_Freq = sum(Freq))
  
  gcm_table$rel_genus_freq <- gcm_table$genus_Freq / gcm_table$sample_cnt_genus
  gcm_table$rel_species_freq <- gcm_table$Freq / 4
  return(gcm_table)
}

# calculates confusion matrix
build_gcm <- function(preds, records, class_map) {
  preds$Y_gt <- factor(preds$Y_gt, levels = rev(str_sort(unique(preds$Y_gt))), ordered=TRUE)
  preds$Y_pred <- factor(preds$Y_pred, levels = rev(str_sort(unique(preds$Y_gt))), ordered=TRUE)
  
  gcm <- confusionMatrix(data=preds$Y_pred, reference = preds$Y_gt)
  gcm_table <- as.data.frame(gcm$table)
  
  gcm_table <- merge(x = gcm_table, y = class_map, by.x = "Reference", by.y = 'idx')
  gcm_table <- merge(x = gcm_table, y = class_map, by.x = "Prediction", by.y = 'idx', suffixes = c('', '_pred'))
  gcm_table <- merge(gcm_table, records[,c('species_name', 'genus_name')], by.x='species_name', by.y =
                       'species_name')
  gcm_table <- merge(gcm_table, records[,c('species_name', 'genus_name')], by.x='species_name_pred', by.y =
                       'species_name', suffixes = c("", "_pred"))
  gcm_table <- merge(gcm_table, distinct(preds[,c('genus_name', 'sample_cnt_genus')]),
                     by.x="genus_name", by.y="genus_name",
                     all.x=TRUE)
  return(gcm_table)
}

# adds information on which of the modalities identified correctly & helped the other one out
get_combi <- function(ground_truth, data) {
  data_splt <- strsplit(data, ", ")[[1]]
  img_idx = which(data_splt == 'images')
  bar_idx = which(data_splt == 'barcodes')
  fused_idx = which(data_splt == 'fused')
  
  ground_truth_splt = strsplit(ground_truth, ", ")[[1]]
  img_gt <- as.logical(ground_truth_splt[img_idx])
  bar_gt <- as.logical(ground_truth_splt[bar_idx])
  fused_gt <- as.logical(ground_truth_splt[fused_idx])
  
  if (!img_gt & bar_gt & fused_gt) {
    return('barcode helps')
  } else if (img_gt & !bar_gt & fused_gt) {
    return('image helps')
  } else if (!img_gt & !bar_gt & fused_gt) {
    return('more than their parts')
  } else if (!img_gt & !bar_gt & !fused_gt) {
    return('just wrong')
  } else if (img_gt & !bar_gt & !fused_gt) {
    return("image correct")
  } else if (!img_gt & bar_gt & !fused_gt) {
    return("DNA correct")
  } else {
    return("whatever")
  }
}

# loads dataset
gather_ds <- function(job_id, marker) {
  records = read.table(str_interp('${base_dir}/data/${job_id}/records/merged/${job_id}_${marker}.tsv'), 
                       sep='\t', header=TRUE) %>%
    rownames_to_column(var = 'test_idx')
  
  genetic_distances <- gather_genetic_distances(job_id, marker)
  class_map = read.table(str_interp('${base_dir}/results/${job_id}/class_map.tsv'), sep='\t', header=TRUE)
  
  preds <- gather_preds(job_id, records)
  preds$bar <- add_information(preds$bar, class_map, 'barcodes')
  preds$img <- add_information(preds$img, class_map, 'images')
  preds$fused <- add_information(preds$fused, class_map, 'fused')
  
  gcm_table <- rbind(preds$bar, preds$img, preds$fused)
  gcm_table <- calculate_rel_confusions(gcm_table)
  genetic_dist_gcm <- merge_data(gcm_table, genetic_distances)
  
  return(genetic_dist_gcm)
}

# loads genetic distances from pre-calculated file and checks for duplicate sequences
gather_genetic_distances <- function(job_id, marker) {
  if (file.exists(str_interp('${base_dir}/data/${job_id}/stats/${marker}_distance_matrix_R_input.tsv'))) {
    genetic_distances <- read.table(str_interp('${base_dir}/data/${job_id}/stats/${marker}_distance_matrix_R_input.tsv'), sep='\t', header=TRUE)
  } else {
    genetic_distances <- read.table(str_interp('${base_dir}/data/${job_id}/stats/${marker}_distance_matrix.tsv'),
                                    sep='\t', header=TRUE)
    records <- read.table(str_interp('${base_dir}/data/${job_id}/records/merged/${job_id}_${marker}.tsv'), 
                          sep='\t', header=TRUE)
    records$species_name <- gsub(" ", "_", as.character(records$species_name))
    
    # wide to long genetic distances
    genetic_distances$target_id <- rownames(genetic_distances)
    genetic_distances <- genetic_distances %>% pivot_longer(-target_id, names_to = "query_id", values_to = "distance")
    
    genetic_distances <- merge(genetic_distances, records[,c('record_id', 'species_name', 'genus_name', 'dataset')],
                               by.x='target_id', by.y = 'record_id', 
                               all.x=TRUE, all.y=FALSE)
    genetic_distances <- merge(genetic_distances, records[,c('record_id', 'species_name', 'genus_name', 'dataset')],
                               by.x='query_id', by.y = 'record_id', suffixes = c("", "_query"), 
                               all.x=TRUE, all.y=FALSE)
    
    genetic_distances <- genetic_distances[genetic_distances$query_id != genetic_distances$target_id,]
    
    genetic_distances <- genetic_distances %>%
      group_by(target_id, species_name_query) %>%
      mutate(species_dist = mean(distance)) %>%
      ungroup() %>%
      group_by(target_id, genus_name_query) %>%
      mutate(genus_dist = mean(distance)) %>%
      ungroup()
    
    genetic_distances_diff_spec <- genetic_distances[genetic_distances$species_name != genetic_distances$species_name_query,] %>%
      group_by(species_name) %>%
      mutate(
        duplicate = case_when(
          min(distance) == 0 ~ 'intergeneric',
          TRUE ~ NA
        )) %>%
      ungroup() %>%
      group_by(species_name, genus_name, genus_name_query) %>%
      mutate(
        duplicate = case_when(
          (min(distance) == 0) & (genus_name == genus_name_query) ~ 'intrageneric',
          TRUE ~ duplicate
        )) %>%
      ungroup() %>%
      group_by(species_name, species_name_query) %>%
      mutate(
        duplicate = case_when(
          min(distance) == 0 ~ 'combi',
          is.na(duplicate) ~ 'no duplicate',
          TRUE ~ duplicate
        )
      )
    
    genetic_distances <- genetic_distances[genetic_distances$species_name == genetic_distances$species_name_query,] %>%
      group_by(species_name, species_name_query) %>%
      mutate(
        duplicate = case_when(
          min(distance) == 0 ~ 'intraspecific',
          TRUE ~ 'no duplicate'
        ))
    
    genetic_distances <- rbind(genetic_distances, genetic_distances_diff_spec)
    genetic_distances <- genetic_distances[!duplicated(genetic_distances[c('species_name', 'species_name_query')]),]
    genetic_distances <- subset(genetic_distances, select=-c(distance))
    
    write.table(genetic_distances, str_interp('${base_dir}/data/${job_id}/stats/${marker}_distance_matrix_R_input.tsv'), sep='\t', row.names=FALSE,
                quote = FALSE)
  }
  return(genetic_distances)
}

# loads classification results with predictions
gather_preds <- function(job_id, records) {
  bar_prep <- 'aligned_barcode'
  bar_enc <- 'one_hot_bar'
  
  preds = list(
    "bar" = NULL,
    "img" = NULL,
    "fused" = NULL
  )
  
  preds_loocv <- read.table(str_interp('${base_dir}/results/loocv/${job_id}/results_total.tsv'), sep='\t', header=TRUE)
  preds_loocv$species_name <- gsub(" ", "_", as.character(preds_loocv$species_name))
  preds_loocv <- preds_loocv[(preds_loocv$round_2 != 'BLAST') & (preds_loocv$barcode_encoding == bar_enc) & (preds_loocv$barcode_preprocessing == bar_prep),]
  
  # determine best fused according to loocv
  preds_loocv <- preds_loocv %>%
    group_by(round_2) %>%
    mutate(overall_acc = mean(val_acc)) %>%
    ungroup()
  
  preds_loocv <- merge(preds_loocv, records[,c('test_idx', 'record_id')], by='test_idx', all.x=TRUE, all.y=FALSE)
  
  preds_fused <- preds_loocv[!preds_loocv$round_2 %in% c('bar', 'img'),]
  best_fused <- preds_fused[preds_fused$overall_acc == max(preds_fused$overall_acc), 'round_2']
  
  # subset for best epoch
  preds$bar <- preds_loocv[preds_loocv$round_2 == 'bar',]
  preds$img <- preds_loocv[preds_loocv$round_2 == 'img',]
  preds$fused <- preds_loocv[preds_loocv$round_2 == best_fused[[1]],]
  
  return(preds)
}

# merges confusion datafreamw tih genetic distances and adds information on which of the modalities classified correctly
merge_data <- function(gcm_table, genetic_distances) {
  genetic_dist_gcm <- merge(x=gcm_table,
                            y=genetic_distances[, c('species_name', 'species_name_query',
                                                    'species_dist', 'genus_dist')],
                            by.x=c("species_name", "species_name_pred"), 
                            by.y=c("species_name", "species_name_query"), all.x=TRUE, all.y=FALSE)
  
  genetic_dist_gcm <- genetic_dist_gcm %>%
    group_by(species_name, species_name_pred) %>%
    mutate(total_freq = sum(Freq)) %>%
    ungroup() %>%
    group_by(species_name) %>%
    mutate(mean_reference_dist = mean(species_dist))
  
  sorted_dist <- genetic_dist_gcm[!duplicated(genetic_dist_gcm$species_name), 
                                  c("species_name", "mean_reference_dist")]
  sorted_dist <- sorted_dist %>%
    arrange(mean_reference_dist)
  sorted_dist$ref_idx <- 1:nrow(sorted_dist)
  
  genetic_dist_gcm <- merge(genetic_dist_gcm, sorted_dist[,c('species_name', 'ref_idx')], by.x='species_name', by.y='species_name', all.x=TRUE, all.y=FALSE)
  
  genetic_dist_gcm <- genetic_dist_gcm %>%
    mutate(level = case_when(
      species_name == species_name_pred ~ 'ground truth',
      genus_name == genus_name_pred ~ 'intrageneric',
      TRUE ~ 'intergeneric'
    ))
  
  genetic_dist_gcm <- merge(genetic_dist_gcm, 
                            genetic_distances[,c('species_name', 'species_name_query', 'duplicate')], 
                            by.x=c('species_name', 'species_name_pred'), 
                            by.y=c('species_name', 'species_name_query'), all.x=TRUE, all.y=FALSE)
  
  # who's helping out
  genetic_dist_gcm <- genetic_dist_gcm %>%
    mutate(ground_truth = level == 'ground truth') %>%
    arrange(data) %>%
    group_by(record_id) %>%
    mutate(Combination = toString(ground_truth), Factors = toString(data)) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(who_helps = get_combi(Combination, Factors))
  
  genetic_dist_gcm <- genetic_dist_gcm[genetic_dist_gcm$who_helps != 'whatever',]
  return(genetic_dist_gcm)
}

# main part that iterates over datasets and gathers data
datalist = vector("list", length = 4)

jobs <- c('Asteraceae', 'Poaceae', 'Coccinellidae', 'Lycaenidae')
for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  print(job_id)
  
  if (job_id == 'Asteraceae' | job_id == 'Poaceae') {
    marker = 'rbcLa'
  } else {
    marker = 'COI-5P'
  }
  
  genetic_dist_gcm <- gather_ds(job_id, marker)
  datalist[[i]] <- genetic_dist_gcm
}

# plot function
plot_conf <- function(genetic_dist_gcm_full, genetic_dist_gcm, job_id, best_fusion) {
  if (nrow(genetic_dist_gcm) == 0) {
    p <- ggplot(genetic_dist_gcm, aes(x=data, y=species_dist_rounded)) +
      geom_blank() +
      labs(x="", y="", title=str_interp(job_id)) +
      theme(
        legend.position='none',
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(linewidth = 0.5, linetype = 'solid',
                                          colour = "lightgrey"),
        panel.grid.minor.y = element_line(linewidth = 0.2, linetype = 'solid',
                                          colour = "lightgrey"),
        panel.background = element_blank(),
        plot.margin=margin(1, 0.5, 0.5, 0.5, 'cm'),
        plot.title=element_text(hjust=0.5, size=36, face="bold", margin=margin(0,0,50,0)),
        panel.border = element_rect(colour = "grey", fill=NA, linewidth=.5),
        axis.text = element_text(size=32),
        axis.text.x = element_text(margin=margin(20,0,0,0)),
        strip.background = element_blank())
    return(p)
  }
  
  max_y <- max(genetic_dist_gcm$species_dist_rounded) + 0.01
  
  p <- ggplot(genetic_dist_gcm, aes(x=data, y=species_dist_rounded)) +
    geom_smooth(data=genetic_dist_gcm_full, aes(group=who_helps), se=FALSE, color="#6C8645") +
    geom_point(aes(fill=level, shape=duplicate, size=data_points), color='black', stroke = .5, position=position_dodge(.3)) +
    scale_shape_manual(values=c('intraspecific' = 23, 'combi' = 21, 'intrageneric' = 24,
                                'intergeneric' = 22, 'no duplicate' = 25)) +
    scale_fill_manual(values=c('ground truth'=pal[[1]], 'intrageneric'=pal[[2]], 'intergeneric'=pal[[3]])) +
    scale_size_continuous(range = c(5,11)) +
    facet_wrap(~who_helps, labeller = labeller(who_helps=c("barcode helps" = "barcode saves\nidentification",
                                                           "image helps" = "image saves\nidentification",
                                                           "more than their parts" = "teamwork",
                                                           "image correct" = "image identifies correctly",
                                                           "DNA correct" = "DNA identifies correctly",
                                                           "just wrong" = "altogether wrong"))) +
    labs(x="", y="", title=str_interp(job_id)) +
    theme(
      legend.position='none',
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(linewidth = 0.5, linetype = 'solid',
                                        colour = "lightgrey"),
      panel.grid.minor.y = element_line(linewidth = 0.2, linetype = 'solid',
                                        colour = "lightgrey"),
      panel.background = element_blank(),
      plot.margin=margin(1, 0.5, 0.5, 0.5, 'cm'),
      plot.title=element_text(hjust=0.5, size=36, face="bold", margin=margin(0,0,50,0)),
      panel.border = element_rect(colour = "grey", fill=NA, linewidth=.5),
      axis.text = element_text(size=32),
      axis.text.x = element_text(margin=margin(20,0,0,0)),
      strip.background = element_blank()) +
    scale_y_continuous(limits=c(-0.01, max_y),
                       expand = c(0, 0, 0, 0),
                       breaks = seq(0, max_y, by=0.05),
                       minor_breaks = seq(0, max_y, 0.01)) +
    scale_x_discrete(labels=c('images' = '  images', 'barcodes' = 'DNA', 'fused' = 'fused '))
  
  if (job_id == 'Asteraceae' | job_id == 'Poaceae') {
    p <- p + theme(strip.text.x = element_text(size=30, face='bold', margin=margin(0,0,20,0)))
  } else {
    p <- p + theme(strip.text.x = element_blank())
  }
  
  return(p)
}

# helper method for arrangement of individual dataset plots with shared legend
fetch_legend <- function(genetic_dist_gcm, job_id) {
  p <- ggplot(genetic_dist_gcm, aes(x=data, y=species_dist, group=record_id, color=level)) +
    geom_line(alpha=.2, color='black', position=position_dodge(.3), linetype='longdash') +
    geom_point(aes(color=level, shape=duplicate, size=data_points), stroke = .5, position=position_dodge(.3)) +
    scale_shape_manual(values=c('intraspecific' = 23, 'combi' = 21, 'intrageneric' = 24,
                                'intergeneric' = 22, 'no duplicate' = 25)) +
    scale_color_manual(values=c('ground truth'=pal[[1]], 'intrageneric'=pal[[2]], 'intergeneric'=pal[[3]])) +
    guides(color=guide_legend(title="Confusion level", override.aes = list(size = 10)),
           shape=guide_legend(title="Duplicate level", override.aes = list(size = 10)),
           size=guide_legend(title="No. of samples")) +
    geom_line(alpha=.1, color='black')
  
  legend <- get_legend(
    # create some space to the left of the legend
    p + theme(legend.box.margin = margin(0, 0, 0, 0),
              legend.title=element_text(size=24, face='bold', margin=margin(0,0,30,0)),
              legend.text=element_text(size=22),
              legend.key.height=grid::unit(60, "pt"),
              legend.key.width=grid::unit(60, "pt"),
              legend.key=element_blank(),
              legend.background = element_blank())
  )
  
  return(legend)
}

# iterates over datasets to generate plots
plots = vector("list", length = 4)
legends = vector("list", length = 4)

just_wrong_plots = vector("list", length = 4)
dna_correct_plots = vector("list", length = 4)
image_correct_plots = vector("list", length = 4)
just_wrong_legends = vector("list", length = 4)

for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  datalist[[i]]$data <- factor(datalist[[i]]$data, levels=c('barcodes', 'fused', 'images'), ordered=TRUE)
  datalist[[i]]$duplicate <- factor(datalist[[i]]$duplicate, 
                                    levels=c('intraspecific', 'combi', 'intrageneric', 
                                             'intergeneric', 'no duplicate'),
                                    ordered=TRUE)
  datalist[[i]]$level <- factor(datalist[[i]]$level, 
                                levels=c('ground truth', 'intrageneric', 'intergeneric'),
                                ordered=TRUE)
  
  overall_results <- read.table(str_interp('${base_dir}/results/${job_id}/results.tsv'), fill=TRUE, sep='\t', header=TRUE)
  
  datalist[[i]]$species_dist_rounded <- round(datalist[[i]]$species_dist, 2)
  
  # just wrong - supplement figure
  print('just wrong')
  just_wrong_datalist <- datalist[[i]][datalist[[i]]$who_helps == 'just wrong',]
  pooled_data <- just_wrong_datalist %>%
    group_by(species_dist_rounded, level, who_helps, duplicate, data) %>%
    mutate(data_points = n())
  pooled_data <- pooled_data[!duplicated(pooled_data[c('species_dist_rounded', 'level', 'who_helps', 'duplicate', 'data')]),]
  p <- plot_conf(just_wrong_datalist, pooled_data, job_id, 'fused')
  just_wrong_legends[[i]] <- fetch_legend(pooled_data, job_id)
  just_wrong_plots[[i]] <- p
  
  # image correct - supplement figure
  print('image correct')
  just_wrong_datalist <- datalist[[i]][datalist[[i]]$who_helps == 'image correct',]
  pooled_data <- just_wrong_datalist %>%
    group_by(species_dist_rounded, level, who_helps, duplicate, data) %>%
    mutate(data_points = n())
  pooled_data <- pooled_data[!duplicated(pooled_data[c('species_dist_rounded', 'level', 'who_helps', 'duplicate', 'data')]),]
  p <- plot_conf(just_wrong_datalist, pooled_data, job_id, 'fused')
  just_wrong_legends[[i]] <- fetch_legend(pooled_data, job_id)
  image_correct_plots[[i]] <- p
  
  # DNA correct - supplement figure
  print('dna correct')
  just_wrong_datalist <- datalist[[i]][datalist[[i]]$who_helps == 'DNA correct',]
  pooled_data <- just_wrong_datalist %>%
    group_by(species_dist_rounded, level, who_helps, duplicate, data) %>%
    mutate(data_points = n())
  pooled_data <- pooled_data[!duplicated(pooled_data[c('species_dist_rounded', 'level', 'who_helps', 'duplicate', 'data')]),]
  p <- plot_conf(just_wrong_datalist, pooled_data, job_id, 'fused')
  just_wrong_legends[[i]] <- fetch_legend(pooled_data, job_id)
  dna_correct_plots[[i]] <- p
  
  # other constellations
  print('main figs')
  datalist[[i]] <- datalist[[i]][(datalist[[i]]$who_helps != 'just wrong') & (datalist[[i]]$who_helps != 'DNA correct') & (datalist[[i]]$who_helps != 'image correct'),]
  pooled_data <- datalist[[i]]
  pooled_data <- pooled_data %>%
    group_by(species_dist_rounded, level, who_helps, duplicate, data) %>%
    mutate(data_points = n())
  
  pooled_data <- pooled_data[!duplicated(pooled_data[c('species_dist_rounded', 'level', 'who_helps', 'duplicate', 'data')]),]
  
  p <- plot_conf(datalist[[i]], pooled_data, job_id, 'fused')
  legends[[i]] <- fetch_legend(pooled_data, job_id)
  plots[[i]] <- p
}

# merges plots
merged_plot <- ggarrange(
  plots[[1]], plots[[2]], plots[[3]], plots[[4]],
  nrow = 2,
  ncol = 2
) +
  theme(plot.margin = margin(.1,.1,.1,.1, "cm"))

merged_plot <- annotate_figure(merged_plot,
                               left = text_grob("Mean genetic distance (sample to predicted species)", color = "black", size = 36, rot=90),
                               bottom = text_grob("Models", color = "black", size = 36))

plot_grid(merged_plot, legends[[1]], rel_widths = c(15, 1.6))

ggsave(str_interp('${base_dir}/plots/dist_duplicate_confusion.pdf'),
       device='pdf',
       width=30,
       height=20,
       dpi='retina')

# merges supplement plots (all wrong)
merged_plot <- ggarrange(
  just_wrong_plots[[1]], just_wrong_plots[[2]], just_wrong_plots[[3]], just_wrong_plots[[4]],
  nrow = 2,
  ncol = 2
) +
  theme(plot.margin = margin(.1,.1,.1,.1, "cm"))

merged_plot <- annotate_figure(merged_plot,
                               left = text_grob("Mean genetic distance (sample to predicted species)", color = "black", size = 36, rot=90),
                               bottom = text_grob("Models", color = "black", size = 36))

plot_grid(merged_plot, just_wrong_legends[[1]], rel_widths = c(15, 1.6))

ggsave(str_interp('${base_dir}/plots/dist_duplicate_confusion_supp.pdf'),
       device='pdf',
       width=30,
       height=20,
       dpi='retina')

# merges supplement plots (image correct)
merged_plot <- ggarrange(
  plotlist = image_correct_plots,
  nrow = 2,
  ncol = 2
) +
  theme(plot.margin = margin(.1,.1,.1,.1, "cm"))

merged_plot <- annotate_figure(merged_plot,
                               left = text_grob("Mean genetic distance (sample to predicted species)", color = "black", size = 36, rot=90),
                               bottom = text_grob("Models", color = "black", size = 36))

plot_grid(merged_plot, just_wrong_legends[[1]], rel_widths = c(15, 1.6))

ggsave(str_interp('${base_dir}/plots/dist_duplicate_confusion_supp_image.pdf'),
       device='pdf',
       width=30,
       height=20,
       dpi='retina')

# merges supplement plots (dna correct)
merged_plot <- ggarrange(
  dna_correct_plots[[1]], dna_correct_plots[[2]], dna_correct_plots[[3]], dna_correct_plots[[4]],
  nrow = 2,
  ncol = 2
) +
  theme(plot.margin = margin(.1,.1,.1,.1, "cm"))

merged_plot <- annotate_figure(merged_plot,
                               left = text_grob("Mean genetic distance (sample to predicted species)", color = "black", size = 36, rot=90),
                               bottom = text_grob("Models", color = "black", size = 36))

plot_grid(merged_plot, just_wrong_legends[[1]], rel_widths = c(15, 1.6))

ggsave(str_interp('${base_dir}/plots/dist_duplicate_confusion_supp_dna.pdf'),
       device='pdf',
       width=30,
       height=20,
       dpi='retina')
