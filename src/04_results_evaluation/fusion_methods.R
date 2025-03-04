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
bar_pal <- c("#CC8B86", "#F8AFA8", "#F5CDB4", "#E8AE68", "#FFD275", "#9DC4B5", "#8E9AAF", "#CBC0D3", "#DEE2FF")

# assign significance letters to unimodal/multimodal methods
collect_letters <- function(loocv_results, stat_results, test="Cochran's Q") {
  sorted_acc <- loocv_results[!duplicated(loocv_results$round_2), c('round_2', 'accuracy_loocv')]
  sorted_acc <- sorted_acc[order(sorted_acc$accuracy_loocv, decreasing = TRUE),]
  sorted_acc['idx'] <- 1:nrow(sorted_acc)
  
  if (stat_results[stat_results$test == test,]$p < 0.05) {
    stat_results <- stat_results[stat_results$test == str_interp('${test} pairw'),]
    
    if (any(stat_results$p.adj < 0.05)) {
      stat_results <- merge(stat_results, sorted_acc[c('round_2', 'idx')], by.x='group1', by.y='round_2')
      stat_results <- merge(stat_results, sorted_acc[c('round_2', 'idx')], by.x='group2', by.y='round_2', suffixes = c('1', '2'))
      stat_results <- stat_results %>%
        rowwise() %>%
        mutate(comb = paste(sort(c(idx1, idx2)), collapse='|'))
      
      # because Coccinellidae dense and dense_late are the same and p.adj is NA in that case
      stat_results[is.na(stat_results$p.adj), 'p.adj'] <- 1
      
      letter = 'A'
      for (i in 1:nrow(sorted_acc)) {
        if (i == 1) {
          sorted_acc[1, 'Letter'] <- letter
        } else if (i != nrow(sorted_acc)) {
          if (stat_results[stat_results$comb == paste(i-1, i, sep='|'),]$p.adj < 0.05) {
            # i and i-1 are significantly different (a-B lettering)
            new_letter <- LETTERS[match(letter, LETTERS)+1]
            sorted_acc[i, 'Letter'] <- new_letter
            letter <- new_letter
          } else if (stat_results[stat_results$comb == paste(i, i+1, sep='|'),]$p.adj < 0.05) {
            # a-A-b
            sorted_acc[i, 'Letter'] <- letter
          } else if (stat_results[stat_results$comb == paste(i-1, i+1, sep='|'),]$p.adj < 0.05) {
            # a-AB-b
            new_letter <- LETTERS[match(letter, LETTERS)+1]
            sorted_acc[i, 'Letter'] <- paste(letter, new_letter, sep='')
            letter <- new_letter
          } else {
            # a-A-a
            sorted_acc[i, 'Letter'] <- letter
          }
        } else {
          if (stat_results[stat_results$comb == paste(i-1, i, sep='|'),]$p.adj < 0.05) {
            # i and i-1 are significantly different (a-B lettering)
            sorted_acc[i, 'Letter'] <- LETTERS[match(letter, LETTERS)+1]
          } else {
            # a-A-end
            sorted_acc[i, 'Letter'] <- letter
          }
        }
      }
    }
    loocv_results <- merge(loocv_results, sorted_acc[c('round_2', 'Letter')], by='round_2')
  } else {
    loocv_results['Letter'] <- ''
  }
  return(loocv_results)
}

# run non-parametric Cochran's Q for paired data to test differences in proportions (here: successful vs failed classification)
run_cochran <- function(loocv_results, stat_results) {
  # run Cochran's Q
  cochrans_loocv <- loocv_results[,c('val_idx', 'round_2', 'val_acc')]
  
  # Cochran's Q Test table prep
  cochrans_loocv$round_2 <- factor(
    cochrans_loocv$round_2, levels = unique(cochrans_loocv$round_2)
  )
  cochrans_loocv$val_acc <- factor(
    cochrans_loocv$val_acc, levels = c(1, 0)
  )
  
  # run Cochran's Q
  cochran <- cochran.qtest(val_acc ~ round_2|val_idx, data=cochrans_loocv)
  stat_results[nrow(stat_results) + 1,] <- c("Cochran's Q", NA, NA, as.numeric(cochran$p.value), NA, cochran$statistic[["Q"]], cochran$parameter[["df"]])
  
  if (cochran$p.value < 0.05) {
    # run post-hoc pairwise McNemar
    mcnemar <- pairwise_mcnemar_test(cochrans_loocv, val_acc ~ round_2|val_idx, p.adjust.method = "holm")
    mcnemar['test'] <- "Cochran's Q pairw"
    mcnemar['test_val'] <- NA
    mcnemar['df'] <- NA
    stat_results <- rbind(stat_results, mcnemar[,c('test', 'group1', 'group2', 'p', 'p.adj', 'test_val', 'df')])
  }
  
  return(stat_results)
}

# run statistical test and assign letters according to accuracy and statistics
run_stats <- function(job_id, loocv_results) {
  stat_results <- setNames(data.frame(matrix(ncol = 7, nrow = 0)), c("test", "group1", "group2", "p", "p.adj", "test_val", "df"))
  stat_results <- run_cochran(loocv_results, stat_results)
  
  stat_results$p <- as.numeric(stat_results$p)
  stat_results$p.adj <- as.numeric(stat_results$p.adj)
  
  stat_results <- stat_results %>%
    mutate(sign = case_when(
      is.na(p.adj) ~ case_when(
        p < 0.0001 ~ '****',
        p < 0.001 ~ '***',
        p < 0.01 ~ '**',
        p < 0.05 ~ '*',
        TRUE ~ 'ns'
      ),
      !is.na(p.adj) ~ case_when(
        p.adj < 0.0001 ~ '****',
        p.adj < 0.001 ~ '***',
        p.adj < 0.01 ~ '**',
        p.adj < 0.05 ~ '*',
        TRUE ~ 'ns'
      )))
  
  stat_results_acc <- stat_results %>%
    mutate(job_id = job_id, .before = colnames(stat_results)[1])
  
  if (file.exists(str_interp("${base_dir}/statistics/fusion_loocv_acc.tsv"))) {
    stat_results_acc_total <- read.table(str_interp("${base_dir}/statistics/fusion_loocv_acc.tsv"), sep = '\t', header = TRUE, quote = NULL)
    stat_results_acc <- rbind(stat_results_acc_total, stat_results_acc) 
  }
  
  write.table(stat_results_acc, str_interp("${base_dir}/statistics/fusion_loocv_acc.tsv"), sep = '\t', quote = FALSE, row.names = FALSE)
  loocv_results <- collect_letters(loocv_results, stat_results)
  
  return(loocv_results[!duplicated(loocv_results$round_2),])
}

wrapper <- function(x) paste(strsplit(x, "")[[1]], collapse = "\n")

# change color slightly according to parameters
alter_col <- function(col, ds, dv, alph, grey_out=TRUE) {
  if (grey_out) {
    col = rgb2hsv(col2rgb(col))
    col["v", ] = col["v", ] + dv * (1 - col["v", ])
    col["s", ] = ds * col["s", ]
    col = hsv(col[1], col[2], col[3])
  }
  col = alpha(col, alpha=alph)
  return(col)
}

# load LOOCV results
add_loocv <- function(job_id, overall_results) {
  if (file.exists(str_interp('${base_dir}/results/loocv/${job_id}/results_total.tsv'))) {
    bar_prep = 'aligned_barcode'
    marker = 'rbcLa'
    
    if (job_id == 'Coccinellidae') {
      to_test = c('bar', 'img', 'BLAST', 'dense', 'dense_late')
      marker = 'COI-5P'
    } else if (job_id == 'Poaceae') {
      to_test = c('bar', 'img', 'BLAST', 'dense', 'rf')
    } else if (job_id == 'Asteraceae') {
      to_test = c('bar', 'img', 'BLAST', 'slf-product_score')
    } else {
      to_test = c('bar', 'img', 'BLAST', 'dense')
      marker = 'COI-5P'
    }
    
    loocv_results <- read.table(str_interp('${base_dir}/results/loocv/${job_id}/results_total.tsv'),
                                fill=TRUE, sep='\t', header=TRUE)
    
    loocv_results <- loocv_results[(loocv_results$round_2 %in% to_test) & (((loocv_results$barcode_processing == bar_prep) & (loocv_results$barcode_encoding == 'one_hot_bar')) | (loocv_results$round_2 == 'BLAST')),]
    
    loocv_results <- loocv_results[order(loocv_results$val_idx),]
    
    loocv_results <- loocv_results %>%
      group_by(round_2) %>%
      mutate(accuracy_loocv = mean(val_acc),
             mean_loss = mean(val_loss))
    
    loocv_results <- run_stats(job_id, loocv_results)
    loocv_results <- loocv_results[!duplicated(loocv_results[c('barcode_processing', 'barcode_encoding', 'round_1', 'round_2')]),]
    overall_results <- merge(overall_results, loocv_results[c('round_1', 'round_2', 'accuracy_loocv', 'mean_loss', 'Letter')], all.x=TRUE)
  } else {
    overall_results$Letter <- ''
    overall_results$accuracy_loocv <- 0.0
    overall_results$mean_loss <- 0.0
  }
  
  overall_results[(is.na(overall_results$Letter)) | (overall_results$Letter == ''), 'Letter'] <- ''
  
  overall_results_means <- overall_results[!duplicated(overall_results$round_2), c('round_1', 'round_2', 'barcode_processing', 
                                                                                   'barcode_encoding', 'accuracy_loocv', 'mean_loss', 'Letter')]
  overall_results_means <- overall_results_means %>%
    mutate(job_id = job_id, .before = colnames(overall_results_means)[1])
  
  if (file.exists(str_interp("${base_dir}/statistics/fusion_loocv_means.tsv"))) {
    overall_results_means_total <- read.table(str_interp("${base_dir}/statistics/fusion_loocv_means.tsv"), sep = '\t', header = TRUE, quote = NULL)
    overall_results_means <- rbind(overall_results_means_total, overall_results_means)
  }
  
  write.table(overall_results_means, str_interp("${base_dir}/statistics/fusion_loocv_means.tsv"), sep = '\t', quote = FALSE, row.names = FALSE)
  return(overall_results)
}

# main part; iterates over datasets
jobs <- c('Asteraceae', 'Poaceae', 'Coccinellidae', 'Lycaenidae')
results_dfs = vector("list", length = 4)
plots = vector("list", length = 4)

if (file.exists(str_interp("${base_dir}/statistics/fusion_loocv_means.tsv"))) {
  file.remove(str_interp("${base_dir}/statistics/fusion_loocv_means.tsv"))
}

if (file.exists(str_interp("${base_dir}/statistics/fusion_loocv_acc.tsv"))) {
  file.remove(str_interp("${base_dir}/statistics/fusion_loocv_acc.tsv"))
}

for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  print(job_id)
  
  bar_prep = 'aligned_barcode'
  bar_enc = 'one_hot_bar'
  
  overall_results <- read.table(str_interp('${base_dir}/results/${job_id}/results.tsv'), fill=TRUE, 
                                sep='\t', header=TRUE)
  overall_results <- overall_results[(((overall_results$barcode_processing == bar_prep) &
                                         (overall_results$barcode_encoding == bar_enc)) |
                                        (overall_results$round_2 == 'img')) | (overall_results$round_2 == 'BLAST'),]
  
  overall_results <- overall_results[overall_results$round_1 == 'sep',]
  best_val <- max(overall_results[(overall_results$round_2 == 'bar') | (overall_results$round_2 == 'img'),]$val_acc)
  
  overall_results <- overall_results %>%
    mutate(classifier = case_when(
      round_2 == 'bar' ~ 'DNA',
      round_2 == 'img' ~ 'image',
      round_2 == 'dense' ~ 'mid fusion',
      round_2 == 'dense_single' ~ 'single layer',
      round_2 == 'dense_late' ~ 'late fusion',
      round_2 == 'slf-sum_score' ~ 'sum score',
      round_2 == 'slf-max_score' ~ 'max score',
      round_2 == 'slf-product_score' ~ 'product score',
      round_2 == 'BLAST' ~ 'BLAST',
      TRUE ~ 'random forest'
    ),
    pre_training = case_when(
      round_1 == 'sep' ~ 'unimodal',
      round_1 == 'dense' ~ 'mid\nfusion',
      round_1 == 'dense_late' ~ 'late\nfusion',
      round_1 == 'slf-sum_score' ~ 'sum\nscore',
      round_1 == 'slf-max_score' ~ 'max\nscore',
      round_1 == 'slf-product_score' ~ 'product\nscore'
    ),
    group = case_when(
      (round_2 == 'bar') | (round_2 == 'img') | (round_2 == 'BLAST') ~ 'unimodal',
      (round_2 == 'dense') | (round_2 == 'rf') ~ 'mid',
      round_2 == 'dense_late' ~ 'late',
      grepl('slf', round_2, fixed = TRUE) ~ 'score level'
    ),
    best_x = case_when(
      (round_2 == 'bar') | (round_2 == 'img') | (round_2 == 'BLAST') ~ TRUE,
      val_acc > best_val ~ TRUE,
      TRUE ~ FALSE
    ))
  
  overall_results$group <- factor(overall_results$group, 
                                  levels = c('unimodal', 'mid', 'late', 'score level'),
                                  ordered=TRUE)
  classifier_levels <- c('DNA', 'image', 'BLAST', 
                         'random forest', 'mid fusion', 'late fusion', 
                         'max score', 'product score', 'sum score')
  overall_results$classifier <- factor(overall_results$classifier, 
                                       levels = classifier_levels, 
                                       ordered=TRUE)
  overall_results$dataset <- job_id
  
  overall_results <- add_loocv(job_id, overall_results)
  overall_results$color_group <- paste(overall_results$group, overall_results$classifier,
                                       as.character(overall_results$best_x))
  overall_results$color_group <- factor(overall_results$color_group, 
                                        levels = overall_results[order(overall_results$group,
                                                                       overall_results$classifier),][['color_group']], 
                                        ordered=TRUE)
  
  colors <- setNames(data.frame(bar_pal[1:length(classifier_levels)], classifier_levels), c('col', 'classifier'))
  overall_results <- merge(overall_results, colors, by='classifier', all.x=TRUE)
  overall_results <- overall_results %>%
    rowwise %>%
    mutate(col = case_when(
      TRUE ~ col),
      loocv_col = alter_col(col, ds=0.2, dv=0.5, alph=0.3, grey_out=FALSE))
  
  overall_results$val_acc <- as.numeric(overall_results$val_acc)
  results_dfs[[i]] <- overall_results
}

# plot results for each of the datasets
for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  print(job_id)
  overall_results <- results_dfs[[i]]
  
  pal <- overall_results[order(overall_results$color_group),][['col']]
  loocv_pal <- overall_results[order(overall_results$color_group),][['loocv_col']]
  overall_results$perc <- round(overall_results$val_acc * 100, 1)
  overall_results$perc_loocv <- round(overall_results$accuracy_loocv * 100, 1)
  bar_acc <- overall_results[overall_results$round_2 == 'bar',][['perc']]  
  overall_results$perc_str <- paste(round(overall_results$val_acc * 100, 1), '%', sep='')
  overall_results <- overall_results %>%
    group_by(group) %>%
    mutate(n = n(), fac = n / 3)
  
  p <- ggplot(overall_results, aes(x=classifier, fill=classifier)) +
    geom_bar(stat = 'identity', aes(y=perc_loocv, alpha=0.5)) +
    geom_bar(stat = 'identity', aes(y=perc)) +
    scale_fill_manual(values = bar_pal) +
    geom_text(data=overall_results, aes(y=102, x=classifier, label=Letter), size = 11, color='black') +
    geom_text(data=overall_results, aes(y=5, x=classifier, label=perc_str), size = 9, color='black') +
    geom_hline(yintercept = bar_acc, colour=bar_pal[1]) +
    labs(x="", y="", title=job_id) +
    facet_grid(. ~ group, scales = "free", space='free') +
    theme(plot.title = element_text(size=46, face='bold', hjust=0.5),
          plot.background = element_rect(color = "grey", fill = NA, linewidth = .5),
          panel.grid.major.x = element_line(),
          panel.spacing = unit(0, "lines"),
          legend.position = "none",
          panel.background = element_blank(),
          panel.border = element_blank(),
          strip.text.x = element_text(size=34, face='bold'),
          strip.background = element_blank(),
          axis.ticks.x = element_blank(),
          axis.title=element_text(size=42),
          axis.title.y = element_blank(),
          axis.title.x = element_blank(),
          axis.text=element_text(size=30),
          plot.margin = unit(c(.5,.5,.5,.5), "cm")) +
    scale_x_discrete(labels=c('image' = 'image', 'barcode' = 'barcode', 'BLAST' = 'BLAST',
                              'max score' = 'max', 'product score' = 'product',
                              'sum score' = 'sum', 'mid fusion' = 'FC',
                              'late fusion' = 'FC', 'random forest' = 'RF')) +
    scale_y_continuous(limits=c(0, 102), breaks = seq(0, 100, 20), expand = c(0, 0, .05, 0))
  
  if (i == 1 | i == 3) {
    p <- p +
      theme(axis.text.y = element_text(size=32))
  } else {
    p <- p +
      theme(axis.text.y = element_blank()) +
      theme(axis.ticks.y = element_blank())
  }
  
  plots[[i]] <- p
}

# merge plots
p <- ggarrange(
  plots[[1]], plots[[2]], plots[[3]], plots[[4]],
  nrow = 2,
  ncol = 2
)

annotate_figure(p,
                left = text_grob("Identification accuracy", color = "black", size = 42, rot=90))

