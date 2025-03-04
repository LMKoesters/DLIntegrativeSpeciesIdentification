pacman::p_load("caret")
pacman::p_load("combinat")
pacman::p_load("cowplot")
pacman::p_load("dplyr")
pacman::p_load("forcats")
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
pacman::p_load("rcompanion")
pacman::p_load("rstatix")
pacman::p_load("RVAideMemoire")
pacman::p_load("stats")
pacman::p_load("stringr")
pacman::p_load("svglite")
pacman::p_load("tidyr")
pacman::p_load("tidyverse")
pacman::p_load("wesanderson")

base_dir = ''
pal <- c("#2A5D8D", "#4387C7", "#81AFD9", "#AF301D", "#E0543E", "#EB9486")
border_pal <- c("#17344F", "#2A5D8D", "#4387C7", "#691D11", "#AF301D", "#E0543E")
pal <- c("#2A5D8D", "#AF301D")

# assign letters to barcode processings based on statistics and overall classification accuracy
collect_letters <- function(loocv_results, stat_results, test="Cochran's Q") {
  sorted_acc <- loocv_results[!duplicated(loocv_results$bar_prep), c('bar_prep', 'accuracy_loocv')]
  sorted_acc <- sorted_acc[order(sorted_acc$accuracy_loocv, decreasing = TRUE),]
  sorted_acc['idx'] <- 1:nrow(sorted_acc)
  
  if (stat_results[stat_results$test == test,]$p < 0.05) {
    stat_results <- stat_results[stat_results$test == str_interp('${test} pairw'),]
    
    if (any(stat_results$p.adj < 0.05)) {
      stat_results <- merge(stat_results, sorted_acc[c('bar_prep', 'idx')], by.x='group1', by.y='bar_prep')
      stat_results <- merge(stat_results, sorted_acc[c('bar_prep', 'idx')], by.x='group2', by.y='bar_prep', 
                            suffixes = c('1', '2'))
      # for accessibility
      stat_results <- stat_results %>%
        rowwise() %>%
        mutate(comb = paste(sort(c(idx1, idx2)), collapse='|'))
      
      letter = 'A'
      for (i in 1:nrow(sorted_acc)) {
        if (i == 1) { # highest accuracy
          sorted_acc[1, 'letter'] <- letter
        } else if (i != nrow(sorted_acc)) { # everything between highest and lowest accuracy
          # capital letter in comment is the model we're assigning the letter to at the moment ;)
          if (stat_results[stat_results$comb == paste(i-1, i, sep='|'),]$p.adj < 0.05) {
            # i and i-1 are significantly different (a-B lettering)
            new_letter <- LETTERS[match(letter, LETTERS)+1]
            sorted_acc[i, 'letter'] <- new_letter
            letter <- new_letter
          } else if (stat_results[stat_results$comb == paste(i, i+1, sep='|'),]$p.adj < 0.05) {
            # a-A-b
            sorted_acc[i, 'letter'] <- letter
          } else if (stat_results[stat_results$comb == paste(i-1, i+1, sep='|'),]$p.adj < 0.05) {
            # a-AB-b
            new_letter <- LETTERS[match(letter, LETTERS)+1]
            sorted_acc[i, 'letter'] <- paste(letter, new_letter, sep='')
            letter <- new_letter
          } else {
            # a-A-a
            sorted_acc[i, 'letter'] <- letter
          }
        } else { # lowest accuracy
          if (stat_results[stat_results$comb == paste(i-1, i, sep='|'),]$p.adj < 0.05) {
            # i and i-1 are significantly different (a-B lettering)
            sorted_acc[i, 'letter'] <- LETTERS[match(letter, LETTERS)+1]
          } else {
            # a-A-end
            sorted_acc[i, 'letter'] <- letter
          }
        }
      }
    }
    # add letters to results dataframe
    loocv_results <- merge(loocv_results, sorted_acc[c('bar_prep', 'letter')], by='bar_prep')
  } else {
    loocv_results['letter'] <- ''
  }
  return(loocv_results)
}

# run non-parametric Cochran's Q for paired data to test differences in proportions (here: successful vs failed classification)
run_cochran <- function(loocv_results, stat_results) {
  # run Cochran's Q
  cochrans_loocv <- loocv_results[,c('val_idx', 'bar_prep', 'val_acc')]
  
  # Cochran's Q Test table prep
  cochrans_loocv$bar_prep <- factor(
    cochrans_loocv$bar_prep, levels = unique(cochrans_loocv$bar_prep)
  )
  cochrans_loocv$val_acc <- factor(
    cochrans_loocv$val_acc, levels = c(1, 0)
  )
  
  # run Cochran's Q
  cochran <- cochran.qtest(val_acc ~ bar_prep|val_idx, data=cochrans_loocv)
  stat_results[nrow(stat_results) + 1,] <- c("Cochran's Q", NA, NA, as.numeric(cochran$p.value), NA, cochran$statistic[["Q"]], cochran$parameter[["df"]])
  
  if (cochran$p.value < 0.05) {
    # run post-hoc pairwise McNemar
    mcnemar <- pairwise_mcnemar_test(cochrans_loocv, val_acc ~ bar_prep|val_idx, p.adjust.method = "holm")
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
  
  # add significance indicators :)
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
  
  # write file with p-values
  stat_results <- stat_results %>%
    mutate(job_id = job_id, .before = colnames(stat_results)[1])
  
  if (file.exists(str_interp("${base_dir}/statistics/bar_prep_loocv_acc.tsv"))) {
    stat_results_total <- read.table(str_interp("${base_dir}/statistics/bar_prep_loocv_acc.tsv"), sep = '\t', header = TRUE)
    stat_results <- rbind(stat_results_total, stat_results) 
  }
  
  write.table(stat_results, str_interp("${base_dir}/statistics/bar_prep_loocv_acc.tsv"), sep = '\t', quote = FALSE, row.names = FALSE)
  loocv_results <- collect_letters(loocv_results, stat_results)
  # write file with accuracy and loss for barcode processing/encoding combination
  loocv_results_means <- loocv_results[!duplicated(loocv_results$bar_prep), c('round_1', 'round_2', 'barcode_processing',
                                                                              'barcode_encoding', 'accuracy_loocv', 'mean_loss', 'letter')]
  loocv_results_means <- loocv_results_means %>%
    mutate(job_id = job_id, .before = colnames(loocv_results_means)[1])
  
  if (file.exists(str_interp("${base_dir}/statistics/bar_prep_loocv_means.tsv"))) {
    loocv_results_means_total <- read.table(str_interp("${base_dir}/statistics/bar_prep_loocv_means.tsv"), sep = '\t', header = TRUE)
    loocv_results_means <- rbind(loocv_results_means_total, loocv_results_means) 
  }
  
  write.table(loocv_results_means,
              str_interp("${base_dir}/statistics/bar_prep_loocv_means.tsv"), sep = '\t', quote = FALSE, row.names = FALSE) 
  
  return(loocv_results)
}

# main part that iterates over datasets
jobs <- c('Asteraceae', 'Poaceae', 'Coccinellidae', 'Lycaenidae')
results_dfs = vector("list", length = length(jobs))

if (file.exists(str_interp("${base_dir}/statistics/bar_prep_loocv_acc.tsv"))) {
  file.remove(str_interp("${base_dir}/statistics/bar_prep_loocv_acc.tsv"))
}

if (file.exists(str_interp("${base_dir}/statistics/bar_prep_loocv_means.tsv"))) {
  file.remove(str_interp("${base_dir}/statistics/bar_prep_loocv_means.tsv"))
}

for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  print(job_id)
  
  if (job_id %in% c('Asteraceae', 'Poaceae')) {
    marker = 'rbcLa'
  } else {
    marker = 'COI-5P'
  }
  
  # load results
  loocv_results = read.table(str_interp('${base_dir}/results/loocv/${job_id}/results_total.tsv'), sep='\t', header=TRUE)
  
  # subset results
  loocv_results = loocv_results[loocv_results$round_2 == 'bar',]
  
  # calculcate accuracy per barcode processing/encoding combinations
  loocv_results <- loocv_results %>%
    group_by(barcode_processing, barcode_encoding) %>%
    mutate(accuracy_loocv = mean(val_acc),
           mean_loss = mean(val_loss))
  
  # for easier accessibility
  loocv_results$bar_prep <- paste(loocv_results$barcode_encoding, loocv_results$barcode_processing, sep = '|')
  
  # calculate statistics
  loocv_results <- run_stats(job_id, loocv_results)
  
  loocv_results$barcode_processing <- factor(loocv_results$barcode_processing, levels = sort(unique(loocv_results$barcode_processing)))
  loocv_results$dataset <- job_id
  results_dfs[[i]] <- loocv_results
}

# plot results
plots = vector("list", length = 4)

for (i in 1:length(jobs)) {
  job_id <- jobs[[i]]
  print(job_id)
  overall_results <- results_dfs[[i]]
  
  overall_results <- overall_results[!duplicated(overall_results[c('bar_prep', 'dataset')]),]
  overall_results$accuracy_label <- paste(round(overall_results$accuracy_loocv * 100, 1), '%', sep='')
  overall_results$dataset <- factor(overall_results$dataset, levels = c('Asteraceae', 'Poaceae', 'Coccinellidae', 'Lycaenidae'))
  
  overall_results <- overall_results %>%
    mutate(al = case_when(barcode_processing == 'aligned_barcode' ~ 1,
                          barcode_processing == 'aligned_barcode_snp' ~ .7,
                          barcode_processing == 'padded_barcode' ~ .4))
  
  overall_results$barcode_processing <- factor(overall_results$barcode_processing, levels = c('aligned_barcode', 'aligned_barcode_snp', 'padded_barcode'))
  
  p <- ggplot(overall_results, aes(x=barcode_processing, y=accuracy_loocv * 100, fill=barcode_encoding, alpha=al)) +
    geom_bar(stat = 'identity', position=position_dodge(width = 0.9)) +
    geom_text(aes(y=5, label=accuracy_label, alpha=1), size=12, color='black', check_overlap = T) +
    geom_text(aes(y=102, label=letter, alpha=1), size=12, color='black', check_overlap = T) +
    theme_minimal() +
    facet_grid(. ~ barcode_encoding, scales = "free", space='free', labeller = as_labeller(c('one_hot_bar' = 'fractional', 'sequential_bar' = 'ordinal'))) +
    scale_fill_manual(name='DNA encoding', values = pal, guide='none') +
    labs(y="", x="", title=job_id) +
    theme(plot.title = element_text(size=46, face='bold', hjust=0.5),
          plot.background = element_rect(color = "black", fill = NA, linewidth = .5),
          legend.position = "none",
          legend.key.size = unit(2.5, "lines"),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.border = element_blank(),
          strip.text.x = element_text(size=34, face='bold'),
          axis.title=element_text(size=42),
          axis.title.x = element_blank(),
          axis.text=element_text(size=28)) +
    scale_alpha_identity(breaks = c(1, .7, .4), labels = c("aligned", "aligned-SNP", "unaligned"), guide='legend') +
    scale_x_discrete(labels = c('aligned_barcode' = 'aligned',
                                'aligned_barcode_snp' = 'aligned\nSNP',
                                'padded_barcode' = 'unaligned')) +
    guides(alpha = guide_legend(
      title='DNA arrangement',
      override.aes = aes(label = "")))
  
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
) +
  theme(plot.margin = margin(.1,.1,1.5,.1, "cm"))

annotate_figure(p,
                left = text_grob("Identification accuracy [%]", color = "black", size = 42, rot=90),
                bottom = text_grob("Barcode arrangement", color = 'black', size = 42))
