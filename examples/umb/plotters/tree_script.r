library(ggtree)
library(tidyverse)

table <- read_tsv(
  "D:\\chronostrain\\umb\\output\\phylogeny\\ClermonTyping\\umb_phylogroups.txt",
  col_names = c("strain_fasta", "genes", "presence", "ambiguous", "phylogroup", "tab_output"),
  show_col_types = FALSE
)
table$strain = substr(table$strain_fasta,1,nchar(table$strain_fasta)-6)
print(table)

tree <- read.tree("D:\\chronostrain\\umb\\output\\phylogeny\\tree\\tree.nwk")


# TODO: Learn out how to extract this from column of table.
phylogroups <- c("A", "B1", "B2", "C", "D", "E", "F", "G", "fergusonii", "albertii", "E or cladeI", "cladeI", "Unknown", "cladeV")
cls <- list()

for (pgroup in phylogroups)
{
  result <- filter(table, phylogroup == pgroup)
  cls <- append(cls, list(deframe(result[,"strain"])))
}
names(cls) <- phylogroups

tree <- groupOTU(tree, cls)

# ========= Determine which strains to label
tip_df <- data.frame(
  #strain=c('NZ_CP026028.1', 'NZ_CP031256.1', 'NZ_CP049852.1', 'NZ_LR882052.1', 'NZ_CP023371.1', 'NZ_LR134075.1', 'NZ_CP045277.1', 'NZ_CP025903.1', 'NZ_CP024886.1', 'NZ_CP051688.1', 'NZ_LR134231.1', 'NZ_CP048920.1', 'NZ_CP040269.1', 'NZ_CP051001.1', 'NZ_CP076706.1', 'NZ_CP026932.2', 'NZ_CP033250.1', 'NZ_CP070162.1', 'NZ_CP070026.1', 'NZ_CP022664.1', 'NZ_CP053079.1', 'NZ_CP047594.1', 'NZ_CP053296.1', 'NZ_CP069657.1', 'NZ_LR134152.1', 'NZ_CP080223.1', 'NZ_CP012379.1', 'NZ_CP054345.1', 'NZ_CP022279.1', 'NZ_LS992166.1', 'NZ_LS992171.1'),
  strain=c('asdf'),
  present=FALSE
)

# ========= Clade colors
color_df <- data.frame(
  phylogroup=c("A", "B1", "B2", "C", "D", "E", "F", "G", "fergusonii", "albertii", "E or cladeI", "cladeI", "Unknown", "cladeV"),
  color_code=c(
    rgb(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    rgb(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    rgb(1.0, 0.4980392156862745, 0.054901960784313725),
    rgb(1.0, 0.7333333333333333, 0.47058823529411764),
    rgb(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    rgb(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    rgb(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    rgb(1.0, 0.596078431372549, 0.5882352941176471),
    rgb(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    rgb(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    rgb(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    rgb(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    rgb(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    rgb(0.9686274509803922, 0.7137254901960784, 0.8235294117647058)
  )
)
color_vec <- setNames(as.character(color_df$color_code), color_df$phylogroup)

# ========= Render the tree
p <- ggtree(tree, aes(color=group))
p <- p %<+% color_df + scale_color_manual(values=color_vec)
#p <- p %<+% tip_df + geom_tiplab(
#  aes(label=label, subset=present), 
#  size=2, 
#  color="black"
#)


ggsave(
  filename="D:\\chronostrain\\tree_new.pdf", 
  plot=p,
  device="pdf", 
  width=12, 
  height=12, 
  units="in", 
  limitsize=FALSE
)
