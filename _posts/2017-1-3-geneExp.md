---
layout: post
title: Statistical analysis of gene expression data
---

Data: RNASeq data  
Techniques: Multiple comparison testing, clustering, PCA

Code:  
[Analysis of gene expression data](https://github.com/JoomiK/GeneExpression) 

---


This document goes through the workflow for differential expression of RNA-Seq (gene expression) data, starting with raw counts of sequencing reads.

### The Pasilla dataset:
It contains RNA-Seq count data for RNAi treated and untreated *Drosophila melanogaster* cells.

Install `gplots`, `RColorBrewer` and Bioconductor packages `DESeq` and `pasilla` if needed.


### The Workflow:
Much of the workflow follows the vignette in this [DESeq manual.](http://bioconductor.org/packages/release/bioc/vignettes/DESeq/inst/doc/DESeq.pdf)

First get the path to the file containing the raw count data. The file countains counts for each gene(row) in each sample (column). Here we're loading the file using read.table and storing as count.table. Header = TRUE if the first row of your file contains the names of variables.


```r
datafile <- system.file( "extdata/pasilla_gene_counts.tsv", package="pasilla" )
count.table <- read.table(datafile, header=TRUE, row.names=1)
```

The first few rows of data:

```r
head(count.table)
```

```
##             untreated1 untreated2 untreated3 untreated4 treated1 treated2
## FBgn0000003          0          0          0          0        0        0
## FBgn0000008         92        161         76         70      140       88
## FBgn0000014          5          1          0          0        4        0
## FBgn0000015          0          2          1          2        1        0
## FBgn0000017       4664       8714       3564       3150     6205     3072
## FBgn0000018        583        761        245        310      722      299
##             treated3
## FBgn0000003        1
## FBgn0000008       70
## FBgn0000014        0
## FBgn0000015        0
## FBgn0000017     3334
## FBgn0000018      308
```

Optional: delete any genes that were never detected (equal to zero in all conditions).

```r
count.table <- count.table[rowSums(count.table) > 0,]
```

Metadata- Columns will be condition or libType. Rows correspond to the 7 (untreated or treated) samples.

```r
pasillaDesign = data.frame(
  row.names = colnames( count.table ), 
  condition = c( "untreated", "untreated", "untreated",
                 "untreated", "treated", "treated", "treated" ), 
  libType = c( "single-end", "single-end", "paired-end",
               "paired-end", "single-end", "paired-end", "paired-end" ) )
```

This dataset contains RNA-Seq counts for RNAi treated vs untreated cells, as well as sequencing data using both single-end sequencing (reading fragments from only one end to the other) and paired-end sequencing (reading form both ends).

```r
pairedSamples = pasillaDesign$libType == "paired-end"
countTable = count.table[ , pairedSamples ]
condition = pasillaDesign$condition[ pairedSamples ]

head(countTable)
```

```
##             untreated3 untreated4 treated2 treated3
## FBgn0000003          0          0        0        1
## FBgn0000008         76         70       88       70
## FBgn0000014          0          0        0        0
## FBgn0000015          1          2        0        0
## FBgn0000017       3564       3150     3072     3334
## FBgn0000018        245        310      299      308
```


```r
cds = newCountDataSet( countTable, condition )
```

Normalize the expression values of each treatment by dividing each column with its own size factor using the estimateSizeFactors function. This estimates the size factor by first taking each column and dividing by the geometric mean of the rows. The median of these ratios is used as the size factor for this column.

```r
cds = estimateSizeFactors( cds )
sizeFactors( cds )
```

```
## untreated3 untreated4   treated2   treated3 
##  0.8730966  1.0106112  1.0224517  1.1145888
```

This divides each column by the size factor, which makes them comparable.

```r
head( counts( cds, normalized=TRUE ) )
```

```
##              untreated3 untreated4   treated2     treated3
## FBgn0000003    0.000000    0.00000    0.00000    0.8971919
## FBgn0000008   87.046493   69.26502   86.06763   62.8034302
## FBgn0000014    0.000000    0.00000    0.00000    0.0000000
## FBgn0000015    1.145349    1.97900    0.00000    0.0000000
## FBgn0000017 4082.022370 3116.92579 3004.54278 2991.2376629
## FBgn0000018  280.610404  306.74508  292.43434  276.3350930
```

With DESeq, differential gene expression is approximated by a negative binomial distribution, for which we need the dispersion parameter. Estimating the dispersion parameter:

```r
cds = estimateDispersions( cds )
```

Plotting dispersion estimates. The level of dispersion is related to the biological dispersion in each treatment.The more variation, the bigger the difference between counts between treatments is required before the differences become significant.

```r
plotDispEsts( cds )
```
![dispersion](https://cloud.githubusercontent.com/assets/16356757/16339197/621b3992-39ef-11e6-89ec-b6eb35f46abe.png)


See whether there is differential expression between untreated and treated. 
We need to correct for the fact that we are doing multiple comparison tests. In the case of gene expression data, it's typical to control for the FDR by using methods like Benjamini-Hochberg, instead of  the Bonferroni correction, which is too conservative and generally would lead to a lot of false negatives.

Output is a data.frame. 


```r
res = nbinomTest( cds, "untreated", "treated" )
head(res)
```

```
##            id     baseMean   baseMeanA    baseMeanB foldChange
## 1 FBgn0000003    0.2242980    0.000000    0.4485959        Inf
## 2 FBgn0000008   76.2956431   78.155755   74.4355310  0.9523999
## 3 FBgn0000014    0.0000000    0.000000    0.0000000        NaN
## 4 FBgn0000015    0.7810873    1.562175    0.0000000  0.0000000
## 5 FBgn0000017 3298.6821506 3599.474078 2997.8902236  0.8328690
## 6 FBgn0000018  289.0312286  293.677741  284.3847165  0.9683564
##   log2FoldChange      pval      padj
## 1            Inf 1.0000000 1.0000000
## 2    -0.07036067 0.8354725 1.0000000
## 3            NaN        NA        NA
## 4           -Inf 0.4160556 1.0000000
## 5    -0.26383857 0.2414208 0.8811746
## 6    -0.04638999 0.7572819 1.0000000
```
padj is the adjusted p-value (controlled for DFR)
To order by pad-j (decreasing):

```r
res <- res[order(res$padj),]
head(res)
```

```
##               id  baseMean baseMeanA  baseMeanB foldChange log2FoldChange
## 8817 FBgn0039155  463.4369  884.9640   41.90977  0.0473576      -4.400260
## 2132 FBgn0025111 1340.2282  311.1697 2369.28680  7.6141316       2.928680
## 570  FBgn0003360 2544.2512 4513.9457  574.55683  0.1272848      -2.973868
## 2889 FBgn0029167 2551.3113 4210.9571  891.66551  0.2117489      -2.239574
## 9234 FBgn0039827  188.5927  357.3299   19.85557  0.0555665      -4.169641
## 6265 FBgn0035085  447.2485  761.1898  133.30718  0.1751300      -2.513502
##               pval          padj
## 8817 1.641210e-124 1.887556e-120
## 2132 3.496915e-107 2.010901e-103
## 570   1.552884e-99  5.953239e-96
## 2889  4.346335e-78  1.249680e-74
## 9234  1.189136e-65  2.735251e-62
## 6265  3.145997e-56  6.030352e-53
```

Plot log2fold changes against mean normalised counts for untreated vs treated.

```r
plotMA(res, col = ifelse(res$padj >=0.01, "black", "violet"))
abline(h=c(-1:1), col="red")
```

![logfoldchange](https://cloud.githubusercontent.com/assets/16356757/16339290/ca4949be-39ef-11e6-8555-36d66d03dea5.png)

Select gene names based on FDR (1%)

```r
gene.kept <- res$id[res$padj <= 0.01 & !is.na(res$padj)]
```

Create a count data set with multiple factors

```r
cdsFull = newCountDataSet(count.table, pasillaDesign)
```

Estimating size factors

```r
cdsFull = estimateSizeFactors( cdsFull )
```

Estimating dispersions

```r
cdsFull = estimateDispersions( cdsFull )
```


Variance stabilizing transformation, which will be useful for certain applications:

```r
cdsBlind = estimateDispersions( cds, method="blind" )
vsd = varianceStabilizingTransformation( cdsBlind )
```

A heatmap of the count table from the variance stabilisation transformed data for the 20 most highly expressed genes:

```r
cdsFullBlind = estimateDispersions( cdsFull, method = "blind" )
vsdFull = varianceStabilizingTransformation( cdsFullBlind )

select = order(rowMeans(counts(cdsFull)), decreasing=TRUE)[1:20]
hmcol = colorRampPalette(brewer.pal(9, "GnBu"))(100)
heatmap.2(exprs(vsdFull)[select,], col = hmcol, trace="none", margin=c(10, 6))
```

![heatmap-count-table](https://cloud.githubusercontent.com/assets/16356757/16339200/64d3e242-39ef-11e6-99fd-1454a738a589.png)

Sample clustering:

```r
distances = dist( t( exprs(vsdFull) ) )

mat = as.matrix( distances )
rownames(mat) = colnames(mat) = with(pData(cdsFullBlind), paste(condition, libType, sep=" : "))
heatmap.2(mat, trace="none", col = rev(hmcol), margin=c(13, 13))
```

![Clustering](https://cloud.githubusercontent.com/assets/16356757/16339205/667839f4-39ef-11e6-9f9d-7f8e5ae4a839.png)

In the case of gene expression data, it's important to verify that the first principal component is your condition- in this case, treated and untreated (and not technical method of sequencing, like paired or single end reads).
PCA plot of the samples:

```r
print(plotPCA(vsdFull, intgroup=c("condition", "libType")))
```

![PCA](https://cloud.githubusercontent.com/assets/16356757/16339207/6839d2ca-39ef-11e6-8b66-dfccffb7c2be.png)




