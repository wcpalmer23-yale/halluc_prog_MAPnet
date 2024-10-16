# Load packages
library(tidyverse)
library(splines)
library(devtools)
install_github("alessandromagrini/gbmt")
library(gbtm)

# Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
clin_dir = paste0(proj_dir, "/clinic")
result_dir = paste0(proj_dir, "/results")

# Load csv
df = read_csv(paste0(clin_dir, "/SOPS_P4_collected.csv"))

# Filter out time points
df_tmp = filter(df, VisitLabel != "C")
df_tmp = filter(df_tmp, VisitLabel != "12")
df_tmp = filter(df_tmp, VisitLabel != "18")
df_tmp = filter(df_tmp, VisitLabel != "24")
df_tmp = filter(df_tmp, VisitLabel != "OS1")
df_tmp = filter(df_tmp, VisitLabel != "OS2")
df_tmp = filter(df_tmp, VisitLabel != "F30")
df_tmp = filter(df_tmp, VisitLabel != "F36")

# Restrict number of time points
tp = group_by(df_tmp, SubjectID) %>% summarize(Count=n())
tp = filter(tp, Count>=3)

# Isolate participants with enough time points
df_red = df_tmp[df_tmp$SubjectID %in% tp$SubjectID,]
df_red = arrange(df_red, SubjectID, bl_time)

# Recount time points
tp = group_by(df_red, SubjectID) %>% summarize(Count=n())

# Check baseline scans
df_baseline = df_red[df_red$bl_time == 0,] 
df_baseline[df_baseline$VisitLabel != "BL",]

# Extract regressors
df_reg = df_red[, c(1,4,6:7)]

write_csv(df_reg, paste0(result_dir, "/gbtm_reduced_P4.csv"))

# Apply spline
tmp = bs(df_reg$bl_time, df=4)


