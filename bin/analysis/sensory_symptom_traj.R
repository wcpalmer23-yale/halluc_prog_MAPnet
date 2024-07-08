library("dplyr")
library("readr")
library("ggplot2")
library("lme4")
library("sjPlot")

# Set directories
data_dir = "/media/wcp27/Data/NAPLS3"
sops_dir = paste0(data_dir, "/SOPS")

# Load csvs
df_group = read_csv(paste0(data_dir, "/Group_Assignment_210115.csv"))
colnames(df_group) = c("SubjectID", "Group")

# Load files
f_sops = list.files(sops_dir)

# Gather info
df_sops4 = data.frame()
for (i in 1:length(f_sops)) {
  print(f_sops[i])
  
  # Load csv
  df_tmp = read_csv(paste0(sops_dir, "/", f_sops[i]))
  
  # Filter data
  df_tmp = df_tmp %>% 
    select("SiteNumber", "SubjectNumber", "VisitLabel", "DataCollectedDate", "DataQuality", 
           "P4_SOPS", "P4_OnsetDateCode", "P4_Onset", "P4_DateOfIncrease") %>%
    filter(DataQuality > 0)
  # Store data
  df_sops4 = rbind(df_sops4, df_tmp)
}

# Generate SubjectID
df_sops4$SubjectID = paste0('0', df_sops4$SiteNumber, '_S0', df_sops4$SubjectNumber)
df_sops4 = df_sops4[,c(10, 3:9)]

# Translate to dates
df_sops4$DataCollectedDate = as.Date(df_sops4$DataCollectedDate, format="%d-%b-%y")
df_sops4$P4_Onset = as.Date(df_sops4$P4_Onset, format="%d-%b-%y")

# Extract unique subjects
df = data.frame()
subs = unique(df_sops4$SubjectID)
for (sub in subs) {
  print(sub)
  
  # Filter by subject
  df_tmp = filter(df_sops4, SubjectID == sub)
  
  # Extract onset
  onset = df_tmp$P4_Onset[!is.na(df_tmp$P4_Onset)][1]
  if (!is.na(onset)) {
    # Set onset
    df_tmp$P4_Onset = onset
    
    # Baseline date
    bl_date = df_tmp[df_tmp$VisitLabel == "BL",]$DataCollectedDate
    
    # Calculate time
    df_tmp$onset_time = as.numeric(difftime(df_tmp$DataCollectedDate, df_tmp$P4_Onset, units="days"))/30.4167
    df_tmp$bl_time = as.numeric(difftime(df_tmp$DataCollectedDate, bl_date, units="days"))/30.4167
    
    # Store
    if (dim(df_tmp)[1] > 1) {
      df = rbind(df, df_tmp)
    }
  }
}

# Add group labels
df = merge(df, df_group, by = "SubjectID")

# Remove NA
df = df[!is.na(df$P4_SOPS),]
df = df[!is.na(df$bl_time),]

# Order VisitLabels
df$VisitLabel = factor(df$VisitLabel, levels = c("BL", "M2", "M4", "M6", "M8", "12", "18", "24", "F30", "F36", "F40", "F46", "OS1", "OS2", "C")) 
df$Group = factor(df$Group, levels = c("Converter", "Non-Con,Non-Remit", "Non-Con,Remit"))

# Isolate Conversion scans
df$Converted = df$VisitLabel == "C"
df[is.na(df$Converted),]$Converted = FALSE

# Plot data
# qplot(bl_time, P4_SOPS, data = df[c(1:250),], color = SubjectID, group = SubjectID)+
#   geom_line()+
#   geom_point()+
#   facet_wrap(~SubjectID)

qplot(bl_time, P4_SOPS, data = filter(df, Group == "Converter"), color=Converted, group = SubjectID)+
  geom_line()+
  geom_point()+
  geom_hline(yintercept = 2.5, linetype = "dashed")+
  geom_hline(yintercept = 5.5, linetype = "dashed")+
  xlab("Time (months)") + ylab("SOPS P4")+
  facet_wrap(~SubjectID, ncol = 7)+
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )

qplot(bl_time, P4_SOPS, data = filter(df, Group == "Non-Con,Remit"), color=Converted, group = SubjectID)+
  geom_line()+
  geom_point()+
  geom_hline(yintercept = 2.5, linetype = "dashed")+
  geom_hline(yintercept = 5.5, linetype = "dashed")+
  xlab("Time (months)") + ylab("SOPS P4")+
  facet_wrap(~SubjectID, ncol=10)+
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )

qplot(bl_time, P4_SOPS, data = filter(df, Group == "Non-Con,Non-Remit"), color=Converted, group = SubjectID)+
  geom_line()+
  geom_point()+
  geom_hline(yintercept = 2.5, linetype = "dashed")+
  geom_hline(yintercept = 5.5, linetype = "dashed")+
  xlab("Time (months)") + ylab("SOPS P4")+
  facet_wrap(~SubjectID, ncol=13)+
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )

# LME model
model = lmer(P4_SOPS ~ bl_time*Group + (1|SubjectID), data=df)
stats = anova(model)
plot_model(model, type = "int") + labs(x='Time (months)', y='SOPS P4')
