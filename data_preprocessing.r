library(dplyr)
library(tidymodels)
library(tidyverse)
library(xgboost)
library(vip)         # For variable importance
library(doParallel)  # For parallel processing

### Download Training Data

tmp <- tempfile()
download.file("https://luminwin.github.io/ASASF/train.rds", tmp, mode = "wb")
# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------

# Register parallel backend to speed up training
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

# Load Data
# Replace with the actual path to your file
data_raw <- readRDS(tmp)

# ---------------------------------------------------------
# 2. DATA CLEANING & PRE-PROCESSING
# ---------------------------------------------------------
df_clean <- data_raw %>%
  # Convert Outcome to Factor (Critical for Classification)
  mutate(LBDHDD_outcome = as.factor(LBDHDD_outcome)) %>%
  
  # Convert key Categorical features to factors
  mutate(
    RIAGENDR = as.factor(RIAGENDR),
    RIDRETH3 = as.factor(RIDRETH3),
    DMDMARTZ = as.factor(DMDMARTZ),
    ALQ111   = as.factor(ALQ111), # Ever had a drink?
    ALQ121   = as.factor(ALQ121)  # Frequency of drinking
  ) %>%
  
  # Handle specific NHANES missing codes (often 7, 9, 77, 99 mean Refused/Don't Know)
  # NOTE: Check your specific data dictionary, but this is standard practice.
  mutate(across(where(is.numeric), ~ifelse(. %in% c(77, 99, 777, 999), NA, .)))

# Split Data (80% Train, 20% Test)
set.seed(123)
data_split <- initial_split(df_clean, prop = 0.80, strata = LBDHDD_outcome)
train_set  <- training(data_split)
test_set   <- testing(data_split)

# ---------------------------------------------------------
# 3. ADVANCED FEATURE ENGINEERING (The "Recipe")
# ---------------------------------------------------------
# ---------------------------------------------------------
# CORRECTED RECIPE
# ---------------------------------------------------------
recipe_spec <- recipe(LBDHDD_outcome ~ ., data = train_set) %>%
  
  # 1. REMOVE ID & IRRELEVANT COLUMNS
  step_rm(starts_with("WT"), contains("SEQN")) %>% 
  
  # 2. IMPUTATION (Fill missing values)
  step_impute_knn(all_numeric_predictors(), neighbors = 5) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  # 3. *** CRITICAL FIX: REMOVE CONSTANT COLUMNS FIRST ***
  # This deletes ALQ111 before it causes the crash
  step_zv(all_predictors()) %>% 
  
  # 4. FEATURE ENGINEERING
  step_mutate(
    Fat_Quality_Index = (DR1TMFAT + DR1TPFAT) / (DR1TSFAT + 0.1), 
    Carb_Fiber_Ratio = DR1TCARB / (DR1TFIBE + 0.1),
    Na_K_Ratio = DR1TSODI / (DR1TPOTA + 1),
    Caloric_Load = DR1TKCAL / (BMXBMI + 1),
    Waist_Age_Interaction = BMXWAIST * RIDAGEYR
  ) %>%
  
  # 5. TRANSFORMATIONS
  step_log(DR1TALCO, DR1TSUGR, DR1TCAFF, offset = 1) %>%
  
  # 6. NOW IT IS SAFE TO RUN DUMMY ENCODING
  step_dummy(all_nominal_predictors()) %>%
  
  # 7. Final Normalize
  step_normalize(all_numeric_predictors())

# ---------------------------------------------------------
# 4. MODEL SPECIFICATION (XGBoost) - CORRECTED
# ---------------------------------------------------------
xgb_spec <- boost_tree(
  trees = 1000,              
  tree_depth = 6,            
  min_n = 10,                
  loss_reduction = 0.001,    
  sample_size = 0.8,         
  learn_rate = 0.01
  # DELETED: mtry = 0.8 (This was causing the error)
) %>%
  set_engine("xgboost", 
             colsample_bynode = 0.8) %>% # Moved here! This allows ratios.
  set_mode("classification")

# ---------------------------------------------------------
# 5. WORKFLOW & TRAINING (Re-run this part too)
# ---------------------------------------------------------
xgb_workflow <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(xgb_spec)

print("Training Model... (This may take a minute)")
xgb_fit <- xgb_workflow %>%
  fit(data = train_set)

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
# Make predictions on test set
results <- augment(xgb_fit, test_set)

# 1. Confusion Matrix
print(conf_mat(results, truth = LBDHDD_outcome, estimate = .pred_class))

# 2. Accuracy & AUC
metrics_result <- results %>%
  metrics(truth = LBDHDD_outcome, estimate = .pred_class, .pred_1) # Assuming '1' is the class of interest
print(metrics_result)

# 3. ROC Curve
results %>%
  roc_curve(truth = LBDHDD_outcome, .pred_1) %>% # Change .pred_1 to .pred_YES or whatever your class name is
  autoplot()

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE (What drove the prediction?)
# ---------------------------------------------------------
# Extract the underlying XGBoost model and plot importance
xgb_fit %>%
  extract_fit_parsnip() %>%
  vip(geom = "point", num_features = 15) +
  labs(title = "Top 15 Predictors of HDL Outcome")