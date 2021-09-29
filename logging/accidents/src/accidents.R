library(optparse)
library(mlflow)
library(carrier)
library(azureml.mlflow)

options <- list(
  make_option(c("-d", "--data_folder"), default="./data")
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste(opt$data_folder)

# Enable mlflow logging to Azure by setting tracking uri to correct AzureML tracking uri
mlflow_set_tracking_uri(mlflow_get_azureml_tracking_uri())

# Test dummy experiment tag
mlflow_set_tag("test", "tagging" , run_id=Sys.getenv("AZUREML_RUN_ID"))

accidents <- readRDS(file.path(opt$data_folder, "accidents.Rd"))
summary(accidents)

# Log dummy parameters
print("Logging dummy parameters to AzureML experiment using MLflow")
mlflow_log_param("alpha", 0.5, run_id=Sys.getenv("AZUREML_RUN_ID"))
mlflow_log_param("lambda", 0.5, run_id=Sys.getenv("AZUREML_RUN_ID"))

mod <- glm(dead ~ dvcat + seatbelt + frontal + sex + ageOFocc + yearVeh + airbag  + occRole, family=binomial, data=accidents)
summary(mod)
predictions <- factor(ifelse(predict(mod)>0.1, "dead","alive"))
accuracy <- mean(predictions == accidents$dead)

# Log accuracy to AzureML experiment
print("Logging accuracy metric to AzureML experiment using MLflow")
mlflow_log_metric(key="Accuracy", value=accuracy, run_id=Sys.getenv("AZUREML_RUN_ID"))

output_dir = "outputs"
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}
saveRDS(mod, file = "./outputs/model.rds")
message("Model saved")

# Log model to experiment using MLflow
predictor <- crate(~ predict.glm(mod, as.matrix(.x)), mod)
#mlflow_log_model(predictor, "accident_model", run_id=Sys.getenv("AZUREML_RUN_ID"))