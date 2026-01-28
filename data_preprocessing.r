library(dplyr)
### Download Training Data

tmp <- tempfile()
download.file("https://luminwin.github.io/ASASF/train.rds", tmp, mode = "wb")
train <- as.data.frame(readRDS(tmp))

### View Variable Labels

lapply(train, attr, "label")

### Download Test Data

download.file("https://luminwin.github.io/ASASF/test.rds", tmp, mode = "wb")
test <- as.data.frame(readRDS(tmp))
class(train)
