library(mlr3verse)
library(Metrics)
require(tidyr)
require(dplyr)
library(mlr3pipelines)

set.seed(123)
cv7 = rsmp('cv', folds = 7)


# implement Weighted Kappa as mlr3 measure
# cf. https://stackoverflow.com/a/61783363 + https://mlr3.mlr-org.com/reference/Measure.html#method-new-
MSR.WKAPPA = R6::R6Class(
  "WKAPPA",
  inherit = mlr3::MeasureClassif,
  public = list(
    initialize = function() {
      super$initialize(
        id = 'classif.wkappa',
        packages = c('Metrics'),
        predict_type = 'response',
        range = c(-1, 1),
        minimize = FALSE,
        label = 'ScoreQuadraticWeightedKappa from Metrics package'
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      Metrics::ScoreQuadraticWeightedKappa(prediction$truth, prediction$response, 1, 8)
    }
  )
)
mlr3::mlr_measures$add('classif.wkappa', MSR.WKAPPA)


data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
# remove cols with many NA vals
removeNAcols = function(data) {
  # no. of NAs refers to train.csv, test.csv has NAs in same columns
  data[,c(
    'Employment_Info_1',  # 19 NAs
    'Employment_Info_4',  # 6779 NAs
    'Employment_Info_6',  # 10854 NAs
    'Insurance_History_5',  # 25396 NAs
    'Family_Hist_2',  # 28656 NAs
    'Family_Hist_3',  # 34241 NAs
    'Family_Hist_4',  # 19184 NAs
    'Family_Hist_5',  # 41811 NAs
    'Medical_History_1',  # 8889 NAs
    'Medical_History_10',  # 58824 NAs
    'Medical_History_15',  # 44596 NAs
    'Medical_History_24',  # 55580 NAs
    'Medical_History_32'  # 58274 NAs
  )] = NULL
  data
}
one_hot_encode = function(data, col) {
  # one-hot-encode Product_Info_2 column (encoding via 'colapply' + 'encode' pipeops much slower)
  data %>% mutate(value = 1) %>% spread(col, value, fill = 0)  # cf. https://stackoverflow.com/a/52540145
}
data = removeNAcols(data)
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')

split = partition(task, ratio = 0.8)
task_train = as_task_classif(data, target = 'Response', row_ids = split$train)
task_test = as_task_classif(data, target = 'Response', row_ids = split$test)


# train a baseline random forest with 100 trees and max depth 5
lrn_baseline = lrn('classif.ranger', num.trees = 100, max.depth = 5, id = 'rf')
rr_baseline = resample(task_train, lrn_baseline, cv7)
rr_baseline$aggregate(msr('classif.wkappa'))


# approach (1): train simple multi-class classifier on entire task
learners_1 = list(
  lrn('classif.glmnet', id = 'elnet', alpha = 0.5, s = 0.01),
  lrn('classif.kknn', id = 'kknn'),
  lrn('classif.multinom', id = 'multinom', MaxNWts = 2000),
  lrn('classif.nnet', id = 'nnet', MaxNWts = 2000),
  lrn('classif.xgboost', id = 'xgboost')
)

bg_1 = benchmark_grid(
  task = task_train,
  learners = learners_1,
  resamplings = cv7
)
b_1 = benchmark(bg_1)
b_1$aggregate(msr('classif.wkappa'))


# approach (2): perform one-vs-rest classification
learners_2 = lapply(
  X = learners_1,
  FUN = function(learner) {
    pipeline_ovr(learner)
  }
)
bg_2 = benchmark_grid(
  task = task_train,
  learners = learners_2,
  resamplings = cv7
)
b_2 = benchmark(bg_2)
b_2$aggregate(msr('classif.wkappa'))


# approach (3): histogram-impute previously removed features + re-train both sets of learners
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')

split = partition(task, ratio = 0.8)
task_train = as_task_classif(data, target = 'Response', row_ids = split$train)
task_test = as_task_classif(data, target = 'Response', row_ids = split$test)

learners_3 = c(list(lrn_baseline), learners_1, learners_2)
learners_3 = lapply(
  X = learners_3,
  FUN = function(learner) {
    as_learner(
      po('imputehist') %>>%
      learner
    )
  }
)
bg_3 = benchmark_grid(
  task = task_train,
  learners = learners_3,
  resamplings = cv7
)
b_3 = benchmark(bg_3)
b_3$aggregate(msr('classif.wkappa'))


################################################################################
################################################################################
################################################################################


# evaluation
# train models on entire dataset and test on unseen data
write_to_csv = function(dt, filename) {
  dt[, 'Id'] = data_test[, 'Id']
  dt = dt[, Response:=as.integer(response)]
  dt[, 'truth'] = NULL
  dt[, 'row_ids'] = NULL
  dt[, 'response'] = NULL
  write.csv(
    dt,
    paste0('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/submissions/', filename, '.csv'),
    row.names = FALSE
  )
}

# approaches (1), (2)
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = removeNAcols(data)
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')
data_test = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/test.csv')
data_test = removeNAcols(data_test)
data_test = one_hot_encode(data_test, 'Product_Info_2')

lrn_baseline$train(task)
res_baseline = as.data.table(lrn_baseline$predict_newdata(data_test))
write_to_csv(res_baseline, 'baseline-1')

res_1 = lapply(
  X = learners_1,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    as.data.table(learner$predict_newdata(data_test))
  }
)
apply(
  matrix(c(res_1, learners_1), ncol = 2),
  MARGIN = 1,
  FUN = function(row) {
    write_to_csv(row[[1]], paste0(row[[2]]$id, '-1'))
  }
)

res_2 = lapply(
  X = learners_2,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    as.data.table(learner$predict_newdata(data_test))
  }
)
apply(
  matrix(c(res_2, learners_2), ncol = 2),
  MARGIN = 1,
  FUN = function(row) {
    write_to_csv(row[[1]], paste0(row[[2]]$id, '-2'))
  }
)


# approach (3)
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')
data_test = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/test.csv')
data_test = one_hot_encode(data_test, 'Product_Info_2')
res_3 = lapply(
  X = learners_3,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    res = as.data.table(learner$predict_newdata(data_test))
    write_to_csv(res, paste0(learner$id, '-3'))
    res
  }
)
