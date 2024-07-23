library(mlr3verse)
library(Metrics)

set.seed(123)


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
data = removeNAcols(data)
task = as_task_classif(data, target = 'Response')

split = partition(task, ratio = 0.8)
task_train = as_task_classif(data, target = 'Response', row_ids = split$train)
task_test = as_task_classif(data, target = 'Response', row_ids = split$test)


# train a baseline random forest with 100 trees and max depth 5
lrn_baseline = lrn('classif.ranger', num.trees = 100, max.depth = 5)
lrn_baseline$train(task_train)
lrn_baseline$predict(task_test)$score(msr('classif.wkappa'))


# train simple multi-class classifier
learners = list(
  lrn('classif.glmnet', id = 'elnet', alpha = 0.5),
  lrn('classif.kknn', id = 'kknn'),
  #lrn('classif.multinom', id = 'multinom'),  # too many weights
  lrn('classif.nnet', id = 'nnet'),
  lrn('classif.xgboost', id = 'xgboost')
)
learners = lapply(
  X = learners,
  FUN = function(learner) {
    as_learner(
      po('colapply', applicator = as.factor, affect_columns = selector_type('character')) %>>%
      po('encode') %>>%
      learner
    )
  }
)

cv7 = rsmp('cv', folds = 7)
bg_1 = benchmark_grid(
  task = task_train,
  learners = learners,
  resamplings = cv7
)
b_1 = benchmark(bg_1)
b_1$aggregate(msr('classif.wkappa'))