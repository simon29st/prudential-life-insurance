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

# PO.REGR_TRAFO = R6::R6Class(
#   'REGR_TRAFO',
#   inherit = mlr3pipelines::PipeOp,
#   public = list(
#     initialize = function() {
#       super$initialize(
#         id = 'po.regr_trafo',
#         label = 'Transform regression outputs to multi-class'
#       )
#     }
#   ),
#   private = list(
#     .train = function(inputs) {
#       inputs[inputs <= 1.5] = 1
#       inputs[inputs > 1.5]
#     },
#     .predict = function(inputs) {}
#   )
# )


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

split = partition(task, ratio = 0.6)
task_train = as_task_classif(data, target = 'Response', row_ids = split$train)
task_val_test = as_task_classif(data, target = 'Response', row_ids = split$test)
split = partition(task_val_test, ratio = 0.5)
task_val = as_task_classif(data, target = 'Response', row_ids = split$train)
task_test = as_task_classif(data, target = 'Response', row_ids = split$test)


# train a baseline random forest with 100 trees and max depth 5
lrn_baseline = lrn('classif.ranger', num.trees = 100, max.depth = 5)
lrn_baseline$train(task_train)
lrn_baseline$predict(task_test)$score(msr('classif.wkappa'))


# approach (1), train a simple multi-class classifier
learners_1 = list()


# approach (2), train a regression model, then transform value to be categorical
learners_2 = list(
  lrn('regr.glmnet', id = '2-elnet', alpha = 0.5),
  lrn('regr.kknn', id = '2-kknn'),
  lrn('regr.lm', id = '2-lm'),
  lrn('regr.nnet', id = '2-nnet'),
  lrn('regr.xgboost', id = '2-xgboost')
)