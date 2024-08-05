library(mlr3verse)
library(Metrics)
require(tidyr)
require(dplyr)
library(mlr3pipelines)

set.seed(123)
cv5 = rsmp('cv', folds = 5)
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


################################################################################
# pre-processing (1): discard NA cols
data = removeNAcols(data)
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')

split = partition(task, ratio = 0.8)
task_train = as_task_classif(data, target = 'Response', row_ids = split$train)
task_test = as_task_classif(data, target = 'Response', row_ids = split$test)

# approach (1): train simple multi-class classifier on entire task
learners_11 = list(
  lrn('classif.ranger', num.trees = 100, max.depth = 5, id = 'rf'),  # baseline
  lrn('classif.glmnet', id = 'elnet', alpha = 0.5, s = 0.01),
  lrn('classif.kknn', id = 'kknn'),
  lrn('classif.multinom', id = 'multinom', MaxNWts = 2000),
  lrn('classif.nnet', id = 'nnet', MaxNWts = 2000),
  lrn('classif.xgboost', id = 'xgboost')
)

bg_11 = benchmark_grid(
  task = task_train,
  learners = learners_11,
  resamplings = cv7
)
b_11 = benchmark(bg_11)
b_11$aggregate(msr('classif.wkappa'))


# approach (2): perform one-vs-rest classification
learners_12 = lapply(
  X = learners_11,
  FUN = function(learner) {
    l_id = learner$id
    learner = as_learner(pipeline_ovr(learner))
    learner$id = paste0('ovr_', l_id)
    learner
  }
)
bg_12 = benchmark_grid(
  task = task_train,
  learners = learners_12,
  resamplings = cv7
)
b_12 = benchmark(bg_12)
b_12$aggregate(msr('classif.wkappa'))


################################################################################
# pre-processing (2): histogram-impute NA vals
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')

split = partition(task, ratio = 0.8)
task_train = as_task_classif(data, target = 'Response', row_ids = split$train)
task_test = as_task_classif(data, target = 'Response', row_ids = split$test)

# approach (1): train simple multi-class classifier on entire task
learners_21 = lapply(
  X = learners_11,
  FUN = function(learner) {
    l_id = learner$id
    learner = as_learner(
      po('imputehist') %>>%
      learner
    )
    learner$id = paste0('imputehist_', l_id)
    learner
  }
)
bg_21 = benchmark_grid(
  task = task_train,
  learners = learners_21,
  resamplings = cv7
)
b_21 = benchmark(bg_21)
b_21$aggregate(msr('classif.wkappa'))


# approach (2): perform one-vs-rest classification
learners_22 = lapply(
  X = learners_12,
  FUN = function(learner) {
    l_id = learner$id
    learner = as_learner(
      po('imputehist') %>>%
      learner
    )
    learner$id = paste0('imputehist_', l_id)
    learner
  }
)
bg_22 = benchmark_grid(
  task = task_train,
  learners = learners_22,
  resamplings = cv7
)
b_22 = benchmark(bg_22)
b_22$aggregate(msr('classif.wkappa'))

################################################################################
# pre-processing (3): regression task, histogram-impute NA vals
# approach (1): xgboost, metric transforming output to classes
wkappa_func = function(
    prediction,
    cuts = c(unlist(seq(1.5, 7.5, by = 1))),
    labels = seq(1, 8, by = 1)
  ) {
  cuts = c(-Inf, cuts, Inf)
  response = cut(prediction$response, cuts, labels)
  Metrics::ScoreQuadraticWeightedKappa(prediction$truth, response, min(labels), max(labels))
}
MSR.WKAPPA.REGR = R6::R6Class(
  "WKAPPA.REGR",
  inherit = mlr3::MeasureRegr,
  public = list(
    initialize = function() {
      super$initialize(
        id = 'regr.wkappa',
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
      wkappa_func(prediction)
    }
  )
)
mlr3::mlr_measures$add('regr.wkappa', MSR.WKAPPA.REGR)

data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_regr(data, target = 'Response')

split = partition(task)
task_train = as_task_regr(data, target = 'Response', row_ids = split$train)
task_test = as_task_regr(data, target = 'Response', row_ids = split$test)

xgb_regr = lrn(
  'regr.xgboost',
  nrounds = 100,
  max_depth = to_tune(1, 20),
  subsample = to_tune(0.5, 1),
  colsample_bytree = to_tune(0.5, 1),
  min_child_weight = to_tune(10, 1000, logscale = TRUE),  # cf. https://stats.stackexchange.com/a/587618
  eta = to_tune(0.1, 0.5, logscale = TRUE)
)
learner = as_learner(po('imputehist') %>>% xgb_regr, id = 'xgb_regr_imputehist')

at = auto_tuner(
  tnr('random_search'),
  learner,
  cv5,
  msr('regr.wkappa'),
  term_evals = 100
)
at$train(task_train)
at$predict(task_test)$score(msr('regr.wkappa'))

tuned_learner_3 = at$learner

rr = resample(task_train, tuned_learner_3, cv7)
rr$aggregate(msr('regr.wkappa'))

cuts_optim = optim(
  seq(1.5, 7.5, by = 1),
  wkappa_func,
  prediction = rr$prediction(),
  control = list(fnscale = -1)  # maximise function
)
cuts_optim$par

# evaluate with optimised cutoffs
preds = tuned_learner_3$predict(task_test)
wkappa_func(preds, cuts = cuts_optim$par)


################################################################################
################################################################################
################################################################################


# submission
# train models on entire dataset and test on unseen data
write_to_csv = function(dt, filename) {
  dt[, 'Id'] = data_test[, 'Id']
  dt = dt[, Response:=as.integer(response)]
  dt = dt[, c('Id', 'Response')]
  write.csv(
    dt,
    paste0('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/submissions/', filename, '.csv'),
    row.names = FALSE
  )
}

# pre-processing (1)
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = removeNAcols(data)
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')
data_test = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/test.csv')
data_test = removeNAcols(data_test)
data_test = one_hot_encode(data_test, 'Product_Info_2')

res_11 = lapply(
  X = learners_11,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    res = as.data.table(learner$predict_newdata(data_test))
    write_to_csv(res, paste0('res-11-', learner$id))
    res
  }
)

# OVR pipeline needs a response column with >= 2 levels: create dummy response column
data_test['Response'] = round(runif(n = nrow(data_test), min = 1, max = 8), digits = 0)
res_12 = lapply(
  X = learners_12,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    res = as.data.table(learner$predict_newdata(data_test))
    write_to_csv(res, paste0('res-12-', learner$id))
    res
  }
)


################################################################################
# pre-processing (2)
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_classif(data, target = 'Response')
data_test = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/test.csv')
data_test = one_hot_encode(data_test, 'Product_Info_2')

res_21 = lapply(
  X = learners_21,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    res = as.data.table(learner$predict_newdata(data_test))
    write_to_csv(res, paste0('res-21-', learner$id))
    res
  }
)

# OVR pipeline needs a response column with >= 2 levels: create dummy response column
data_test['Response'] = round(runif(n = nrow(data_test), min = 1, max = 8), digits = 0)
res_22 = lapply(
  X = learners_22,
  FUN = function(learner) {
    print(learner$id)
    learner$train(task)
    res = as.data.table(learner$predict_newdata(data_test))
    write_to_csv(res, paste0('res-22-', learner$id))
    res
  }
)


################################################################################
# pre-processing (3)
data = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/train.csv')
data = one_hot_encode(data, 'Product_Info_2')
task = as_task_regr(data, target = 'Response')
data_test = read.csv('C:/workplace/uni/ss-24/applied-ml/kaggle competition/data/test.csv')
data_test = one_hot_encode(data_test, 'Product_Info_2')

res = as.data.table(tuned_learner_3$predict_newdata(data_test))
res_table = as.data.table(res)
res_table$response = cut(res$response, c(-Inf, cuts_optim$par, Inf), seq(1, 8, by = 1))
res_table
write_to_csv(res_table, 'res-3-optim-xgb')
