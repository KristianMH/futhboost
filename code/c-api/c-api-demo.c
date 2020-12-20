/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file c-api-demo.c
 * \brief A simple example of using xgboost C API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xgboost/c_api.h>
#include <unistd.h>
#include <sys/time.h>

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

int main(int argc, char** argv) {
  int silent = 0;
  int use_gpu = 1;  // set to 1 to use the GPU for training
  struct timeval start, end;

  DMatrixHandle dtrain;
  /* safe_xgboost(XGDMatrixCreateFromFile("../data/HIGGS_training.csv?format=csv&label_column=0", */
  /*                                      silent, &dtrain)); */
  safe_xgboost(XGDMatrixCreateFromFile("5M.bin", silent, &dtrain));

  // create the booster
  BoosterHandle booster;
  DMatrixHandle eval_dmats[1] = {dtrain};
  safe_xgboost(XGBoosterCreate(eval_dmats, 1, &booster));
  // configure the training
  safe_xgboost(XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "exact"));
  if (use_gpu) {
    // set the GPU to use;
    safe_xgboost(XGBoosterSetParam(booster, "max_bin", "256"));
  } else {
    // avoid evaluating objective and metric on a GPU
    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "-1"));
  }

  safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logitraw"));
  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "0"));
  safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "reg_lambda", "0.5"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "6"));
  safe_xgboost(XGBoosterSetParam(booster, "missing", "-999.0"));
  safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1"));

  // train and evaluate for 10 iterations
  int n_trees = 100;
  const char* eval_names[1] = {"train"};
  const char* eval_result = NULL;
  int i;
  // start timer
  gettimeofday(&start, NULL); 

  for (i = 0; i < n_trees; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 1, &eval_result));
    //printf("%s\n", eval_result);
  }
  gettimeofday(&end, NULL);

  double time_taken = (end.tv_usec - start.tv_usec) + (end.tv_sec - start.tv_sec)*1e6;
  printf("time program took %f us to execute\n", time_taken);


  safe_xgboost(XGBoosterEvalOneIter(booster, i-1, eval_dmats, eval_names, 1, &eval_result));
  printf("%s\n", eval_result);


  // free everything
  safe_xgboost(XGBoosterFree(booster));
  safe_xgboost(XGDMatrixFree(dtrain));
  //safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
