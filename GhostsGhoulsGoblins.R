library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(bonsai)
library(lightgbm)
library(discrim)
library(themis)

###missingtrain <- vroom("trainWithMissingValues.csv")
train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type ~ . , data=missingtrain) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul), neighbors = 7) %>% 
  step_impute_knn(rotting_flesh, impute_with = imp_vars(bone_length, hair_length, has_soul), neighbors = 7) %>% 
  step_impute_knn(hair_length, impute_with = imp_vars(rotting_flesh, bone_length, has_soul), neighbors = 7) 

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = missingtrain)

rmse_vec(train[is.na(missingtrain)],
         baked[is.na(missingtrain)])

#### NEURAL NETWORK
nn_recipe <- recipe(type ~ . , data=train) %>%
update_role(id, new_role="id") %>%
 # step_mutate(color = as.factor(color)) %>% # Convert 'color' to a factor
 # step_dummy(color) %>%  # Dummy encode 'color'
##step_...() %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

prep <- prep(nn_recipe)
baked <- bake(prep, new_data = train)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 250
) %>%
set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)
  
nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=3)
                            
folds_nn <- vfold_cv(train, v = 3, repeats=3)

cv_results <- nn_wf %>%
              tune_grid(resamples = folds_nn,
              grid = nn_tuneGrid, 
              metrics = metric_set(accuracy))

cv_results %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results

bestTune <- cv_results %>%
  select_best()

final_wf <-
  nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf %>%
  predict(new_data = train, type = "class")

GGG_pred <- predict(final_wf,
                         new_data = test,
                         type = "class")

kaggle_submission <- GGG_pred %>% 
  bind_cols(., test) %>% 
  rename(type = .pred_class) %>% 
  select(id, type)

vroom_write(x = kaggle_submission, file = "./NNPreds.csv" , delim = ",")

#### BOOSTED MODEL

boosted_recipe <- recipe(type ~ . , data=train) %>%
  update_role(id, new_role="id") %>%
  # step_mutate(color = as.factor(color)) %>% # Convert 'color' to a factor
  # step_dummy(color) %>%  # Dummy encode 'color'
  ##step_...() %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

prep <- prep(boosted_recipe)
baked <- bake(prep, new_data = train)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boosted_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boost_model)

boosted_tuneGrid <- grid_regular(tree_depth(),
                                 trees(),
                                 learn_rate(),
                                 levels = 3)

folds_boost <- vfold_cv(train, v = 5, repeats=1)

cv_results <- boosted_wf %>%
  tune_grid(resamples = folds_boost,
            grid = boosted_tuneGrid, 
            metrics = metric_set(accuracy))

## CV tune, finalize and predict here and save results

bestTune <- cv_results %>%
  select_best()

final_wf <-
  boosted_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf %>%
  predict(new_data = train, type = "class")

GGG_pred <- predict(final_wf,
                    new_data = test,
                    type = "class")

kaggle_submission <- GGG_pred %>% 
  bind_cols(., test) %>% 
  rename(type = .pred_class) %>% 
  select(id, type)

vroom_write(x = kaggle_submission, file = "./BoostedPreds.csv" , delim = ",")

### NAIVE BAYES ####

bayes_recipe <- recipe(type ~ . , data=train) %>%
  step_mutate(id, features = id) %>% 
 # update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color))# %>% # Convert 'color' to a factor
 # step_dummy(color) %>%  # Dummy encode 'color'
 # step_range(all_numeric_predictors(), min=0, max=1) %>%     #scale to [0,1]
#  step_normalize(all_predictors())# %>%
#  step_pca(all_predictors(), threshold=0.9) %>%
 # step_other(all_nominal_predictors(), threshold = 0.001) %>% 
 # step_lencode_glm(all_nominal_predictors(), outcome = vars(type))# %>% 
 # step_smote(all_outcomes(), neighbors=5) %>% 
 # step_downsample()


prep <- prep(bayes_recipe)
baked <- bake(prep, new_data = train)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(bayes_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(smoothness(),
                            Laplace(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 7, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>%
  select_best(metric = "accuracy")

final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf %>%
  predict(new_data = train, type = "class")

GGG_pred <- predict(final_wf,
                    new_data = test,
                    type = "class")

kaggle_submission <- GGG_pred %>% 
  bind_cols(., test) %>% 
  rename(type = .pred_class) %>% 
  select(id, type)

vroom_write(x = kaggle_submission, file = "./BayesPreds.csv" , delim = ",")

