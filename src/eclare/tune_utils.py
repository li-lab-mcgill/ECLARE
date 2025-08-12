import optuna
from optuna.trial import TrialState
import mlflow

from eclare.models import get_clip_hparams 
from eclare.run_utils import run_CLIP, run_ECLARE

def study_summary(study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def Optuna_propose_hyperparameters(trial, suggested_hyperparameters, override_with_default=['teacher_temperature']):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Suggest hyperparameters
    tuned_hyperparameters = {
        param_name: trial._suggest(param_name, param_info['suggest_distribution'])
        for param_name, param_info in suggested_hyperparameters.items()
    }

    for param_name in override_with_default:
        tuned_hyperparameters[param_name] = suggested_hyperparameters[param_name]['default']

    #assert set(tuned_hyperparameters.keys()) in set(default_hyperparameters_keys), \
    #    "Tuned hyperparameters keys must match default hyperparameters keys"
    
    return tuned_hyperparameters


def champion_callback(study, frozen_trial):
  """
  Logging callback that will report when a new trial iteration improves upon existing
  best trial values.

  Note: This callback is not intended for use in distributed computing systems such as Spark
  or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
  workers or agents.
  The race conditions with file system state management for distributed trials will render
  inconsistent values with this callback.
  """

  winner = study.user_attrs.get("winner", None)

  if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def tune_CLIP(args, experiment_id, run_args):

    suggested_hyperparameters = get_clip_hparams(context='teacher')

    def run_CLIP_wrapper(trial, run_args):
        with mlflow.start_run(experiment_id=experiment_id, run_name=f'Trial {trial.number}', nested=True):

            params = Optuna_propose_hyperparameters(trial, suggested_hyperparameters=suggested_hyperparameters)
            run_args['trial'] = trial

            mlflow.log_params(params)
            _, metric_to_optimize = run_CLIP(**run_args, params=params)

            return metric_to_optimize

    ## create study and run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            consider_prior=False,  # not recommended when sampling from categorical variables
            n_startup_trials=0,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Don't prune until this many trials have completed
            n_warmup_steps=20,   # Don't prune until this many steps in each trial
            interval_steps=1,     # Check for pruning every this many steps
        )
    )
    Optuna_objective = lambda trial: run_CLIP_wrapper(trial, run_args)
    study.optimize(Optuna_objective, n_trials=args.n_trials, callbacks=[champion_callback])

    ## log best trial
    mlflow.log_params(study.best_params)
    mlflow.log_metrics({f"best_{args.metric_to_optimize}": study.best_trial.value})

    ## log metadata
    mlflow.set_tags(tags={
        'suggested_hyperparameters': suggested_hyperparameters
    })

    return study.best_params
    

def tune_ECLARE(args, experiment_id, run_args, device):
    suggested_hyperparameters = get_clip_hparams(context='student')

    def run_CLIP_wrapper(trial, run_args):
        with mlflow.start_run(experiment_id=experiment_id, run_name=f'Trial {trial.number}', nested=True):

            params = Optuna_propose_hyperparameters(trial, suggested_hyperparameters=suggested_hyperparameters)
            run_args['trial'] = trial

            mlflow.log_params(params)
            _, metrics = run_ECLARE(**run_args, params=params, device=device)
            metric_to_optimize = metrics[args.metric_to_optimize]

            return metric_to_optimize

    ## create study and run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            consider_prior=False,  # not recommended when sampling from categorical variables
            n_startup_trials=0,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Don't prune until this many trials have completed
            n_warmup_steps=20,   # Don't prune until this many steps in each trial
            interval_steps=1,     # Check for pruning every this many steps
        )
    )
    Optuna_objective = lambda trial: run_CLIP_wrapper(trial, run_args)
    study.optimize(Optuna_objective, n_trials=args.n_trials, callbacks=[champion_callback])

    ## log best trial
    mlflow.log_params(study.best_params)
    mlflow.log_metrics({f"best_{args.metric_to_optimize}": study.best_trial.value})

    ## log metadata
    mlflow.set_tags(tags={
        'suggested_hyperparameters': suggested_hyperparameters
    })

    return study.best_params