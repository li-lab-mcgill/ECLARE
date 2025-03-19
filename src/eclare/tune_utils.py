from optuna.trial import TrialState
from copy import deepcopy

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


def Optuna_propose_hyperparameters(trial, suggested_hyperparameters):
    """
    Objective function for Optuna hyperparameter tuning.
    Uses run_spatial_CLIP with parameters proposed by Optuna.
    """
    # Suggest hyperparameters
    tuned_hyperparameters = {
        param_name: trial._suggest(param_name, param_info['suggest_distribution'])
        for param_name, param_info in suggested_hyperparameters.items()
    }

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