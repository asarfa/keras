import pandas as pd
import json


def analyse_trials_opt(max_trials: int, directory: str, project_name: str, sort_by: str = 'val_sharpe', ascending: str = False):
    """
    Analysing the tuning, finding the hyper-parameters leading to the maximal performance on validation set
    """
    list_trials = []
    len_max = len(str(max_trials))
    for num_trial in range(max_trials):
        len_diff = len_max - len(str(num_trial))
        if len_diff != 0:
            str_num_trial = '0' * len_diff + str(num_trial)
        else:
            str_num_trial = str(num_trial)
        try:
            with open(directory + '/' + project_name + '/trial_' + str_num_trial + '/trial.json') as f:
                trial = json.load(f)
            found = True
        except FileNotFoundError:
            found = False
        if found:
            try:
                trial_hp = pd.DataFrame(trial['hyperparameters']['values'].values(), index=trial['hyperparameters']['values'].keys()).T
                trial_hp['val_sharpe'] = trial['metrics']['metrics']['val_sharpe']['observations'][0]['value'][0]
                trial_hp['val_mean_wacc'] = trial['metrics']['metrics']['val_mean_wacc']['observations'][0]['value'][0]
                #trial_hp['val_mean_acc'] = trial['metrics']['metrics']['val_mean_acc']['observations'][0]['value'][0]
                list_trials.append(trial_hp)
            except:
                pass
    trials = pd.concat(list_trials, ignore_index=True)
    trials = trials.sort_values(by=sort_by, ascending=ascending)
    print(100 * '-')
    print(f'Trials configuration maximizing the {sort_by} of the cross validation: ')
    print(100 * '-')
    print(trials)
    return trials


def select_hp(trials: pd.DataFrame, config_hp: list, params, num: int = 0):
    trial = trials.iloc[num]
    for hp, value in trial.to_dict().items():
        if hp in config_hp:
            value = int(value) if value - round(value) == 0 else value
            params.__dict__[hp] = value
    return params