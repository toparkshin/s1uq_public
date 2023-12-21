# Uncertainty Quantification for Calcium Imaging Data

#### This project involves using a transformer model to calculate both epistemic and aleatoric uncertainties from fluorescence intensity data of 2-photon calcium imaging obtained during the vibration frequency discrimination task in the fS1

## Data Preparing for Training

Assuming a total of 1000 trials, 200 cells, and 120 data points per frame for each trial.

    'data': Consists of a list of 1000 elements, each an ndarray of shape (200, 120).
    'stimulus': An ndarray of shape (1000,) representing the type of stimulus (e.g., Go, Nogo, Probe) as integers.
    'response': An ndarray of shape (1000,) indicating the mouse's response (e.g., Hit, Miss, CR, FA) as integers.
    'freq': An ndarray of shape (1000,) containing the frequencies of the vibration stimulus.
    'day_info': An ndarray of shape (1000,) showing the day of the behavioral experiment as integers.
    'trial_info': An ndarray of shape (1000,) indicating the trial number for each date as integers.
    'mouse_info': An ndarray of shape (1000,) representing each mouse as integers.
    'label': An ndarray of shape (1000,) where labels are assigned as 0 or 1 based on the above stimulus or response.

The essential components for learning are 'data' and 'label', while the rest facilitate the analysis by organizing results.

Three pickle files are required: 'prefix_train.pickle', 'prefix_valid.pickle', and 'prefix_test.pickle', following the aforementioned structure.

## Config

	"root_data_dir": "your_data_dir",         # the directory where the pickle file data is stored
	"project_name": "test",                   # the name of the project in wandb
	"run_name": "run_name_stored_in_WandB",   # the name of the run in wandb
	"save_name": "filename_stored_in_the_checkpoints_folder.pth",
                                              # the name of the saved model in the checkpoints folder

	"data_prefix": "your_data_prefix",        # the prefix of the pickle file
	"data_start_idx": 28,                     # the starting dataframe of each trial for training
    "data_end_idx": 37,                       # the ending dataframe of each trial for training
	"n_signal": 9,                            # the number of dataframes for each trial for training
                                              # (data_end_idx - data_start_idx)
	"uq_mode": "combined",                    # the mode of uncertainty quantification: 'combined', 'aleatoric', or 'epistemic'

	"validate": true,                         # whether to validate during training
	"save_at_end": true,                      # whether to save the model at the end of training
	"save_with_val_loss": true,               # whether to save the model with the lowest validation loss

## Training

The training can be executed through 'train.py'. Prepare a config file and load it from the file selection dialog.

If a checkpoints folder does not exist, create a checkpoints folder in this path.

For training, we use wandb, so in  
```python
writer = wandb.init(project=config.project_name, entity="your_wandb_id", name=config.run_name)
```
replace 'entity' with your own wandb ID. For detailed usage instructions, please refer to the following link: https://docs.wandb.ai/quickstart.

As a result, the last trained model is saved in the checkpoints folder as 'save_name.pth' when 'save_at_end' is set to true. When 'save_with_val_loss' is true, each model is saved as 'save_at_end_val_loss_XXX.pth' with the respective val_loss values. You can use the necessary .pth model file for testing.

## Testing

The testing can be executed through 'test.py'. Prepare a config file and load it from the file selection dialog.

    num_mcd = 1000                           # the number of inferences with dropout

In testing, the model named 'save_name' is loaded, so if you want to use the model with the lowest val loss, you need to rename the corresponding .pth file to 'save_name'.

The results are saved as a pickle file, and to calculate the epistemic uncertainty, you can compute the variance later in the 'epistemic_result'.

## Result File

The result file is a pickle file with the following structure:

    "epistemic_result": ndarray(# of inferences, trials)
                                             # the epistemic inferences of the model with dropout
                                             # calculation of variance values for determining epistemic uncertainty
    "aleatoric_result": mu and sigma         # sigma: the aleatoric uncertainty
    'stimulus'
    'response'
    'freq'
    'day_info'
    'trial_info'
    'mouse_info'
    'label'
