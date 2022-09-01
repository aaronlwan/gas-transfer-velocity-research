# Application of EQL-DIV to creating equations for gas transfer velocity rate

- Majority of model code cloned from https://github.com/martius-lab/EQL

## Changes:

### Changed files:
- src/mlfg_file.py: added tracking of the best version of model so we can use the best model when creating equations

### Files created:
- datasets/: stores observed.csv and pickle file from save_data.py
- save_data.py: creates all input features and saves data as pickle file
- src/equation_from_model.py: contains functions to evaluate model + generate the learned equation from the model's weights and biases
- src/equations_for_paper.py: script to save generate and evaluate equations for Kl from all combinations of input features

## To replicate results: 
- run $python src/equations_for_paper.py 
