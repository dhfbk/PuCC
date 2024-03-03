# "Putting Context in Context: the Impact of Discussion Structure on Text Classification", EACL2024

Repository for the experiments presented in "[Putting Context in Context: the Impact of Discussion Structure on Text Classification](https://arxiv.org/abs/2402.02975)", by Penzo N., Longa A., Lepri B., Tonelli S., Guerini M., accepted at the Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2024.

In this repo we provide two subfolders:

1. "contextual": it contains the code for experiments on the CONTEXTUAL MODELS;
2. "text_only_baselines": it contains the code for experiments on the BASELINES models.

The file "requirements.txt" contains the libraries needed to run the distributed code.

Each folder has the code to train the models and perform the test runs, with already fixed hyperparameters and for ten different random seeds, and training/validation/test sample set.


If you are interested in having the data only for research purposes, please write an contact npenzo@fbk.eu. We will proceed in contacting the owners of the dataset and send it to you, with anonymised data. We are evaluating more automatic releasing of the dataset in compliance with Kialo's terms of service.

The creation of the input is done in the "dataset_experiments.py" script, and the model in the "model_conversations.py". "model_loop.py" contains the code to manage training loop, evaluation loop and test time (with possibility to extract the probabilities etc.).

Notice that in "dataset_experiment.py" we don't assume to have already anonymised data. In this way, the reader can see how the local IDs are assigned.  

To run the training of the models, you need to run the command:	

- python3 main_modelname.py

In this way you will train the models and save the best for each run.

Then to evaluate on the test set you just need to run the command:
	
- python3 test_modelname.py

You need to substitute modelname with one of the following names:

- SINGLE
- PAIR	
- TC
- TC_T
- TC_U
- TC_U_T

Every possible suggestion for improvements, typo, missing information is highly encouraged. Please contact npenzo@fbk.eu.

If you use the information provided in this repo, please cite:

@misc{penzo2024putting,
      title={Putting Context in Context: the Impact of Discussion Structure on Text Classification}, 
      author={Nicolò Penzo and Antonio Longa and Bruno Lepri and Sara Tonelli and Marco Guerini},
      year={2024},
      eprint={2402.02975},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


