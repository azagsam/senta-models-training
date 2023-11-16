# SENTA

This repository contains the code used to train the SENTA Slovene text simplification system modules.
It is composed of a complexity detector and a text simplifier: when a text is passed into the system, 
the complexity detector first checks if the text is already simple. If **it is not**, it is further passed 
to the text simplifier model.   

Scripts contained in the repository:  
- `newsela-prepare-training.py`: Preprocessing code used to clean the Slovene translation of the Newsela dataset;  
- `t5_train.py`: Code used to train the text simplifier;  
- `train_text_complexity`: Code used to train the text complexity detector (is simple vs is not simple);  
- `t5_inference_twostep.py`: Code used in the simplification system using a two-step pipeline.
