# Which Future Events Can Be Forecast? A Novel Temporal Knowledge Graph Forecasting Evaluation Framework with Strikingness Measuring

## Virtual environment
Building a Venv virtual environment (or Conda environment)
```
python -m venv RSMF
./RSMF/Scripts/activate 
pip install -r requirements.txt
```

## Result reproduction
We provide all script commands for **Strikingness Measuring** and **Evaluation** in `./src/run.sh`. More details are as follows.
## Strikingness Measuring
1. Rule Mining: Mining Temporal Logic Rules for Strikingness Measurement. 

```
# Taking the ICEWS14 dataset as an example
cd data
python raw_data_gen.py -d <dataset>  # Generate samples in text format

cd ../src
python learn.py -d ICEWS14 -l 1 -n 200 -p 16 -s 6
```
- We provide the generated rule set files in path `./output_rule/<dataset>`.

2. Strikingness score calculation
```
# Taking the ICEWS14 dataset as an example
python calc_out_score_test.py -d ICEWS14 -s 6 --rules_file ruleLen[1]_numWalk200_exp_rules.json -l 1 \
    --conf_threshold 0.01 --decay 0.1 --his_len 200 --obj_coeff 0.4 --sub_coeff 0.4 --rel_coeff 0.2 \
    --out_threshold 1.0 --save_out_score
```
- We provide the strikingness score files for reproducing the results of the paper in path `./data/<dataset>/0.01_0.1_200_0.4_0.4_0.2`. The path contains two files:
   - `out_score_test.json`: the file contains the list of strikingness scores of test samples divided by timestamps, the format is `{'test_out_store ': [[Strikingness list of timestamp1], [Strikingness list of timestamp1], ...]}`.
   - `test_extend.txt`: One line for each quadruple of test samples and their corresponding strikingness scores.


## Evaluation
Calculate the original and weighted metrics using the ranking file of the models.
```
# Taking the ICEWS14 dataset and regcn model as an example
python evaluate.py -d ICEWS14 --model regcn --bias 0.1
```
- We provide test set ranking files for each model on four datasets to reproduce the results of the paper in path `./rank_file/<dataset>/<dataset>_<model>.txt`.
- If you want to test and evaluate the forecasting capability of other models, you can generate a corresponding ranking file that includes the test sample and model prediction rank for each row in the file, like `s, r, o, t, rank`. Additionally, ensure that the dataset version you are using is the same as the dataset version we provide, otherwise you may need to make additional adjustments to the dataset.