# Strikingness Measuring

# ICEWS14
# python learn.py -d ICEWS14 -l 1 -n 200 -p 16 -s 6
# python calc_out_score_test.py -d ICEWS14 -s 6 --rules_file ruleLen[1]_numWalk200_exp_rules.json -l 1 \
#         --conf_threshold 0.01 --decay 0.1 --his_len 200 --obj_coeff 0.4 --sub_coeff 0.4 --rel_coeff 0.2 \
#         --out_threshold 1.0 --save_out_score

# ICEWS18
# python learn.py -d ICEWS18 -l 1 -n 200 -p 16 -s 6
# python calc_out_score_test.py -d ICEWS18 -s 6 --rules_file ruleLen[1]_numWalk200_exp_rules.json -l 1 \
#         --conf_threshold 0.01 --decay 0.1 --his_len 200 --obj_coeff 0.4 --sub_coeff 0.4 --rel_coeff 0.2 \
#         --out_threshold 1.0 --save_out_score

# ICEWS05-15
# python learn.py -d ICEWS05-15 -l 1 -n 200 -p 16 -s 6
# python calc_out_score_test.py -d ICEWS05-15 -s 6 --rules_file ruleLen[1]_numWalk200_exp_rules.json -l 1 \
#         --conf_threshold 0.01 --decay 0.1 --his_len 200 --obj_coeff 0.4 --sub_coeff 0.4 --rel_coeff 0.2 \
#         --out_threshold 1.0 --save_out_score

# GDELT
# python learn.py -d GDELT -l 1 -n 200 -p 16 -s 6
# python calc_out_score_test.py -d GDELT -s 6 --rules_file ruleLen[1]_numWalk200_exp_rules.json -l 1 \
#         --conf_threshold 0.01 --decay 0.1 --his_len 200 --obj_coeff 0.4 --sub_coeff 0.4 --rel_coeff 0.2 \
#         --out_threshold 1.0 --save_out_score

# Striking-Aware Evaluation
# alternative models: [recurrency, titer, tlogic, regcn, tirgn, logcl, tlogic_logcl_0.8/0.7, icl, gentkg]
# PS: The ranking files of icl and gentkg models are only available on ICEWS14 and ICEWS18

# python evaluate.py -d ICEWS14 --model regcn --bias 0.1
# python evaluate.py -d ICEWS18 --model regcn --bias 0.1
# python evaluate.py -d ICEWS05-15 --model tlogic_logcl_0.7 --bias 0.1
# python evaluate.py -d GDELT --model regcn --bias 0.1
