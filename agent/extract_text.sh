#!/bin/bash
lectures=("deep_learning" "dimensionality_reduction" "clustering" "ensemble_learning" "classification" "regression" "preprocessing" "Math_Preliminary" "introduction")
for lecture in ${lectures[@]};
do
	nohup python extract_advanced_text.py $lecture
done
