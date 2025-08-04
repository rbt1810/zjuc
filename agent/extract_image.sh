#!/bin/bash
lectures=("deep_learning" "dimensionality_reduction" "clustering" "ensemble_learning" "classification" "regression" "preprocessing" "Math_Preliminary" "introduction")
for lecture in ${lectures[@]};
do
	mkdir lectures/$lecture/Images
        nohup python image_extract.py $lecture
done
