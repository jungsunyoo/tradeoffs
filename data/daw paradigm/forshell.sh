#!/bin/bash
for ((fnum=1; fnum<=36; fnum+=1))
do
     sbatch forloop.slurm "$fnum"
done 
