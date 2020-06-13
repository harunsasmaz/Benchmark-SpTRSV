#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=sptrsv-job
#SBATCH --nodes=1
#SBATCH --partition=dgx2q
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:2
#SBATCH --output=sptrsv.out

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."


#parallel version
echo
./sptrsv -d 0 -rhs 1 -forward -mtx ../Flan_1565/Flan_1565.mtx
