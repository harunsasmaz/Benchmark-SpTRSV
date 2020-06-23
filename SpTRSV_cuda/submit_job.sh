#!/bin/bash
#
# -= Resources =-
#
#SBATCH --job-name=sptrsv-job
#SBATCH --nodes=1
#SBATCH --partition=dgx2q
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=sptrsv.out

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

echo "Compiling the code...!"
make
echo "==============================================================================="
echo "Running Job...!"
echo "==============================================================================="

echo
echo "EQUAL COLUMN PARTITION"
echo

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../Cube_Coup_dt6/Cube_Coup_dt6.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../Flan_1565/Flan_1565.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../af_shell10/af_shell10.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../G3_circuit/G3_circuit.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../memchip/memchip.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../nlpkkt120/nlpkkt120.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../nlpkkt160/nlpkkt160.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../nlpkkt200/nlpkkt200.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../nlpkkt240/nlpkkt240.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../cage15/cage15.mtx

echo
#./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../circuit5N_dc/circuit5N_dc.mtx


echo
echo "EQUAL NNZ PARTITION"
echo

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../Cube_Coup_dt6/Cube_Coup_dt6.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../Flan_1565/Flan_1565.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../af_shell10/af_shell10.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../G3_circuit/G3_circuit.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../memchip/memchip.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../nlpkkt120/nlpkkt120.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../nlpkkt160/nlpkkt160.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../nlpkkt200/nlpkkt200.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../nlpkkt240/nlpkkt240.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../cage15/cage15.mtx

# echo
# #./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../circuit5N_dc/circuit5N_dc.mtx
