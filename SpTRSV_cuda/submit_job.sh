#!/bin/bash
#
# -= Resources =-
#
#SBATCH --job-name=sptrsv
#SBATCH --nodes=1
#SBATCH --partition=dgx2q
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=sptrsv.out
#SBATCH --mail-user=hsasmaz16@ku.edu.tr
#SBATCH --mail-type=END

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

echo "==============================================================="
echo "EQUAL COLUMN PARTITION"
echo "==============================================================="

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../Flan_1565/Flan_1565.mtx

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
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../belgium_osm/belgium_osm.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../citationCiteseer/citationCiteseer.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../dblp-2010/dblp-2010.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../delaunay_n20/delaunay_n20.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../pkustk14/pkustk14.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../roadNet-CA/roadNet-CA.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx ../webbase-1M/webbase-1M.mtx

echo "==============================================================="
echo "EQUAL NNZ PARTITION"
echo "==============================================================="

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../Flan_1565/Flan_1565.mtx

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
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../citationCiteseer/citationCiteseer.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../dblp-2010/dblp-2010.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../delaunay_n20/delaunay_n20.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../pkustk14/pkustk14.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../roadNet-CA/roadNet-CA.mtx

echo
./sptrsv -d 0 -rhs 1 -forward -p 1 -mtx ../webbase-1M/webbase-1M.mtx

