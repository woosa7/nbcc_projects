export AMBERHOME=/opt/apps/amber20/amber20
export CUDA_HOME=/opt/cuda-10.0

export CUDA_VISIBLE_DEVICES="0"

pmemd.cuda -O -i min1.in -o min1.out -p ../01_tleap/cv30com_rbd.top -c ../01_tleap/cv30com_rbd_initial.crd -r cv30com_rbd_min1.rst -ref ../01_tleap/cv30com_rbd_initial.crd
pmemd.cuda -O -i min2.in -o min2.out -p ../01_tleap/cv30com_rbd.top -c cv30com_rbd_min1.rst -r cv30com_rbd_min2.rst
