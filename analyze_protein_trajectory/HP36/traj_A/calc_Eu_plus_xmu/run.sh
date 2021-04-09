cp ../../traj_A/anal_tenergy/tenergy.dat .

cp ../../traj_A/3D-RISM/xmu_calc/xmu_vs_time/xmu_vs_time.dat .
cp ../../traj_A/3D-RISM/xmu_calc/xmu_vs_time/xmu_vs_time_every_250ps.dat .


python calc_Eu_plus_xmu.py


rm -rf tenergy.dat

