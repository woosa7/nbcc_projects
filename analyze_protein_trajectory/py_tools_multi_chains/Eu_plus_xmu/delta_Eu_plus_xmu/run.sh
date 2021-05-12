
cp ../../../anal_tenergy/inter_energy.dat .

cp ../../xmu_only_calc/data_vs_time/BINARY/xmu_vs_time.dat    ./xmu_BINARY_only_vs_time.dat
cp ../../xmu_only_calc/data_vs_time/monomer_1/xmu_vs_time.dat ./xmu_monomer_1_vs_time.dat
cp ../../xmu_only_calc/data_vs_time/monomer_2/xmu_vs_time.dat ./xmu_monomer_2_vs_time.dat

/homes/eta/users/chong/project_covid19/tools/calc_delta_Eu_plus_xmu/1000_ps_interval/run.exe

rm -rf inter_energy.dat

rm -rf xmu_BINARY_only_vs_time.dat
rm -rf xmu_monomer_1_vs_time.dat
rm -rf xmu_monomer_2_vs_time.dat

