
cp ../../../anal_tenergy/tenergy_monomer_1.dat ./energy.dat
cp ../../xmu_only_calc/data_vs_time/monomer_1/xmu_vs_time.dat .

/homes/eta/users/chong/project_covid19/tools/calc_Eu_plus_xmu/1000_ps_interval/run.exe

rm -rf energy.dat

