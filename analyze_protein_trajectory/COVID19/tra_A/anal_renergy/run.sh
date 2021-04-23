
# perform analysis
# (cpptraj-15 is necessary here)

/opt/nbcc/bin/cpptraj-15 -i anal.in

# make directories if necessary

for dirname in BINARY monomer_1 monomer_2
do

  if [ ! -d $dirname ]; then
    mkdir $dirname
  fi

done

# rename and make list_files

mv renergy.dat.*           ./BINARY
mv renergy_monomer_1.dat.* ./monomer_1
mv renergy_monomer_2.dat.* ./monomer_2

cd BINARY
sh ../rename.sh
ls renergy_*.dat > list_files
cd ..

cd monomer_1
sh ../rename_monomer_1.sh
ls renergy_*.dat > list_files_monomer_1
cd ..

cd monomer_2
sh ../rename_monomer_2.sh
ls renergy_*.dat > list_files_monomer_2
cd ..

