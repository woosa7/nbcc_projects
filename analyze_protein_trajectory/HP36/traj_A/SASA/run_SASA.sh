#
# set environments to use naccess
#

NACCESS_PATH="/opt/apps/naccess"
NACCESS_ARGS=" -l -f -y"

#
# main part
#

cat list_files | while read filename
do

# create directory and cd to the directory

  echo $filename

  cd SASA_data

  mkdir $filename
  cd $filename

# copy pdb file

  cp ../pdb_from_crd/$filename.pdb ./protein.pdb

# calculate SASA

  $NACCESS_PATH/naccess protein.pdb -r $NACCESS_PATH/vdw.radii -s $NACCESS_PATH/standard.data $NACCESS_ARGS

# delete unnecessary files

  rm -rf protein.pdb
  rm -rf protein.log
  rm -rf accall.input

# for the next step

  cd ../..

done

