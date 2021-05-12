#!//bin/bash
for((i=0;i<10;i++)); do
  python autoscript.py next --work_dir /mnt/storage/pdb_database
done
