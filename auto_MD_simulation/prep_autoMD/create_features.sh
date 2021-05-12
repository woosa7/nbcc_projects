yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/make_features_prowise.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/make_single_features.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/make_3Dimage_LJ_elec.py .
yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/feature_parameters .

python make_features_prowise.py feature_parameters

# yes|cp /homes/eta/users/aiteam/prowise/nbcc_autoMD/make_features_from_unseen.py .
# python make_features_from_unseen.py feature_parameters
