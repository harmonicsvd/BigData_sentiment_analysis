printf "Conda Env creation...\n"
conda create -n a3 python=$PYTHON_VERSION -y >/dev/null 2>&1
printf "Conda Env creation successful!\n"
conda env list
printf "\n"