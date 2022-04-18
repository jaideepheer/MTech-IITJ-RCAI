#! /bin/bash

# Input environment name
read -p "Enter environment name[mtpignite]: " name
if [ -z "$name" ]
then 
    name="mtpignite"
fi
name=${name,,}
echo "Using name: $name"

# Ensure conda is ready
# Ref: https://stackoverflow.com/a/56155771/10027894
eval "$(conda shell.bash hook)"

# Check if env exists
if $(conda env list | grep -q "^$name"); then
    # env exists
    echo "Environment $name already exists, want to update it? [y/N]"
    read inp
    if [ "$inp" = "y" ] || [ "$inp" = "Y" ]
    then
        mamba env update -f environment.yaml
    fi
    conda activate $name
else
    # Ask to install from lockfile
    echo "Install from lockfile? [Y/n]"
    read inp
    echo "Creating environment $name..."
    if [ "$inp" = "y" ] || [ "$inp" = "Y" ] || [ -z "$inp" ]
    then
        mamba env create -f environment.lock.yaml
    else
        mamba env create -f environment.yaml
        conda env export | head -n -1 > environment.lock.yaml
    fi
    conda activate $name
    python -m ipykernel install --user --name research --display-name "research ($(python --version))"
fi