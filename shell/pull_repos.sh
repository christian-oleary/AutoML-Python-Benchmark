#!/bin/bash

# Usage: sh ./shell/pull_repos.sh

echo Cloning repositories...

# Clone repository or pull if already existing
if cd ./repositories/adanet; then echo Adanet: && git pull && cd ../../; else mkdir -p ./repositories/adanet && git clone https://github.com/tensorflow/adanet.git ./repositories/adanet; fi
if cd ./repositories/Auto-PyTorch; then echo Auto-PyTorch: && git pull && cd ../../; else mkdir -p ./repositories/Auto-PyTorch && git clone https://github.com/automl/Auto-PyTorch.git ./repositories/Auto-PyTorch; fi
if cd ./repositories/autogluon; then echo AutoGluon: && git pull && cd ../../; else mkdir -p ./repositories/autogluon && git clone https://github.com/autogluon/autogluon.git ./repositories/autogluon; fi
if cd ./repositories/autokeras; then echo AutoKeras: && git pull && cd ../../; else mkdir -p ./repositories/autokeras && git clone https://github.com/keras-team/autokeras.git ./repositories/autokeras; fi
if cd ./repositories/auto-sklearn; then echo auto-sklearn: && git pull && cd ../../; else mkdir -p ./repositories/temp && git clone https://github.com/automl/auto-sklearn ./repositories/auto-sklearn; fi
if cd ./repositories/h2o-3; then echo H2O: && git pull && cd ../../; else mkdir -p ./repositories/h2o-3 && git clone https://github.com/h2oai/h2o-3 ./repositories/h2o-3; fi
if cd ./repositories/tpot; then echo TPOT: && git pull && cd ../../; else mkdir -p ./repositories/tpot && git clone https://github.com/epistasislab/tpot ./repositories/tpot; fi
if cd ./repositories/MLBox; then echo MLBox: && git pull && cd ../../; else mkdir -p ./repositories/MLBox && git clone https://github.com/AxeldeRomblay/MLBox ./repositories/MLBox; fi
if cd ./repositories/hyperopt-sklearn; then echo hyperopt-sklearn: && git pull && cd ../../; else mkdir -p ./repositories/hyperopt-sklearn && git clone https://github.com/hyperopt/hyperopt-sklearn ./repositories/hyperopt-sklearn; fi
if cd ./repositories/autogluon; then echo AutoGluon: && git pull && cd ../../; else mkdir -p ./repositories/autogluon && git clone https://github.com/awslabs/autogluon ./repositories/autogluon; fi
if cd ./repositories/mljar-supervised; then echo mljar: && git pull && cd ../../; else mkdir -p ./repositories/mljar-supervised && git clone https://github.com/mljar/mljar-supervised ./repositories/mljar-supervised; fi
if cd ./repositories/pycaret; then echo PyCaret: && git pull && cd ../../; else mkdir -p ./repositories/pycaret && git clone https://github.com/pycaret/pycaret ./repositories/pycaret; fi
if cd ./repositories/LightAutoML; then echo LightAutoML: && git pull && cd ../../; else mkdir -p ./repositories/LightAutoML && git clone https://github.com/AILab-MLTools/LightAutoML ./repositories/LightAutoML; fi
if cd ./repositories/AutoTS; then echo AutoTS: && git pull && cd ../../; else mkdir -p ./repositories/AutoTS && git clone https://github.com/winedarksea/AutoTS ./repositories/AutoTS; fi
if cd ./repositories/evalml; then echo EvalML: && git pull && cd ../../; else mkdir -p ./repositories/evalml && git clone https://github.com/alteryx/evalml ./repositories/evalml; fi
if cd ./repositories/FLAML; then echo FLAML: && git pull && cd ../../; else mkdir -p ./repositories/FLAML && git clone https://github.com/microsoft/FLAML ./repositories/FLAML; fi
if cd ./repositories/FEDOT; then echo FEDOT: && git pull && cd ../../; else mkdir -p ./repositories/FEDOT && git clone https://github.com/nccr-itmo/FEDOT ./repositories/FEDOT; fi
if cd ./repositories/etna; then echo ETNA: && git pull && cd ../../; else mkdir -p ./repositories/etna && git clone https://github.com/tinkoff-ai/etna ./repositories/etna; fi
if cd ./repositories/pyodds; then echo PyODDs: && git pull && cd ../../; else mkdir -p ./repositories/pyodds && git clone https://github.com/datamllab/pyodds ./repositories/pyodds; fi
if cd ./repositories/Kats; then echo Kats: && git pull && cd ../../; else mkdir -p ./repositories/Kats && git clone https://github.com/facebookresearch/Kats ./repositories/Kats; fi
if cd ./repositories/Meta-AAD; then echo Meta-AAD: && git pull && cd ../../; else mkdir -p ./repositories/Meta-AAD && git clone https://github.com/daochenzha/Meta-AAD ./repositories/Meta-AAD; fi
if cd ./repositories/MetaOD; then echo MetaOD: && git pull && cd ../../; else mkdir -p ./repositories/MetaOD && git clone https://github.com/yzhao062/MetaOD ./repositories/MetaOD; fi

echo Repositories ready
