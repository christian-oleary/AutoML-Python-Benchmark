
FROM christianoleary/fedot

# Clone the Fedot.Industrial repository which contains source code and tests
# RUN rm -rf /src/fedot && rm -rf /src/fedot_ind && git clone https://github.com/aimclub/Fedot.Industrial /src/fedot_ind
RUN rm -rf /src/fedot /src/fedot_ind /src/Fedot.Industrial && \
    git clone https://github.com/aimclub/Fedot.Industrial /src/Fedot.Industrial

# Set the working directory to the Fedot.Industrial source code
WORKDIR /src/Fedot.Industrial/

# Run unit tests
ARG run_tests
# poetry run pytest --cov=fedot_ind --cov-report xml:coverage.xml tests/unit -vvv
RUN if [ "$run_tests" = "true" ]; then \
        echo "Running unit tests..." && \
        (python3 -m pytest ./tests/unit/ --cov-branch --cov-report xml:coverage.xml --cov=. || ls coverage.xml) && \
        (cat coverage.xml | grep "hits=" | grep -Eo '[0-9]+' | grep -v '^0$' | wc -l || (echo "No coverage data" && exit 1)) && \
        echo "Tests complete"; \
    else \
        echo "Skipping tests"; \
    fi

RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())" && \
    python -c "import tensorflow as tf;print('GPU STATUS:',tf.test.is_built_with_cuda())"
