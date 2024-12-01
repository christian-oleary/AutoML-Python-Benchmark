# Dockerfile for EvalML

FROM tensorflow/tensorflow:latest-gpu

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y git graphviz && rm -rf /var/lib/apt/lists/*

# Clone the EvalML repository which contains source code and tests
RUN rm -rf /src/evalml && git clone https://github.com/alteryx/evalml.git /src/evalml

# Set the working directory to the EvalML source code
WORKDIR /src/evalml/

# Set the test timeout
ENV TIMEOUT=3600

# ARG run_tests
# RUN if [ "$run_tests" = "true" ]; then make installdeps-dev; fi
# RUN if [ "$run_tests" = "true" ]; then make installdeps-prophet; fi
# RUN if [ "$run_tests" = "true" ]; then make installdeps-test; fi

# # RUN if [ "$run_tests" = "true" ]; then pip freeze && (TIMEOUT=$TIMEOUT make test); fi
# RUN if [ "$run_tests" = "true" ]; then pip freeze && make test; fi

# RUN python -c "import torch;print('GPU STATUS:',torch.cuda.is_available())"
