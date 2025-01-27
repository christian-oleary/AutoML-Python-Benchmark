#!/bin/bash

################################################################################
# This program sets up SonarQube and runs a scan on the project.
################################################################################

# shellcheck disable=SC2086

# Exit on error
set -e

########
# Usage:
########
# (ensure python and conda are installed)
# sudo apt-get install zip unzip
# ./scripts/sonar_scanner.sh

################################################################################
# LOGGING FUNCTIONS
################################################################################

# ANSI color codes
LIGHT_BLUE='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RESET='\033[0m'

print_heading() {
    local message="  $1  "
    local border
    border=$(printf '%*s' "${#message}" '' | tr ' ' '=')
    echo -e "\n${LIGHT_BLUE}${border}\n${message}\n${border}${RESET}\n"
}

print_subheading() {
    local message="  $1  "
    local border
    border=$(printf '%*s' "${#message}" '' | tr ' ' '-')
    echo -e "\n${YELLOW}${border}\n${message}\n${border}${RESET}"
}

print_line() {
    local message=$1
    echo -e "${GREEN}$0:${BASH_LINENO[0]}: -> ${message}\n${RESET}"
}

################################################################################
# ENSURE SCRIPT IS RUN ON LINUX OR WSL
################################################################################
if [[ "$(uname -s)" != "Linux" && ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
    echo "This script can only be run on Linux or WSL."
    exit 1
fi

################################################################################
# ENVIRONMENT VARIABLES
################################################################################
print_heading "Environment Variables"

# Delete and replace existing projects in SonarQube
export DELETE_EXISTING_PROJECTS=${DELETE_EXISTING_PROJECTS:-"false"}

# Path to the repositories directory
export REPOSITORIES_DIR=${REPOSITORIES_DIR:-"repositories"}

# Skip tests and running sonar scanner if results already exist
export SKIP_EXISTING_RESULTS=${SKIP_EXISTING_RESULTS:-"true"}

# Skip building existing docker images
export SKIP_REBUILDING_IMAGES=${SKIP_REBUILDING_IMAGES:-"false"}

# Run SonarScanner using Docker
export SONAR_SCANNER_DOCKER=${SONAR_SCANNER_DOCKER:-"false"}

# SonarQube server
export SONAR_HOST_URL=${SONAR_HOST_URL:-"http://localhost:9000"}
export SONAR_LOGIN=${SONAR_LOGIN:-"admin"}
export SONAR_PASSWORD=${SONAR_PASSWORD:-"2c71d75bcs4hq934tdjqngtsojxercm"}
API_URL="${SONAR_HOST_URL}/api/"

print_line "REPOSITORIES_DIR=${REPOSITORIES_DIR}"
print_line "SKIP_EXISTING_RESULTS=${SKIP_EXISTING_RESULTS}"
print_line "SKIP_REBUILDING_IMAGES=${SKIP_REBUILDING_IMAGES}"
print_line "SONAR_SCANNER_DOCKER=${SONAR_SCANNER_DOCKER}"

print_line "API_URL=${API_URL}"
print_line "DELETE_EXISTING_PROJECTS=${REPOSITORIES_DIR}"
print_line "SONAR_HOST_URL=${SONAR_HOST_URL}"
print_line "SONAR_LOGIN=${SONAR_LOGIN}"
print_line "SONAR_PASSWORD=${SONAR_PASSWORD}"

################################################################################
# CHANGE ADMIN PASSWORD
################################################################################
print_heading "Setting admin password"

curl -u ${SONAR_LOGIN}:${SONAR_LOGIN} -X POST \
    "${API_URL}users/change_password?login=${SONAR_LOGIN}&previousPassword=${SONAR_LOGIN}&password=${SONAR_PASSWORD}"

printf "Done\n"

################################################################################
# DOWNLOAD SONARSCANNER CLI TOOL (IF NOT USING DOCKER)
################################################################################
print_heading "SonarScanner CLI Tool Setup"

# Skip if using Docker
if [ "$SONAR_SCANNER_DOCKER" = "false" ]; then
    # Check if SonarScanner exists
    TOOL="sonar-scanner-cli"
    if [ ! -d $TOOL ]; then
        #######################
        # Download SonarScanner
        #######################
        print_line "Downloading $TOOL..."
        URL="https://binaries.sonarsource.com/Distribution/${TOOL}/${TOOL}"
        if [ "$(uname)" = "Linux" ]; then
            scanner_url="${URL}-4.6.2.2472-linux.zip"
        else
            scanner_url="${URL}-6.2.1.4610-windows-x64.zip"
        fi

        # if $TOOL directory is not found, download it
        if [ ! -d $TOOL ]; then
            curl --clobber -sL "${scanner_url}" -o $TOOL.zip || \
                curl -sL "${scanner_url}" -o $TOOL.zip
        fi

        ########################################
        # Unzip SonarScanner and remove zip file
        ########################################
        print_line "Download finished. Unzipping..."
        unzip -q $TOOL.zip
        rm $TOOL.zip

        ###############################
        # Rename SonarScanner directory
        ###############################
        mv sonar-scanner-* $TOOL
        print_line "SonarScanner downloaded to $(realpath .)/${TOOL}"
    else
        print_line "${TOOL} already exists"
    fi
    ##########################
    # Add SonarScanner to PATH
    ##########################
    PATH=$PATH:$(realpath .)/${TOOL}/bin
else
    print_line "Using SonarScanner Docker Image. Skipping download..."
fi

################################################################################
# FIND ALL REPOSITORIES IN REPOSITORIES_DIR
################################################################################
print_heading "Repositories in '${REPOSITORIES_DIR}'"

repositories=$(find ./$REPOSITORIES_DIR -mindepth 1 -maxdepth 1 -type d)
# repositories="./repositories/auto_pytorch"
print_line "Repository directories found:\n$repositories"

##################################
# Delete all projects in SonarQube
##################################
# print_heading "Deleting All Projects"
# # Fetch all project keys
# project_keys=$(curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
#     -X GET "${API_URL}projects/search" | grep -oP '(?<="key":")[^"]*')
# # Loop over each project key and delete the project
# for project_key in $project_keys; do
#     print_line "Deleting project: $project_key"
#     curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
#         "${API_URL}projects/delete?project=${project_key}"
# done

#########################
# Create logs directories
#########################
mkdir -p logs; mkdir -p logs/sca/sonar/

################################################################################
# LOOP OVER EACH REPOSITORY
################################################################################
previous_image=""

for repo_path in $repositories; do
    # Get the name of the repository from directory path
    repo_name=$(basename "$repo_path")
    image_name="christianoleary/${repo_name}"

    print_heading "Processing directory: ${repo_path} (${repo_name} - ${image_name})"

    ############################################################################
    # SKIP IF NOT A DIRECTORY
    ############################################################################
    if [ ! -d "$repo_path" ]; then
        echo "Skipping $repo_path (not a directory)"
        continue
    fi

    ############################################################################
    # CREATE OUTPUT DIRECTORY
    ############################################################################
    OUTPUT_DIR="./results/sca/sonar/${repo_name}"
    mkdir -p $OUTPUT_DIR
    OUTPUT_FILE="${OUTPUT_DIR}/measures.json"

    ############################################################################
    # ENVIRONMENT VARIABLES
    ############################################################################
    print_subheading "Environment Variables (project: $repo_name)"

    export PROJECT_BRANCH="master"
    export PROJECT_NAME=$repo_name
    export PROJECTKEY=$PROJECT_NAME
    export PROJECT_KEY=$PROJECT_NAME
    export SONAR_PROJECTKEY=$PROJECT_NAME
    export SONAR_PROJECT_KEY=$PROJECT_NAME
    export TARGET_DIR="${repo_path}/"

    # Print newly added environment variables
    print_line "PROJECT_BRANCH=${PROJECT_BRANCH}"
    print_line "PROJECT_NAME=${PROJECT_NAME}"
    print_line "PROJECTKEY=${PROJECTKEY}"
    print_line "PROJECT_KEY=${PROJECT_KEY}"
    print_line "SONAR_PROJECTKEY=${SONAR_PROJECTKEY}"
    print_line "SONAR_PROJECT_KEY=${SONAR_PROJECT_KEY}"
    print_line "TARGET_DIR=${TARGET_DIR}"

    ################################################################################################
    # RUN PYTHON TESTS
    ################################################################################################
    print_subheading "Running Python Tests (project: $repo_name)"

    # Remove irrelevant files that may be picked up by Sonar
    rm -f coverage.xml report.xml

    ###################################################################################
    # Pull or build Docker image if not SKIP_REBUILDING_IMAGES or if image does not exist
    ###################################################################################
    print_line "Attempting to pull Docker image (${image_name})..."
    docker pull $image_name || (echo "Pull failed!")  # && docker image ls)

    if [ -z "$(docker images -q ${image_name} 2> /dev/null)" ]; then
        print_line "Docker image not yet pushed (${image_name})"
        # Build Docker image
        if [ "$SKIP_REBUILDING_IMAGES" == "false" ]; then
            print_line "Building Docker image ${image_name}..."

            # Build image with tests enabled and save logs to file
            (docker build --build-arg run_tests=true --progress plain -t "${image_name}" \
                -f ./src/ml/automl/$repo_name/Dockerfile . 2>&1 | tee ./logs/sca/sonar/$repo_name.log) || exit 1

            print_line "Docker image ${image_name} built successfully."
        else
            # Skip building image
            print_line "Docker image ${image_name} already exists. Skipping build..."
        fi
    fi

    ##################################################
    # Copy contents of relevant coverage files to host
    ##################################################
    rm -f .coverage coverage.xml report.xml

    # coverage.xml
    print_line "Reading coverage.xml from Docker container..."
    docker run --gpus all --rm --name $repo_name "${image_name}" bash -c "cat coverage.xml" > $OUTPUT_DIR/coverage.xml
    print_line "Misses: $(cat $OUTPUT_DIR/coverage.xml | grep -c "hits=\"0\"")"
    print_line "Hits: $(cat $OUTPUT_DIR/coverage.xml | grep -c "hits=\"1\"")"
    print_line "Total: $(cat $OUTPUT_DIR/coverage.xml | grep -c "hits=")"

    # .coverage
    print_line "Reading .coverage from Docker container..."
    docker run --gpus all --rm --name $repo_name "${image_name}" bash -c "cat .coverage" > $OUTPUT_DIR/.coverage

    # If fedot, delete first 3 lines of coverage.xml
    if [ "$repo_name" = "fedot" ]; then
        sed -i '1,4d' $OUTPUT_DIR/coverage.xml
        sed -i '1,4d' $OUTPUT_DIR/.coverage
    fi

    ##########################################################
    # Copy coverage files to target directory for SonarScanner
    ##########################################################
    cp $OUTPUT_DIR/{coverage.xml,.coverage} $TARGET_DIR
    ls $TARGET_DIR/coverage.xml $TARGET_DIR/.coverage || (echo "missing files" && exit 1)

    ######################
    # Show coverage report
    ######################
    print_line "Coverage Report:"
    docker run --gpus all --rm --name $repo_name "${image_name}" bash -c "coverage report" > $OUTPUT_DIR/report.txt

    ################################################################################################
    # PUSH DOCKER IMAGE TO DOCKER HUB (docker push should skip existing layers)
    ################################################################################################
    print_line "Checking for previous Docker push to complete before continuing${previous_image}..."
    wait
    print_line "Continuing with Docker push (project: $repo_name)..."
    previous_image=" (${image_name})"
    docker push "${image_name}" &

    # continue
    # break

    ############################################################################
    # SKIP IF OUTPUT FILE EXISTS AND SKIP_EXISTING_RESULTS IS TRUE
    ############################################################################
    if [ -f "${OUTPUT_FILE}" ] && [ "${SKIP_EXISTING_RESULTS}" = "true" ]; then
        print_line "Output file $OUTPUT_FILE already exists. Skipping ${repo_name}..."
        continue
    fi

    ################################################################################################
    # CREATE/REPLACE USER ACCESS TOKEN
    ################################################################################################
    print_subheading "Generating user access token (project: $repo_name)"

    ###############################
    # Revoke old token if it exists
    ###############################
    print_line "Revoking old token..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
        "${API_URL}user_tokens/revoke?name=${PROJECT_NAME}&login=${SONAR_LOGIN}"

    ################################
    # Remove token.json if it exists
    ################################
    rm -f token.json

    ####################
    # Generate new token
    ####################
    print_line "Generating new user access token..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
        -X POST "${API_URL}user_tokens/generate?name=${PROJECT_NAME}&login=${SONAR_LOGIN}" -o token.json

    ###############################################
    # Read token from file and export as a variable
    ###############################################
    SONAR_TOKEN=$(grep -oP '(?<="token":")[^"]*' token.json)
    export SONAR_TOKEN
    print_line "SONAR_TOKEN: $SONAR_TOKEN"

    #############
    # List tokens
    #############
    # print_line "User tokens in SonarQube:\n"
    # curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} \
    #     -X POST "${API_URL}user_tokens/search"  | python -m json.tool

    ################################################################################################
    # CREATE sonar-project.properties FILE
    ################################################################################################
    print_subheading "Updating sonar-project.properties file (project: $repo_name)"
    # ABSOLUTE_PATH=$(realpath $TARGET_DIR)
    ABSOLUTE_PATH=$TARGET_DIR
    cat <<EOL >$OUTPUT_DIR/sonar-project.properties
sonar.projectKey=$SONAR_PROJECT_KEY
sonar.projectName=$SONAR_PROJECT_KEY
sonar.sources=$ABSOLUTE_PATH
sonar.host.url=$SONAR_HOST_URL
sonar.login=$SONAR_LOGIN
sonar.password=$SONAR_PASSWORD
sonar.token=$SONAR_TOKEN
sonar.language=py
sonar.python.coverage.reportPaths=coverage.xml,$TARGET_DIR/coverage.xml,$TARGET_DIR/**/coverage.xml,$OUTPUT_DIR/coverage.xml,$ABSOLUTE_PATH/coverage.xml
sonar.python.coverage.itReportPath=$TARGET_DIR/.coverage
sonar.scm.disabled=true
EOL
    # sonar.python.coverage.reportPaths=coverage.xml,$TARGET_DIR/coverage.xml,$TARGET_DIR/**/coverage.xml,$OUTPUT_DIR/coverage.xml,$ABSOLUTE_PATH/coverage.xml

    # sonar.working.directory=$OUTPUT_DIR
    cat "$OUTPUT_DIR/sonar-project.properties"
    # sonar.python.version=3.x
    # sonar.sourceEncoding=UTF-8

    ################################################################################################
    # SET UP PROJECT IN SONARQUBE
    ################################################################################################
    print_subheading "Set Up Project (project: $repo_name)"

    #####################################
    # Exit if coverage.xml does not exist
    #####################################
    ls $TARGET_DIR/coverage.xml || (echo "coverage.xml not found in $TARGET_DIR" && exit 1)

    #####################################
    # Delete project if it already exists
    #####################################
    if [ "$DELETE_EXISTING_PROJECTS" = "true" ]; then
        print_line "Deleting old project..."
        curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
            "${API_URL}projects/delete?project=${PROJECT_NAME}&branch=${PROJECT_BRANCH}"
    fi

    ####################
    # Create new project
    ####################
    print_line "Creating new project..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
        "${API_URL}projects/create?name=${PROJECT_NAME}&project=${PROJECT_NAME}&projectKey=${PROJECT_KEY}&token=${SONAR_TOKEN}&branch=${BRANCH}"
    printf "\n"

    # ###############
    # # List projects
    # ###############
    # print_line "Projects in SonarQube:\n"
    # curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
    #     -X POST "${API_URL}components/search?qualifiers=TRK" | python -m json.tool

    ################################################################################################
    # RUN SONAR SCANNER
    ################################################################################################
    print_subheading "Running SonarScanner (project: $repo_name)"

    # Remove irrelevant files
    rm -f coverage.xml report.xml

    if [ "${SONAR_SCANNER_DOCKER}" = "false" ]; then
        ###########################
        # Run SonarScanner CLI tool
        ###########################
        # Run downloaded binary
        print_line "Running downloaded SonarScanner CLI tool..."
        # Alias sonar-scanner to the correct binary
        if [ "$(uname)" = "Linux" ]; then
            sonar_scanner() { sonar-scanner "$@"; }
        else
            sonar_scanner() { sonar-scanner.bat "$@"; }
        fi
        sonar_scanner -X -Dproject.settings=$OUTPUT_DIR/sonar-project.properties
    else
        ###############################
        # Run SonarScanner Docker image
        ###############################
        print_line "Running SonarScanner Docker image..."
        docker run --network=host --rm \
            -e PROJECT_SETTINGS=$OUTPUT_DIR/sonar-project.properties \
            -e SONAR_PROJECT_SETTINGS=$OUTPUT_DIR/sonar-project.properties \
            -v "${TARGET_DIR}:/usr/src" \
            sonarsource/sonar-scanner-cli:4 -X
        # -v "${OUTPUT_DIR}:/usr/src/.scannerwork" \
        # -v "${OUTPUT_DIR}:/usr/src" \
        # -v .scannerwork/${PROJECT_NAME}/:/usr/src/.scannerwork/ \
    fi

    ####################################################
    # Remove used files to prevent false positives later
    ####################################################
    rm -f "${TARGET_DIR}/coverage.xml" "${TARGET_DIR}/report.xml"

    ################################################################################################
    # EXPORT SONARQUBE RESULTS TO FILE
    ################################################################################################
    print_subheading "Exporting SonarQube Results to File (project: $repo_name)"

    #############################
    # List all available measures
    #############################
    print_line "Listing all available measures..."
    measures=$(curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X GET \
        "${API_URL}metrics/search" | grep -oP '(?<="key":")[^"]*')

    ##########################################
    # Loop over each measure and fetch metrics
    ##########################################
    for measure in $measures; do
        print_line "Fetching metrics for measure: $measure"
        curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X GET \
            "${API_URL}measures/component?component=${PROJECT_NAME}&metricKeys=${measure}" >> $OUTPUT_FILE
        printf "\n" >> $OUTPUT_FILE
        tail -n 1 $OUTPUT_FILE
    done

    # break
done
print_heading "Program completed successfully"
