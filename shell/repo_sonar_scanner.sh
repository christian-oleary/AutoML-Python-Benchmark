#!/bin/bash

# shellcheck disable=SC2086

# Usage:
# (ensure python and conda are installed)
# sudo apt-get install zip unzip
# ./shell/sonar_scanner.sh

################################################################################
# This program sets up SonarQube and runs a scan on the project.
################################################################################
# Exit on error
set -e

# Functions to print messages with fancy formatting
print_heading() {
    local message="  $1  "
    local border=$(printf '%*s' "${#message}" '' | tr ' ' '=')
    local color='\033[1;36m' # Light blue color
    local reset='\033[0m'    # Reset color
    echo -e "\n${color}${border}\n${message}\n${border}${reset}\n"
}

print_subheading() {
    local message="  $1  "
    local border=$(printf '%*s' "${#message}" '' | tr ' ' '-')
    local color='\033[1;33m' # Yellow color
    local reset='\033[0m'    # Reset color
    echo -e "\n${color}${border}\n${message}\n${border}${reset}"
}

print_line() {
    local message=$1
    local color='\033[1;32m' # Green color
    local reset='\033[0m'    # Reset color
    echo -e "${color}-> ${message}${reset}"
}

################################################################################
# Environment Variables
################################################################################
print_heading "Environment Variables"
export DOCKER=${DOCKER:-"false"}
export REPOSITORIES_DIR=${REPOSITORIES_DIR:-"repositories"}
export SONAR_HOST_URL=${SONAR_HOST_URL:-"http://localhost:9000"}
export SONAR_LOGIN=${SONAR_LOGIN:-"admin"}
export SONAR_PASSWORD=${SONAR_PASSWORD:-"2c71d75bcs4hq934tdjqngtsojxercm"}
API_URL="${SONAR_HOST_URL}/api/"
print_line "DOCKER=${DOCKER}"
print_line "REPOSITORIES_DIR=${REPOSITORIES_DIR}"
print_line "SONAR_HOST_URL=${SONAR_HOST_URL}"
print_line "SONAR_LOGIN=${SONAR_LOGIN}"
print_line "SONAR_PASSWORD=${SONAR_PASSWORD}"
print_line "API_URL=${API_URL}"

################################################################################
# Change the default SonarQube server admin password
################################################################################
print_heading "Setting admin password"
curl -u ${SONAR_LOGIN}:${SONAR_LOGIN} -X POST \
    "${API_URL}users/change_password?login=${SONAR_LOGIN}&previousPassword=${SONAR_LOGIN}&password=${SONAR_PASSWORD}"
printf "Done\n"

################################################################################
# Download SonarScanner CLI Tool (if not using Docker)
################################################################################
print_heading "SonarScanner CLI Tool Setup"

if [ "$DOCKER" = "false" ]; then
    # Download SonarScanner if it does not exist
    TOOL="sonar-scanner-cli"
    if [ ! -d $TOOL ]; then
        # Download SonarScanner
        print_line "Downloading $TOOL..."
        URL="https://binaries.sonarsource.com/Distribution/${TOOL}/${TOOL}"
        if [ "$(uname)" = "Linux" ]; then
            scanner_url="${URL}-4.6.2.2472-linux.zip"
        else
            scanner_url="${URL}-6.2.1.4610-windows-x64.zip"
        fi
        curl --clobber -sL "${scanner_url}" -o $TOOL.zip

        # Unzip SonarScanner and remove zip file
        print_line "Download finished. Unzipping..."
        unzip -q $TOOL.zip
        rm $TOOL.zip

        # Rename SonarScanner directory
        mv sonar-scanner-* $TOOL
        print_line "SonarScanner downloaded to $(realpath .)/${TOOL}"
    else
        print_line "${TOOL} already exists"
    fi
    # Add SonarScanner to PATH
    PATH=$PATH:$(realpath .)/${TOOL}/bin
else
    print_line "Using SonarScanner Docker Image. Skipping download..."
fi

################################################################################
# List all directories in REPOSITORIES_DIR (default: "repositories")
################################################################################
repositories=$(find ./$REPOSITORIES_DIR -mindepth 1 -maxdepth 1 -type d)
print_line "All repository directories found in '${REPOSITORIES_DIR}':\n$repositories"

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

current_dir=$(pwd)

# Loop over each directory and analyse
for repo_path in $repositories; do
    # Skip if not a directory
    if [ ! -d "$repo_path" ]; then
        echo "Skipping $repo_path (not a directory)"
        continue
    fi

    repo_name=$(basename "$repo_path")

    # Skip if output file already exists
    OUTPUT_FILE="${OUTPUT_DIR}measures.json"
    if [ -f "$OUTPUT_FILE" ]; then
        print_line "Output file $OUTPUT_FILE already exists. Skipping ${repo_name}..."
        continue
    fi

    # Establish the project name
    print_heading "Processing directory: $repo_path"

    # Create a working directory to store sonar-scanner outputs
    OUTPUT_DIR="./results/sca/.scannerwork/${repo_name}/"
    mkdir -p $OUTPUT_DIR

    ############################################################################
    # Environment variables
    ############################################################################
    print_subheading "Environment Variables (project: $repo_name)"

    export PROJECT_BRANCH="master"
    export PROJECT_NAME=$repo_name
    export PROJECTKEY=$PROJECT_NAME
    export SONAR_PROJECTKEY=$PROJECT_NAME
    export SONAR_PROJECT_KEY=$PROJECT_NAME
    export TARGET_DIR="${repo_path}/"

    # Print newly added environment variables
    print_line "PROJECT_BRANCH=${PROJECT_BRANCH}"
    print_line "PROJECT_NAME=${PROJECT_NAME}"
    print_line "PROJECTKEY=${PROJECTKEY}"
    print_line "SONAR_PROJECTKEY=${SONAR_PROJECTKEY}"
    print_line "SONAR_PROJECT_KEY=${SONAR_PROJECT_KEY}"
    print_line "TARGET_DIR=${TARGET_DIR}"

    ############################################################################
    # Running Python tests
    ############################################################################
    print_subheading "Running Python Tests (project: $repo_name)"
    cd $repo_path

    # Check if coverage.xml exists
    if [ ! -f "coverage.xml" ]; then
        # Create a new conda environment and install dependencies
        if ! conda env list | grep -q "$repo_name"; then
            print_line "Setting up conda environment..."
            conda create -y -n $repo_name python=3.10
        else
            print_line "Conda environment $repo_name found"
        fi

        # Install dependencies from requirements.txt
        # List of possible requirements files
        requirements_files=(
            "requirements.txt"
            "requirements-dev.txt"
            "requirements-test.txt"
        )

        # Install dependencies from each requirements file if it exists
        for req_file in "${requirements_files[@]}"; do
            if [ -f "$req_file" ]; then
                print_line "Installing dependencies from $req_file..."
                conda run -n $repo_name python -m pip install -r "$req_file"
            fi
        done

        # Install pytest and pytest-cov
        print_line "Installing pytest and pytest-cov..."
        conda run -n $repo_name python -m pip install pytest pytest-cov

        # Run tests
        print_line "Running pytest..."
        conda run -n $repo_name pytest --cov-report xml --cov=. || true
        cp coverage.xml $OUTPUT_DIR

        # print_line "Deleting conda environment..."
        # conda remove -n $repo_name --all -y
        conda run -n $repo_name pytest --cov-report xml --cov=. || true
    else
        print_line "coverage.xml found. Skipping tests..."
    fi

    cd $current_dir

    ############################################################################
    # Create/replace user access token
    ############################################################################
    print_subheading "Generating user access token (project: $repo_name)"

    # Revoke old token if it exists
    print_line "Revoking old token..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
        "${API_URL}user_tokens/revoke?name=${PROJECT_NAME}&login=${SONAR_LOGIN}"

    # Remove token.json if it exists
    rm -f token.json

    # Generate new token
    print_line "Generating new user access token..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
        -X POST "${API_URL}user_tokens/generate?name=${PROJECT_NAME}&login=${SONAR_LOGIN}" -o token.json

    # Read token from file and export as a variable
    SONAR_TOKEN=$(grep -oP '(?<="token":")[^"]*' token.json)
    export SONAR_TOKEN
    print_line "SONAR_TOKEN: $SONAR_TOKEN"

    # List tokens
    # print_line "User tokens in SonarQube:\n"
    # curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} \
    #     -X POST "${API_URL}user_tokens/search"  | python -m json.tool

    ############################################################################
    # Create sonar-project.properties file
    ############################################################################
    print_subheading "Updating sonar-project.properties file (project: $repo_name)"
    ABSOLUTE_PATH=$(realpath $TARGET_DIR)
    cat <<EOL >$OUTPUT_DIR/sonar-project.properties
sonar.projectKey=$SONAR_PROJECT_KEY
sonar.projectName=$SONAR_PROJECT_KEY
sonar.sources=$ABSOLUTE_PATH
sonar.host.url=$SONAR_HOST_URL
sonar.login=$SONAR_LOGIN
sonar.password=$SONAR_PASSWORD
sonar.token=$SONAR_TOKEN
sonar.language=py
sonar.python.coverage.reportPaths=coverage.xml
sonar.scm.disabled=true
sonar.working.directory=$OUTPUT_DIR
EOL
    cat "$OUTPUT_DIR/sonar-project.properties"
    # sonar.python.version=3.x
    # sonar.sourceEncoding=UTF-8

    ############################################################################
    # Create projects. Delete old projects if they already exists.
    ############################################################################
    print_subheading "Set Up Projects (project: $repo_name)"

    # Delete project if it already exists
    print_line "Deleting old project..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST "${API_URL}projects/delete?project=${PROJECT_NAME}&branch=${PROJECT_BRANCH}"

    # Create new project
    print_line "Creating new project..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
        "${API_URL}projects/create?name=${PROJECT_NAME}&project=${PROJECT_NAME}&projectKey=${PROJECT_KEY}&token=${SONAR_TOKEN}&branch=${BRANCH}"
    printf "\n"

    # List projects
    print_line "Projects in SonarQube:\n"
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
        -X POST "${API_URL}components/search?qualifiers=TRK" | python -m json.tool

    ############################################################################
    # Run SonarScanner
    ############################################################################
    print_subheading "Running SonarScanner (project: $repo_name)"

    # Run SonarScanner CLI tool
    if [ "${DOCKER}" = "false" ]; then
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
        print_line "Running SonarScanner Docker image..."
        # Run SonarScanner Docker image
        docker run --network=host --rm \
            -e PROJECT_SETTINGS=$OUTPUT_DIR/sonar-project.properties \
            -e SONAR_PROJECT_SETTINGS=$OUTPUT_DIR/sonar-project.properties \
            -v "${TARGET_DIR}:/usr/src" \
            -v "${OUTPUT_DIR}:/usr/src" \
            -v "${OUTPUT_DIR}:/usr/src/.scannerwork" \
            sonarsource/sonar-scanner-cli:4 -X
        # -v .scannerwork/${PROJECT_NAME}/:/usr/src/.scannerwork/ \
    fi

    ################################################################################
    # Export results from SonarQube to file
    ################################################################################
    print_subheading "Exporting SonarQube Results to File (project: $repo_name)"

    # List all available measures
    print_line "Listing all available measures..."
    measures=$(curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X GET \
        "${API_URL}metrics/search" | grep -oP '(?<="key":")[^"]*')

    # Loop over each measure and fetch metrics
    for measure in $measures; do
        print_line "Fetching metrics for measure: $measure"
        curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X GET \
            "${API_URL}measures/component?component=${PROJECT_NAME}&metricKeys=${measure}" \
            >> $OUTPUT_FILE
        printf "\n" >> $OUTPUT_FILE
        tail -n 1 $OUTPUT_FILE
    done

    # break
done
print_heading "Program completed successfully"
