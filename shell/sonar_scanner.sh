#!/bin/bash

# shellcheck disable=SC2086

# sudo apt-get install zip unzip

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
        print_line "Download finished"

        # Unzip SonarScanner and remove zip file
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

# Loop over each directory and analyse
for repo_path in $repositories; do
    print_heading "Processing directory: $repo_path"

    # Skip if not a directory
    if [ ! -d "$repo_path" ]; then
        echo "Skipping $repo_path (not a directory)"
        continue
    fi

    # Establish the project name
    repo_name=$(basename "$repo_path")

    # Create a working directory to store sonar-scanner outputs
    OUTPUT_DIR=".scannerwork/${repo_name}/"
    mkdir -p $OUTPUT_DIR

    ############################################################################
    # Environment variables
    ############################################################################
    print_subheading "Environment Variables"

    export PROJECT_BRANCH="main"
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
    # Create/replace user access token
    ############################################################################
    print_subheading "Generating user access token"

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
    print_subheading "Updating sonar-project.properties file"
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
sonar.python.version=3.x
sonar.sourceEncoding=UTF-8
sonar.python.coverage.reportPaths=coverage.xml
sonar.working.directory=$OUTPUT_DIR
EOL
    cat "$OUTPUT_DIR/sonar-project.properties"

    ############################################################################
    # Create projects. Delete old projects if they already exists.
    ############################################################################
    print_subheading "Set Up Projects"

    # Delete project if it already exists
    print_line "Deleting old project..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST "${API_URL}projects/delete?project=${PROJECT_NAME}&branch=${PROJECT_BRANCH}"

    # Create new project
    print_line "Creating new project..."
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s -X POST \
        "${API_URL}projects/create?name=${PROJECT_NAME}&project=${PROJECT_NAME}&projectKey=${PROJECT_KEY}&token=${SONAR_TOKEN}&branch=${BRANCH}"
    printf "\n"

    # # List projects
    # print_line "Projects in SonarQube:\n"
    # curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
    #     -X POST "${API_URL}components/search?qualifiers=TRK" | python -m json.tool

    ############################################################################
    # Run SonarScanner
    ############################################################################
    print_subheading "Running SonarScanner"

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
            sonarsource/sonar-scanner-cli:4 -X
        # -v .scannerwork/${PROJECT_NAME}/:/usr/src/.scannerwork/ \
    fi

    ################################################################################
    # Export results from SonarQube to file
    ################################################################################
    print_subheading "Exporting SonarQube Results to File"

    # Specify output file
    OUTPUT_FILE="${OUTPUT_DIR}measures.component_tree.json"
    rm -rf $OUTPUT_FILE

    # Export measures/component_tree
    curl -u ${SONAR_LOGIN}:${SONAR_PASSWORD} -s \
        -X GET "${API_URL}measures/component_tree?component=${PROJECT_NAME}&metricKeys=ncloc,complexity,bugs,vulnerabilities,code_smells,duplicated_lines_density&strategy=children" \
        > $OUTPUT_FILE

    # Print results
    print_line "SonarQube results (${OUTPUT_FILE}):"
    cat $OUTPUT_FILE | python -m json.tool

    # break
done

################################################################################
print_heading "Program completed successfully"
