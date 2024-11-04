#!/bin/bash

# Usage: ./shell/repo_line_counts.sh

# Exit on error
set -e

# Functions to print messages with fancy formatting
print_heading() {
    local message="  $1  "
    local border
    border=$(printf '%*s' "${#message}" '' | tr ' ' '=')
    local color='\033[1;36m' # Light blue color
    local reset='\033[0m'    # Reset color
    echo -e "\n${color}${border}\n${message}\n${border}${reset}\n"
}

# Ensure repositories have been cloned
print_heading Cloning repositories...
sh ./shell/repo_clone_or_pull.sh
echo "Done"

# Create directory for results
print_heading "Creating directory for results..."
results_dir="results/sca/line_counts/"
mkdir -p "${results_dir}"
echo "Results directory: ${results_dir}"

# List repository directories
print_heading "Finding repository directories..."
repo_dir="repositories/" # Should exist from repo_clone_or_pull.sh
mapfile -t repos < <(ls "${repo_dir}")
echo "Repositories found: ${repos[*]}"

# Create output file
print_heading "Creating output file..."
output_file_totals="${results_dir}line_counts_totals.txt"
output_file_verbose="${results_dir}line_counts_verbose.txt"
rm -rf "${output_file_verbose}" "${output_file_totals}"
touch "${output_file_verbose}" "${output_file_totals}"
echo "Totals file: ${output_file_totals}"
echo "Verbose file: ${output_file_verbose}"

# Count lines of code in each repository
cd "${repo_dir}" || exit
for repo in "${repos[@]}"; do
    print_heading "Counting lines of code for ${repo}"
    cd "${repo}"
    # Count lines of code
    line_count=$(git ls-files | grep "\.py" | xargs wc -l | grep " total")
    echo "${line_count}"
    echo "${repo}" >> "../../${output_file_totals}"
    echo "${repo}" >> "../../${output_file_verbose}"
    echo "${line_count}" >> "../../${output_file_totals}"
    echo "${line_count}" | grep " total" >> "../../${output_file_verbose}"
    cd ..
done

cd ..
echo "Line counting finished"
