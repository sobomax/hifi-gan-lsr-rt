#!/bin/sh

set -e

capture_and_compress_git_info() {
  # Ensure we have the necessary arguments
  if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [path_to_git_repo] [path_to_output_txt]"
    exit 1
  fi

  local REPO_PATH="$1"
  local OUTPUT_TXT="$2"
  
  # Capture git information
  local GIT_CMD="git -C ${REPO_PATH}"
  REV=$(${GIT_CMD} rev-parse HEAD)
  BRANCH=$(${GIT_CMD} rev-parse --abbrev-ref HEAD)
  REMOTE=$(${GIT_CMD} config --get remote.origin.url)
  DIFF=$(${GIT_CMD} diff)

  # Combine the information
  echo "Revision: $REV" > "${OUTPUT_TXT}"
  echo "Branch: $BRANCH" >> "${OUTPUT_TXT}"
  echo "Remote: $REMOTE" >> "${OUTPUT_TXT}"
  echo -e "\nDiff:\n$DIFF" >> "${OUTPUT_TXT}"

  # Get the directory of OUTPUT_TXT to store the compressed file there
  OUTPUT_DIR=$(dirname "${OUTPUT_TXT}")
  OUTPUT_BASENAME=$(basename "${OUTPUT_TXT}" .txt)

  # Compress the information using lzma
  tar --lzma -cf "${OUTPUT_DIR}/${OUTPUT_BASENAME}.tar.lzma" "${OUTPUT_TXT}"
}

# Call the function with provided arguments
capture_and_compress_git_info "$1" "$2"
