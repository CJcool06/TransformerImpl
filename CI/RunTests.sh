#!/bin/bash

cd ../XFormers

found_error="false"

for dir in */ ; do
    # Removes trailing slash
    dir=${dir%/}

    echo "Running tests inside directory: $dir"

    # Move into directory
    cd $dir

    # Source the test script from the directory
    source Tests.sh

    # Run the test and store the result
    result=$(run_tests)

    # Print the result
    echo "Result: $result"

    # Indicate that a test failed
    if [ "$result" = "Failed" ]; then
        found_error="true"
    fi

    # Exit directory
    cd ..

done

# If one of the tests failed, terminate and indicate error to cause the CI to fail
if [ "$found_error" = "true" ]; then
    exit 1
fi