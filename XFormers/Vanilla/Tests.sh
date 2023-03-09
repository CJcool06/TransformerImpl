
# Runs the tests for the implementation
run_tests() {

    # Convert the Tests notebook to python file
    ipynb-py-convert Tests.ipynb Tests.py

    # Run the tests and return the result
    python3 Tests.py && echo "Success" || echo "Failed"

    # Remove the python file
    rm Tests.py
}