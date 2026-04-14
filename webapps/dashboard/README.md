# Torq Performance Dashboard

This is a web application that can receive test results from the torq-compiler test suite and display them.

## How to run locally

To run locally this application:

1. Install docker compose v2:

    ```
    sudo apt-get install docker-compose-v2
    ```

2. Build the application:

    ```
    docker compose build
    ```

3. Start the application:

    ```
    docker compose up
    ```

4. The application now available on http://localhost:8080

5. To send the results a test run to the application set the following environment variable:

    ```
    export TORQ_PERF_SERVER=http://localhost:8080
    ```
