---
title: Torq Performance
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8080
pinned: false
short_description: Torq Performance Dashboard
license: apache-2.0
---

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
    mkdir data
    sudo chown 1000:1000 data
    docker compose up
    ```

4. The application now available on http://localhost:8080

5. To send the results a test run to the application set the following environment variable:

    ```
    export TORQ_PERF_SERVER=http://localhost:8080
    ```

## Deploy on HuggingFace

This application can be deployed as a private HuggingFace space, to do so just push this folder to the space.

Do not deploy it as a public application as it is not sufficiently secured (especially the REST API).

The space needs a variables in its settings:

- ``DJANGO_ALLOWED_HOSTS`` : organization_name-space_name.hf.space

And a secret:

- ``ADMIN_PASSWORD``: password for the admin user of the admin interface

In order to persist the data across rebuilds of the space you need to enable persistent storage of the space.

To use the deplyed application from pytest you first need to create a Token with read access to the space and then
set two environment variables:

```
export TORQ_PERF_SERVER=https://organization_name-space_name.hf.space
export TORQ_PERF_SERVER_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXX
```