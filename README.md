# dataset-enrichment-with-llms
 Excerpt of the original code put up for display

Talk:

https://events.xebia.com/conference-data-mass-summit-2023?ref=APAC

https://summit.datamass.io/

https://www.youtube.com/watch?v=hJeMtABYkEQ

# Set-up
Before running the project, make sure to update the `.env` file with your secret OpenAI API key. You can obtain your API key from the [OpenAI website](https://platform.openai.com/api-keys).

You can test the CLI by running the following script:
```bash
./bin/cli.sh --help
```

# Run commands:

```bash
docker-compose build
bin/cli.sh enrich-house "1cae07e6" "mock"
bin/pylint.sh
bin/pytest.sh
```

# Package versions:

OpenAI has been updated to the latest version following the [official guide](https://github.com/openai/openai-python/discussions/742). Some outdated packages remain:

- cloudpickle==3.1.0 is available (you have 2.2.1)
- docker==7.1.0 is available (you have 6.1.3)
- firebase-admin==6.6.0 is available (you have 6.1.0)
- Flask==3.1.0 is available (you have 2.3.3)
- greenlet==3.1.1 is available (you have 2.0.1)
- grpcio-status==1.69.0 is available (you have 1.62.3)
- gunicorn==23.0.0 is available (you have 21.2.0)
- importlib-metadata==8.5.0 is available (you have 6.11.0)
- mlflow==2.19.0 is available (you have 2.7.1)
- numpy==2.2.1 is available (you have 1.26.4)
- packaging==24.2 is available (you have 23.2)
- pandas==2.2.3 is available (you have 1.5.3)
- playwright==1.49.1 is available (you have 1.32.1)
- protobuf==5.29.2 is available (you have 4.25.5)
- pyarrow==18.1.0 is available (you have 13.0.0)
- pydantic==2.10.4 is available (you have 2.10.3)
- pydantic_core==2.27.2 is available (you have 2.27.1)
- pyee==12.1.1 is available (you have 9.0.4)
- pytz==2024.2 is available (you have 2023.4)
- typer==0.15.1 is available (you have 0.9.0)