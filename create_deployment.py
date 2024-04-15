from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/surfrider325/PublicProjects.git",
        entrypoint="Measurement_Run.py:Measurement_Run",
    ).deploy(
        name="Measurement_Run",
        work_pool_name="main",
        cron="0 1 * * *",
    )