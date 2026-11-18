# The agentfault sandbox: one image for the agent's workspace and for the grader.
# Pinned to the Python and pytest the local backend uses, so both backends print
# the same test output. The corpus needs nothing beyond pytest.
FROM python:3.11.7-slim
RUN pip install --no-cache-dir pytest==7.4.0
WORKDIR /repo
