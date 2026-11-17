FROM python:3.12-slim
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker/compose-bin:v2.29.7 /docker-compose /usr/libexec/docker/cli-plugins/docker-compose
RUN pip install --no-cache-dir harbor==0.26.0
