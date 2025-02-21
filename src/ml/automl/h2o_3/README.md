# H2O Server

Using H2O's Python library h2o-3 requires that a H2O server is running.

```bash
cd ./src/ml/automl/h2o_3/server  # (if starting from project base dir)

url=$(curl http://h2o-release.s3.amazonaws.com/h2o/latest_stable)
# "latest" contains the URL of the latest stable H2O build
ls h2o.zip || curl --output h2o.zip -LO $url

# Running locally:
unzip -d h2o h2o.zip               # Unzip
jar_path=$(find . -name 'h2o.jar') # Find .jar
java -Xmx4g -jar $jar_path         # Run

# Docker
docker rm h2o-server-test; docker run -itd -p 54321:54321 --name h2o-server-test christianoleary/h2o-server

# Using docker-compose
docker compose up -d # Start server
docker compose down  # Stop server
```
