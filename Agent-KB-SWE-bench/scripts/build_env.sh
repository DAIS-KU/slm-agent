
conda install -c conda-forge --yes nodejs=22
conda install -c conda-forge --yes poetry=1.8
# 验证nodejs
node -v
npm -v
# 验证poetry
poetry --version
make build
