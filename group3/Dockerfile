FROM python:3.12-slim
WORKDIR /workspace
COPY requirements.txt ./root-requirements.txt
COPY group3/requirements.txt ./group3-requirements.txt
RUN pip install --no-cache-dir -r root-requirements.txt -r group3-requirements.txt
COPY group3/api ./api
COPY group3/contracts ./contracts
COPY backend ./backend
