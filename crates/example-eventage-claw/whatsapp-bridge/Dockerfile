# syntax=docker/dockerfile:1
FROM node:20-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY bridge.js .

EXPOSE 3001

CMD ["node", "bridge.js"]
