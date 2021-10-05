FROM node:11.12.0-alpine as builder
WORKDIR /app
COPY . .

RUN apk update && apk add --update npm
RUN apk add --no-cache python3 py3-pip make g++
RUN npm install node-gyp
RUN npm install
RUN npm run build
RUN echo $PWD
RUN mkdir -p /dist/src
COPY /src/assets /dist/src 


FROM nginx:alpine
VOLUME /var/cache/nginx
COPY --from=builder /app/dist /usr/shared/nginx/html
EXPOSE 80