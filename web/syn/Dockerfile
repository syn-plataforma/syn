##### Stage 1

FROM node:alpine as builder

WORKDIR /app

# Copy project files to the docker image
COPY . .


# if you prefer npm, replace the above command with
RUN npm install @angular/cli@latest -g

# install packages
RUN npm install

# FOR NPM
# npm install

# SET ENVIRONMENT VARIABLES
ENV ENVIRONMENT=prod
ENV apiUri="https://api.syn.altgovrd.com/v1"
ENV loginUri="https://api.syn.altgovrd.com/v1/users/login"
ENV apiHost="https://google.com"


# Build Angular Application in Production
RUN ng build --prod

#### STAGE 2
#### Deploying the application
FROM nginx:alpine
VOLUME /var/cache/nginx

# Copy the build files from the project
COPY --from=builder /app/dist/syn /usr/share/nginx/html
# Copy Nginx Files


# EXPOSE Port 80
EXPOSE 80
