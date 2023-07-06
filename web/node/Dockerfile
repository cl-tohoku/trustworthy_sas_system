FROM node:latest
WORKDIR /mnt/web/client
COPY ./client/package.json /mnt/web/client
RUN npm install -D create-react-app
RUN npm install -D tailwindcss postcss autoprefixer
COPY ./client /mnt/web/client
RUN cd /mnt/web/client && npm install
