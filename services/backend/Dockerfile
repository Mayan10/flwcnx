FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

# Generate Prisma client
RUN npx prisma generate

RUN npm run build

EXPOSE 3001

# Run migrations and start the server
CMD ["sh", "-c", "npx prisma migrate deploy && npm start"]
