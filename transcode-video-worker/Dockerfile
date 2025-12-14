# Stage 1: The Builder Stage
# We use a specific Go version on Debian (bullseye) which has apt-get.
FROM golang:1.24-bullseye AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy go.mod and go.sum files to download dependencies first
# This leverages Docker's layer caching. Dependencies will only be re-downloaded if these files change.
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of your source code
COPY . .

# Build the Go application.
# The CGO_ENABLED=0 flag creates a statically linked binary, which is ideal for containers.
# We specify '.' to build the main package in the current directory.
RUN CGO_ENABLED=0 GOOS=linux go build -o /main .

# ---

# Stage 2: The Final Stage
# We use a slim Debian image to keep the final image size small.
FROM debian:bullseye-slim

# Install FFmpeg and other potential dependencies. ca-certificates is needed for HTTPS requests.
# We clean up the apt cache to reduce image size.
RUN apt-get update && apt-get install -y ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /main .


# Expose the port your application runs on (change 8080 if necessary)
EXPOSE 8080

# The command to run your application when the container starts
CMD ["./main", "server"]