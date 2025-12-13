# Stage 1: The Builder Stage
# Use a Gradle image with JDK 17 to build the application
FROM gradle:8.5-jdk17-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy the Gradle wrapper files
COPY gradlew .
COPY gradle ./gradle

# Copy the build configuration files
COPY build.gradle .
COPY settings.gradle .

# Grant execute permission to the Gradle wrapper
RUN chmod +x ./gradlew

# Download dependencies first to leverage Docker's layer caching
# This step will only be re-run if the build.gradle or settings.gradle files change
RUN ./gradlew dependencies

# Copy the rest of the source code
COPY src ./src

# Build the application and create the executable JAR
RUN ./gradlew bootJar

# ---

# Stage 2: The Final Stage
# Use a slim JRE image for a smaller footprint
FROM eclipse-temurin:17-jre-jammy

# Set the working directory
WORKDIR /app

# Copy the built JAR from the builder stage
# The JAR file will be in build/libs/ and will have a name based on the project
COPY --from=builder /app/build/libs/*.jar app.jar

# Expose the port the application will run on (as defined in application.yml and .env)
EXPOSE 8000

# The command to run the application when the container starts
ENTRYPOINT ["java", "-jar", "app.jar"]
