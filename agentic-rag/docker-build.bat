@echo off
REM Build script for Agentic RAG Docker image (Windows)

setlocal enabledelayedexpansion

set IMAGE_NAME=agentic-rag
set IMAGE_TAG=%1
if "%IMAGE_TAG%"=="" set IMAGE_TAG=latest
set FULL_IMAGE_NAME=%IMAGE_NAME%:%IMAGE_TAG%

echo Building Docker image: %FULL_IMAGE_NAME%

REM Build the image
docker build -t %FULL_IMAGE_NAME% .

if %ERRORLEVEL% EQU 0 (
    echo ✅ Docker image built successfully: %FULL_IMAGE_NAME%
    
    REM Optionally tag as latest if not already
    if not "%IMAGE_TAG%"=="latest" (
        docker tag %FULL_IMAGE_NAME% %IMAGE_NAME%:latest
        echo ✅ Tagged as %IMAGE_NAME%:latest
    )
    
    echo.
    echo To run the container:
    echo   docker run -p 8002:8002 --env-file .env %FULL_IMAGE_NAME%
    echo.
    echo Or use docker-compose:
    echo   docker-compose up -d
) else (
    echo ❌ Build failed!
    exit /b 1
)

