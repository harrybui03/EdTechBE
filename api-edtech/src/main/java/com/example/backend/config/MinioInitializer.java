package com.example.backend.config;

import io.minio.BucketExistsArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.SetBucketPolicyArgs;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
public class MinioInitializer {

    private final MinioClient minioClient;

    @Value("${minio.bucket.name}")
    private String bucketName;


    @PostConstruct
    public void initialize() {
        try {
            ensureBucketExistsAndSetPolicy();
        } catch (Exception e) {
            log.error("Failed to initialize MinIO bucket '{}'. Please check MinIO connection and configuration.", bucketName, e);
        }
    }

    private void ensureBucketExistsAndSetPolicy() throws Exception {
        boolean bucketExists = minioClient.bucketExists(BucketExistsArgs.builder().bucket(bucketName).build());

        if (!bucketExists) {
            minioClient.makeBucket(MakeBucketArgs.builder().bucket(bucketName).build());
            log.info("MinIO bucket '{}' created successfully.", bucketName);
        } else {
            log.info("MinIO bucket '{}' already exists.", bucketName);
        }

    }
}