package com.example.backend.config;

import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MinioConfig {

    @Value("${minio.api.url}")
    private String apiUrl;

    @Value("${minio.access.key}")
    private String accessKey;

    @Value("${minio.access.secret}")
    private String secretKey;

    @Bean
    public MinioClient minioClient() {
        try {
            String endpoint = this.apiUrl;
            String host;
            int port;
            boolean secure = false;

            if (endpoint.startsWith("https://")) {
                endpoint = endpoint.substring(8);
                secure = true;
            } else if (endpoint.startsWith("http://")) {
                endpoint = endpoint.substring(7);
            }

            String[] parts = endpoint.split(":");
            host = parts[0];
            if (parts.length > 1) {
                port = Integer.parseInt(parts[1]);
            } else {
                port = secure ? 443 : 9000;
            }

            return MinioClient.builder()
                    .endpoint(host, port, secure)
                    .credentials(accessKey, secretKey)
                    .build();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize Minio client with URL: '" + this.apiUrl + "'", e);
        }
    }
}